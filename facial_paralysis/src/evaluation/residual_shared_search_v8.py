"""Participant-disjoint evaluation for the shared-core residual router v8."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.medically_gated_shared_search_v2 import MedicalSharedDatasetV2,_model_inputs,_scaled,_tensor
from src.evaluation.shared_clinical_encoder_v1 import SOURCES,fit_clinical_scaler,participant_disjoint_folds,source_class_balanced_weights
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.residual_shared_router_v8 import ResidualCandidateV8,ResidualSharedRouterV8,candidate_registry_v8

_SOURCE_TASK_CODE={source:index for index,source in enumerate(SOURCES)}


def _immutable(values):
    x=np.ascontiguousarray(values); return np.frombuffer(x.tobytes(),dtype=x.dtype).reshape(x.shape)


@dataclass(frozen=True)
class ResidualEvaluationV8:
    probabilities:np.ndarray
    metrics:dict[str,dict[str,float]]
    model_fits:int
    threshold:float
    shared_gradient_sources:tuple[str,...]


def evaluate_residual_candidate(dataset:MedicalSharedDatasetV2,candidate:ResidualCandidateV8,*,epochs:int,adapter_epochs:int=0,n_splits:int=6,seed:int=0,device:str="cpu"):
    if type(dataset) is not MedicalSharedDatasetV2 or type(candidate) is not ResidualCandidateV8 or candidate not in candidate_registry_v8() or isinstance(epochs,bool) or not isinstance(epochs,int) or epochs<1 or isinstance(adapter_epochs,bool) or not isinstance(adapter_epochs,int) or adapter_epochs<0: raise ValueError("invalid v8 evaluation")
    runtime=torch.device(device); base=dataset.base; folds=participant_disjoint_folds(base,n_splits=n_splits)
    probabilities=np.full(len(base.labels),np.nan); covered=set()
    for fold_index,(train,held) in enumerate(folds):
        local_seed=seed*1009+fold_index; torch.manual_seed(local_seed)
        if runtime.type=="cuda": torch.cuda.manual_seed_all(local_seed)
        scaler=fit_clinical_scaler(base,train); original=_scaled(base.clinical_original,scaler.mean,scaler.scale); mirrored=_scaled(base.clinical_mirrored,scaler.mean,scaler.scale)
        model=ResidualSharedRouterV8(candidate).to(runtime); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
        sources=tuple(base.sources[i] for i in train); weights=_tensor(source_class_balanced_weights(base.labels[train],sources).astype(np.float32),runtime)
        labels=_tensor(base.labels[train].astype(np.float32),runtime); tasks=_tensor(np.asarray([_SOURCE_TASK_CODE[x] for x in sources],dtype=np.int64),runtime)
        inputs=_model_inputs(dataset,original,mirrored,train,runtime)
        if fold_index==0:
            for source in SOURCES:
                local=torch.tensor([i for i,x in enumerate(sources) if x==source],dtype=torch.long,device=runtime); model.zero_grad(set_to_none=True)
                local_inputs=tuple(x.index_select(0,local) for x in inputs); tokens=model.shared_action_tokens(*local_inputs)
                loss=F.binary_cross_entropy_with_logits(model.routed_logits(tokens,local_inputs[-2],tasks.index_select(0,local)),labels.index_select(0,local)); loss.backward()
                if model.base.backbone.clinical_encoder[0].weight.grad is None or model.base.backbone.patient_projection.weight.grad is None: raise RuntimeError("source failed shared v8 gradient audit")
                covered.add(source)
        for _ in range(epochs):
            model.train(); optimizer.zero_grad(set_to_none=True); tokens=model.shared_action_tokens(*inputs)
            common=model.base.endpoint_embedding(tokens,inputs[-2],tasks); endpoint=model.adapt_endpoint(common,tasks); universal=model.base.universal_embedding(tokens,inputs[-2])
            task_logits=model.base.task_logits_from_embedding(endpoint,tasks); universal_logits=model.base.universal_head(universal).squeeze(-1); routed=.75*task_logits+.25*universal_logits
            losses=F.binary_cross_entropy_with_logits(routed,labels,reduction="none")+.5*F.binary_cross_entropy_with_logits(universal_logits,labels,reduction="none")
            torch.sum(losses*weights).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if adapter_epochs:
            for name,parameter in model.named_parameters():
                parameter.requires_grad_(
                    name.startswith("adapters")
                    or name.startswith("base.task_queries")
                    or name.startswith("base.backbone.task_heads")
                )
            endpoint_parameters=tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
            endpoint_optimizer=torch.optim.AdamW(endpoint_parameters,lr=1e-3,weight_decay=1e-3)
            model.eval()
            for _ in range(adapter_epochs):
                endpoint_optimizer.zero_grad(set_to_none=True); tokens=model.shared_action_tokens(*inputs)
                common=model.base.endpoint_embedding(tokens,inputs[-2],tasks); endpoint=model.adapt_endpoint(common,tasks)
                task_logits=model.base.task_logits_from_embedding(endpoint,tasks)
                universal=model.base.universal_embedding(tokens,inputs[-2]); universal_logits=model.base.universal_head(universal).squeeze(-1)
                routed=.75*task_logits+.25*universal_logits
                torch.sum(F.binary_cross_entropy_with_logits(routed,labels,reduction="none")*weights).backward()
                torch.nn.utils.clip_grad_norm_(endpoint_parameters,1.0); endpoint_optimizer.step()
        held_inputs=_model_inputs(dataset,original,mirrored,held,runtime); held_tasks=_tensor(np.asarray([_SOURCE_TASK_CODE[base.sources[i]] for i in held],dtype=np.int64),runtime)
        model.eval()
        with torch.no_grad():
            tokens=model.shared_action_tokens(*held_inputs); probabilities[held]=torch.sigmoid(model.routed_logits(tokens,held_inputs[-2],held_tasks)).cpu().numpy()
        del model,optimizer
    if not np.isfinite(probabilities).all() or covered!=set(SOURCES): raise RuntimeError("v8 evaluation incomplete")
    metrics={}
    for source in SOURCES:
        selected=np.asarray([x==source for x in base.sources]); metrics[source]=binary_metrics(base.labels[selected],probabilities[selected])
    return ResidualEvaluationV8(_immutable(probabilities),metrics,len(folds),.5,SOURCES)


def rank_residual_results(results):
    expected={x.candidate_id for x in candidate_registry_v8()}
    if set(results)!=expected: raise ValueError("v8 ranking incomplete")
    def key(item):
        m=results[item].metrics; b=[m[s]["balanced_accuracy"] for s in SOURCES]; q=[m[s]["specificity"] for s in SOURCES]; a=[m[s]["auroc"] for s in SOURCES]
        return (-min(b),-min(q),-min(a),-float(np.mean(b)),item)
    return tuple(sorted(expected,key=key))


__all__=["ResidualEvaluationV8","evaluate_residual_candidate","rank_residual_results"]
