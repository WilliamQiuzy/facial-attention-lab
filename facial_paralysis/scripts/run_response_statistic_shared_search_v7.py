#!/usr/bin/env python3
"""Run the response-statistic shared-router v7 search on H200."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dense_clinical_shared_encoder_v1 as v1_runner  # noqa: E402
from scripts import run_medically_gated_shared_search_v2 as v2_runner  # noqa: E402
from src.evaluation.response_statistic_shared_search_v7 import (  # noqa: E402
    evaluate_response_statistic_candidate, rank_response_statistic_results,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.models.response_statistic_shared_router_v7 import candidate_registry_v7  # noqa: E402


def _parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root",type=Path,required=True)
    parser.add_argument("--reviewed-identity-manifest",type=Path,required=True)
    parser.add_argument("--review-ledger",type=Path,required=True)
    parser.add_argument("--split-registry",type=Path,required=True)
    parser.add_argument("--neuroface-cache",type=Path,required=True)
    parser.add_argument("--neuroface-collection-sha256",required=True)
    parser.add_argument("--neuroface-manifest",type=Path,required=True)
    parser.add_argument("--neuroface-manifest-sha256",required=True)
    parser.add_argument("--meei-cache",type=Path,required=True)
    parser.add_argument("--meei-collection-sha256",required=True)
    parser.add_argument("--meei-manifest",type=Path,required=True)
    parser.add_argument("--meei-manifest-sha256",required=True)
    parser.add_argument("--phase",choices=("screen","confirm"),required=True)
    parser.add_argument("--candidate-ids",nargs="*",default=None)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--folds",type=int,default=6)
    return parser


def _rank(ids,evaluations):
    def key(item):
        m=evaluations[item]; b=[m[s]["balanced_accuracy"] for s in SOURCES]
        q=[m[s]["specificity"] for s in SOURCES]; a=[m[s]["auroc"] for s in SOURCES]
        return (-min(b),-min(q),-min(a),-float(np.mean(b)),item)
    return tuple(sorted(ids,key=key))


def _implementation_sha256():
    paths=(Path(__file__).resolve(),PROJECT_ROOT/"src/evaluation/response_statistic_shared_search_v7.py",PROJECT_ROOT/"src/models/response_statistic_shared_router_v7.py",PROJECT_ROOT/"src/preprocessing/shared_response_statistics_v7.py",PROJECT_ROOT/"scripts/run_medically_gated_shared_search_v2.py",PROJECT_ROOT/"src/evaluation/medically_gated_shared_search_v2.py",PROJECT_ROOT/"src/models/medically_gated_shared_encoder_v2.py",PROJECT_ROOT/"scripts/run_dense_clinical_shared_encoder_v1.py",PROJECT_ROOT/"src/evaluation/shared_clinical_encoder_v1.py",PROJECT_ROOT/"src/preprocessing/shared_clinical_tokens_v1.py",PROJECT_ROOT/"src/preprocessing/clinical_landmarks.py",PROJECT_ROOT/"src/preprocessing/generalization_110d.py",PROJECT_ROOT/"src/preprocessing/trajectory_features.py")
    digest=hashlib.sha256()
    for path in paths:
        payload=path.read_bytes(); digest.update(str(path.relative_to(PROJECT_ROOT)).encode()); digest.update(b"\0"); digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def main():
    args=_parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!="NVIDIA H200": raise RuntimeError("v7 requires NVIDIA H200")
    registry=candidate_registry_v7(); lookup={x.candidate_id:x for x in registry}; expected=tuple(lookup)
    ids=expected if args.candidate_ids is None else tuple(args.candidate_ids)
    if len(ids)!=len(set(ids)) or any(x not in lookup for x in ids) or (args.phase=="screen" and ids!=expected) or (args.phase=="confirm" and len(ids)!=2): raise ValueError("v7 candidate phase drifted")
    if args.epochs!=20 or args.folds!=6 or (args.phase=="screen" and args.seed!=0) or (args.phase=="confirm" and args.seed not in (1,2)): raise ValueError("v7 training protocol drifted")
    started=time.monotonic(); palsy,pcommit=v1_runner._load_palsynet(args)
    neuroface,ncommit=v1_runner._load_dense_profile(profile="neuroface",cache_root=args.neuroface_cache,collection_sha256=args.neuroface_collection_sha256,manifest_path=args.neuroface_manifest,manifest_sha256=args.neuroface_manifest_sha256)
    meei,mcommit=v1_runner._load_dense_profile(profile="meei",cache_root=args.meei_cache,collection_sha256=args.meei_collection_sha256,manifest_path=args.meei_manifest,manifest_sha256=args.meei_manifest_sha256)
    dataset=v2_runner.pack_participant_bags_v2((*palsy,*neuroface,*meei)); counts={s:sum(x==s for x in dataset.base.sources) for s in SOURCES}
    if counts!={"palsynet":38,"neuroface":36,"meei":56}: raise ValueError("v7 participant counts drifted")
    results={}; evaluations={}
    for item in ids:
        result=evaluate_response_statistic_candidate(dataset,lookup[item],epochs=args.epochs,n_splits=args.folds,seed=args.seed,device="cuda")
        results[item]=result; evaluations[item]=result.metrics
    ranking=rank_response_statistic_results(results) if args.phase=="screen" else _rank(ids,evaluations)
    report={
        "schema_version":"response_statistic_shared_router_v7_search","status":"exposed_development_candidate_search_not_clinically_validated","phase":args.phase,
        "model":{"name":"Response-Statistic Shared Router v7","dense_statistics":["median","q10","q90","range","max_real_time_velocity"],"view":"mean_absdiff","pca_fit":"outer_training_fold_label_free","shared_layers":["clinical_encoder","dense_pca_encoder","action_transformer","patient_projection","universal_head"],"endpoint_specific_layers":["script_query","binary_head"]},
        "candidate_registry":[x.__dict__ for x in registry],"candidate_ids":list(ids),"counts":counts,"evaluations":evaluations,
        "selection":{"primary_metric":"minimum_source_balanced_accuracy","secondary_metric":"minimum_source_specificity","tertiary_metric":"minimum_source_auroc","ranking":list(ranking)},
        "runtime":{"gpu":torch.cuda.get_device_name(0),"epochs":args.epochs,"seed":args.seed,"folds":args.folds,"elapsed_seconds":time.monotonic()-started,"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},
        "commitments":{**pcommit,"neuroface_collection_sha256":ncommit,"meei_collection_sha256":mcommit,"neuroface_manifest_sha256":args.neuroface_manifest_sha256,"meei_manifest_sha256":args.meei_manifest_sha256,"implementation_sha256":_implementation_sha256()},
        "audit":{"palsynet_protected_reads":0,"mayo_reads":0,"mayo_predictions":0},"decision":{"promotion_authorized":False,"clinical_claim_authorized":False}}
    v1_runner.write_report_no_overwrite(args.output,report); print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__=="__main__": main()
