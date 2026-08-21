"""Validation tests for src/models/multitask.py.

Run with the project's torch env (no pytest dependency):
    KMP_DUPLICATE_LIB_OK=TRUE \
    /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python tests/test_multitask.py

Covers docs/model_design.md §9 for the multi-task model: shape contracts,
image-as-1-frame path, missing-action masking, coupled vs region severity,
multi-task routing (a task's loss touches only its own head + shared trunk),
gradient flow into the shared severity from every head, and a mixed-batch
planted-signal training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.multitask import (  # noqa: E402
    DEFAULT_TASKS,
    MultiTaskSeverityModel,
    SeverityTrunk,
    TaskSpec,
    TrunkConfig,
    multitask_loss,
)

from _testlib import Check  # noqa: E402

D, NA, H = 768, 5, 256


def _model(tasks=DEFAULT_TASKS, pool="mean"):
    torch.manual_seed(0)
    return MultiTaskSeverityModel(tasks=tasks, trunk_cfg=TrunkConfig(
        embed_dim=D, hidden_dim=H, n_actions=NA, action_pool=pool))


def test_forward_shapes(c: Check):
    m = _model()
    out = m(torch.randn(4, NA, D))
    for t in DEFAULT_TASKS:
        c.true(t.name in out, f"output has task {t.name}")
        c.eq(out[t.name].shape, (4, t.n_classes - 1), f"{t.name} cum_logits shape")


def test_frames_path_and_image_case(c: Check):
    """frame path works; an image (n_frames=1) is a valid degenerate input and
    gives the same result as feeding the single pooled action embedding."""
    m = _model().eval()
    frame_emb = torch.randn(3, NA, 1, D)            # 1 frame == image
    out_frames = m.forward_with_frames(frame_emb)
    # mean over a single real frame == that frame, so action_emb == frame_emb[:, :, 0]
    out_actions = m(frame_emb[:, :, 0])
    for t in DEFAULT_TASKS:
        c.true(bool(torch.allclose(out_frames[t.name], out_actions[t.name], atol=1e-6)),
               f"{t.name}: image path == pooled-action path")


def test_missing_action_masking(c: Check):
    """Masking out an action must change the representation but stay finite, and
    a present/absent flip must matter (mask is actually used)."""
    m = _model().eval()
    x = torch.randn(2, NA, D)
    full = torch.ones(2, NA, dtype=torch.bool)
    partial = full.clone()
    partial[0, 2] = False                            # patient 0 missing action 2
    o_full = m(x, full)["hb"]
    o_part = m(x, partial)["hb"]
    c.true(torch.isfinite(o_part).all(), "masked output finite")
    c.true(not torch.allclose(o_full[0], o_part[0]), "masking action changes that patient")
    c.true(torch.allclose(o_full[1], o_part[1]), "unmasked patient unchanged")


def test_no_present_action_rejected(c: Check):
    m = _model()
    x = torch.randn(2, NA, D)
    mask = torch.ones(2, NA, dtype=torch.bool)
    mask[1] = False                                  # patient 1 has nothing
    c.raises(lambda: m(x, mask), ValueError, "all-absent patient rejected")


def test_coupled_shares_severity(c: Check):
    """All coupled tasks read the same severity: their cum_logits differ only by
    thresholds, so (logit_taskA + thetaA_k) == severity == (logit_taskB + thetaB_j)
    column-broadcast. Concretely: per-sample severity recovered from any coupled
    head matches across heads."""
    m = _model().eval()
    x = torch.randn(5, NA, D)
    h, s = m.trunk.represent(x)
    out = m(x)
    for t in DEFAULT_TASKS:
        if not t.coupled:
            continue
        theta = m.thresholds[t.name].thresholds()    # (K-1,)
        recovered = out[t.name] + theta.unsqueeze(0)  # severity broadcast (B, K-1)
        # every column should equal the shared severity s
        c.true(bool(torch.allclose(recovered, s.unsqueeze(1).expand_as(recovered), atol=1e-5)),
               f"coupled task {t.name} reads shared severity")


def test_region_task_has_private_severity(c: Check):
    m = _model()
    coupled = [t.name for t in DEFAULT_TASKS if t.coupled]
    region = [t.name for t in DEFAULT_TASKS if not t.coupled]
    c.true(all(n not in m.region_proj for n in coupled), "coupled tasks have no private proj")
    c.true(all(n in m.region_proj for n in region), "region tasks have private proj")
    c.true(len(region) >= 1, "there is at least one region task to exercise")


def test_routing_only_supervises_matching_head(c: Check):
    """A batch of only 'binary' samples must produce gradients on the binary
    thresholds and the shared severity, but NOT on the HB-specific thresholds."""
    m = _model()
    x = torch.randn(6, NA, D)
    out = m(x)
    y = torch.randint(0, 2, (6,))
    task_ids = ["binary"] * 6
    loss, parts = multitask_loss(out, y, task_ids, DEFAULT_TASKS)
    loss.backward()
    c.true(set(parts.keys()) == {"binary"}, f"only binary in loss parts, got {set(parts)}")
    c.true(m.thresholds["binary"].first.grad is not None
           and m.thresholds["binary"].first.grad.abs() > 0, "binary thresholds get grad")
    c.true(m.trunk.severity_proj.weight.grad is not None
           and m.trunk.severity_proj.weight.grad.abs().sum() > 0,
           "shared severity gets grad from binary samples")
    hb_first = m.thresholds["hb"].first.grad
    c.true(hb_first is None or float(hb_first.abs()) == 0.0,
           "HB-only thresholds get NO grad from binary-only batch")


def test_every_coupled_head_feeds_severity(c: Check):
    """Each coupled task alone must produce a gradient on the shared severity."""
    for t in DEFAULT_TASKS:
        if not t.coupled:
            continue
        m = _model()
        x = torch.randn(4, NA, D)
        out = m(x)
        y = torch.randint(0, t.n_classes, (4,))
        loss, _ = multitask_loss(out, y, [t.name] * 4, DEFAULT_TASKS)
        loss.backward()
        g = m.trunk.severity_proj.weight.grad
        c.true(g is not None and g.abs().sum() > 0, f"task {t.name} feeds shared severity")


def test_mixed_batch_loss(c: Check):
    m = _model()
    x = torch.randn(8, NA, D)
    out = m(x)
    task_ids = ["hb", "binary", "coarse3", "eyes", "mouth", "hb", "binary", "coarse3"]
    # label valid for each sample's task
    spec = {t.name: t for t in DEFAULT_TASKS}
    y = torch.tensor([spec[t].n_classes - 1 for t in task_ids])  # max valid label each
    loss, parts = multitask_loss(out, y, task_ids, DEFAULT_TASKS)
    c.true(torch.isfinite(loss), "mixed-batch loss finite")
    c.true(set(parts) == {"hb", "binary", "coarse3", "eyes", "mouth"}, "all present tasks scored")


def test_loss_validation(c: Check):
    m = _model()
    out = m(torch.randn(3, NA, D))
    c.raises(lambda: multitask_loss(out, torch.zeros(3, dtype=torch.long), ["nope"] * 3, DEFAULT_TASKS),
             KeyError, "unknown task name rejected")
    c.raises(lambda: multitask_loss(out, torch.zeros(2, dtype=torch.long), ["hb"] * 3, DEFAULT_TASKS),
             ValueError, "task_ids/label length mismatch rejected")


def test_duplicate_task_names_rejected(c: Check):
    c.raises(lambda: MultiTaskSeverityModel(tasks=[TaskSpec("a", 6), TaskSpec("a", 2)]),
             ValueError, "duplicate task names rejected")


def test_predict_hb(c: Check):
    m = _model().eval()
    pred = m.predict_hb(torch.randn(10, NA, D))
    c.eq(pred.shape, (10,), "predict_hb shape")
    c.true(bool(((pred >= 0) & (pred <= 5)).all()), "HB predictions in [0,5]")
    c.raises(lambda: m.predict_hb(torch.randn(2, NA, D), hb_task="missing"),
             KeyError, "unknown hb_task rejected")


def test_pooling_modes(c: Check):
    for pool in ("mean", "max", "attention"):
        m = _model(pool=pool)
        out = m(torch.randn(3, NA, D))
        c.true(torch.isfinite(out["hb"]).all(), f"pool={pool} produces finite output")


def test_mixed_training_recovers_signal(c: Check):
    """End-to-end: plant a severity signal, generate HB + binary labels from it,
    train on a MIXED stream, and verify HB accuracy rises and the learned global
    severity correlates with the true latent severity."""
    torch.manual_seed(11)
    N = 200
    # latent severity per patient
    true_w = torch.randn(D)
    base = torch.randn(N, NA, D)
    # inject the severity direction into every action embedding
    sev = torch.randn(N)
    X = base + sev.view(N, 1, 1) * true_w.view(1, 1, D) * 0.5
    # HB grade from severity quantiles (6 classes)
    qs = torch.quantile(sev, torch.linspace(0, 1, 7)[1:-1])
    hb = torch.bucketize(sev, qs)
    biny = (sev > sev.median()).long()

    m = _model()
    opt = torch.optim.Adam(m.parameters(), lr=0.02)

    def eval_acc():
        m.eval()
        with torch.no_grad():
            pred = m.predict_hb(X)
        m.train()
        return float((pred == hb).float().mean())

    acc0 = eval_acc()
    for step in range(150):
        opt.zero_grad()
        # alternate HB-labeled and binary-labeled mini-batches (heterogeneous!)
        idx = torch.randint(0, N, (32,))
        out = m(X[idx])
        if step % 2 == 0:
            loss, _ = multitask_loss(out, hb[idx], ["hb"] * 32, DEFAULT_TASKS)
        else:
            loss, _ = multitask_loss(out, biny[idx], ["binary"] * 32, DEFAULT_TASKS)
        loss.backward()
        opt.step()
    acc1 = eval_acc()

    # correlation of learned severity with true latent severity
    m.eval()
    with torch.no_grad():
        _, s_learned = m.trunk.represent(X)
    corr = float(torch.corrcoef(torch.stack([s_learned, sev]))[0, 1].abs())

    c.true(acc1 > acc0, f"HB accuracy improved with mixed training ({acc0:.3f} -> {acc1:.3f})")
    c.true(acc1 > 0.45, f"HB accuracy non-trivial ({acc1:.3f}) [6-class, chance~0.17]")
    c.true(corr > 0.8, f"learned severity correlates with truth (|r|={corr:.3f})")


if __name__ == "__main__":
    from _testlib import run_all
    run_all(__name__, globals())
