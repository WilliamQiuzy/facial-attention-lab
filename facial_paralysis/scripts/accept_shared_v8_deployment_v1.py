#!/usr/bin/env python3
"""Run synthetic, identifier-free HTTP acceptance against shared V8."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from src.deployment.shared_v8_release import DEPLOYMENT_MODEL_ID
from src.deployment.shared_v8_service import encode_request_npz


def _arrays(protocol: str, actions: int, codes: tuple[int, ...], dense: bool):
    generator = np.random.default_rng(20260821 + actions)
    available = np.full(actions, dense, dtype=bool)
    dense_shape = (actions, 32, 478, 3)
    original_dense = np.zeros(dense_shape, dtype=np.float32)
    mirrored_dense = np.zeros(dense_shape, dtype=np.float32)
    timestamps = np.zeros((actions, 32), dtype=np.float32)
    if dense:
        original_dense[:] = generator.normal(0.0, 0.01, dense_shape).astype(np.float32)
        mirrored_dense[:] = original_dense
        mirrored_dense[..., 0] *= -1.0
        timestamps[:] = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    return {
        "clinical_original": generator.normal(0.0, 0.2, (actions, 110)).astype(np.float32),
        "clinical_mirrored": generator.normal(0.0, 0.2, (actions, 110)).astype(np.float32),
        "dense_original": original_dense,
        "dense_mirrored": mirrored_dense,
        "dense_valid_mask": np.repeat(available[:, None], 32, axis=1),
        "dense_available": available,
        "dense_timestamps": timestamps,
        "action_mask": np.ones(actions, dtype=bool),
        "action_codes": np.asarray(codes, dtype=np.int64),
    }


def synthetic_requests() -> dict[str, bytes]:
    specifications = {
        "free_motion_four_window": (4, (0, 1, 2, 3), False),
        "scripted_three_action": (3, (4, 5, 6), True),
        "cue_aligned_action": (7, (7, 8, 9, 10, 11, 12, 5), True),
    }
    return {
        protocol: encode_request_npz(
            protocol, _arrays(protocol, actions, codes, dense)
        )
        for protocol, (actions, codes, dense) in specifications.items()
    }


def _request(url: str, *, payload: bytes | None = None, content_type: str | None = None):
    headers = {} if content_type is None else {"content-type": content_type}
    request = Request(url, data=payload, headers=headers, method="GET" if payload is None else "POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=30) as response:
            status = response.status
            body = response.read()
    except HTTPError as exc:
        status = exc.code
        body = exc.read()
    return status, body, (time.perf_counter() - started) * 1000.0


def _json(body: bytes) -> dict[str, object]:
    value = json.loads(body.decode("utf-8"))
    if type(value) is not dict:
        raise RuntimeError("service response is not a JSON object")
    return value


def run_acceptance(
    *, base_url: str, expected_weights_sha256: str,
    sequential_requests: int, concurrent_requests: int, workers: int,
) -> dict[str, object]:
    base = base_url.rstrip("/")
    status, body, _ = _request(base + "/healthz")
    if status != 200 or _json(body) != {"status": "ok"}:
        raise RuntimeError("health endpoint failed")
    status, body, _ = _request(base + "/readyz")
    ready = _json(body)
    if (
        status != 200
        or ready.get("status") != "ready"
        or ready.get("model_id") != DEPLOYMENT_MODEL_ID
        or ready.get("weights_sha256") != expected_weights_sha256
        or ready.get("device") != "cuda"
    ):
        raise RuntimeError("readiness identity differs from the locked release")
    payloads = synthetic_requests()
    baselines = {}
    for protocol, payload in payloads.items():
        for _ in range(5):
            status, body, _ = _request(
                f"{base}/v1/predict/{protocol}", payload=payload,
                content_type="application/octet-stream",
            )
            if status != 200:
                raise RuntimeError("warmup inference failed")
        baselines[protocol] = _json(body)

    latencies = []
    protocols = tuple(payloads)
    for index in range(sequential_requests):
        protocol = protocols[index % len(protocols)]
        status, body, elapsed = _request(
            f"{base}/v1/predict/{protocol}", payload=payloads[protocol],
            content_type="application/octet-stream",
        )
        if status != 200 or _json(body) != baselines[protocol]:
            raise RuntimeError("sequential inference changed or failed")
        latencies.append(elapsed)

    def one(index: int):
        protocol = protocols[index % len(protocols)]
        status, body, elapsed = _request(
            f"{base}/v1/predict/{protocol}", payload=payloads[protocol],
            content_type="application/octet-stream",
        )
        if status != 200 or _json(body) != baselines[protocol]:
            raise RuntimeError("concurrent inference changed or failed")
        return elapsed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        concurrent_latencies = list(executor.map(one, range(concurrent_requests)))

    negative = {}
    negative["malformed_npz"] = _request(
        base + "/v1/predict/free_motion_four_window",
        payload=b"PK\x03\x04", content_type="application/octet-stream",
    )[0]
    negative["wrong_content_type"] = _request(
        base + "/v1/predict/free_motion_four_window",
        payload=payloads["free_motion_four_window"], content_type="application/json",
    )[0]
    negative["unknown_protocol"] = _request(
        base + "/v1/predict/unknown",
        payload=payloads["free_motion_four_window"],
        content_type="application/octet-stream",
    )[0]
    if negative != {"malformed_npz": 400, "wrong_content_type": 415, "unknown_protocol": 404}:
        raise RuntimeError("negative API cases did not fail closed")
    ordered = sorted(latencies)
    concurrent_ordered = sorted(concurrent_latencies)
    percentile = lambda values, fraction: values[min(len(values) - 1, int(fraction * len(values)))]
    return {
        "schema_version": "shared_v8_deployment_acceptance_v1",
        "status": "pass",
        "model_id": DEPLOYMENT_MODEL_ID,
        "weights_sha256": expected_weights_sha256,
        "test_signal": "deterministic_synthetic_identifier_free",
        "clinical_validation": False,
        "protocols": list(protocols),
        "warmup_per_protocol": 5,
        "sequential_requests": sequential_requests,
        "concurrent_requests": concurrent_requests,
        "concurrency_workers": workers,
        "deterministic_responses": True,
        "latency_ms": {
            "sequential_mean": statistics.fmean(latencies),
            "sequential_p50": percentile(ordered, 0.50),
            "sequential_p95": percentile(ordered, 0.95),
            "sequential_p99": percentile(ordered, 0.99),
            "concurrent_p95": percentile(concurrent_ordered, 0.95),
        },
        "negative_cases": negative,
        "request_payload_sha256": {
            protocol: hashlib.sha256(payload).hexdigest()
            for protocol, payload in payloads.items()
        },
    }


def _write_no_overwrite(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--sequential-requests", type=int, default=1000)
    parser.add_argument("--concurrent-requests", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        len(args.expected_weights_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.expected_weights_sha256)
        or args.sequential_requests < 1000
        or args.concurrent_requests < 200
        or args.workers < 2
        or args.workers > 32
    ):
        raise ValueError("acceptance configuration is below the frozen minimum")
    report = run_acceptance(
        base_url=args.base_url,
        expected_weights_sha256=args.expected_weights_sha256,
        sequential_requests=args.sequential_requests,
        concurrent_requests=args.concurrent_requests,
        workers=args.workers,
    )
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    _write_no_overwrite(args.output, payload)
    print(payload.decode("utf-8").strip())


if __name__ == "__main__":
    main()
