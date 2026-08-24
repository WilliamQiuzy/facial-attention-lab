"""Static release gates for the deployable Facial Process Web stack."""
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_gateway_image_pins_runtime_model_and_nonroot_boundary(c: Check):
    dockerfile = _text("environment/faces_shared_v9_gateway_v1.Dockerfile")
    lock = _text("environment/faces_shared_v9_gateway_v1.lock")
    c.true(bool(re.search(r"^ARG PYTHON_BASE=.+@sha256:[0-9a-f]{64}$", dockerfile, re.MULTILINE)))
    c.true("FROM ${PYTHON_BASE}" in dockerfile)
    c.true("mediapipe==0.10.35" in lock)
    c.true("opencv-contrib-python==4.11.0.86" in lock)
    c.true("python-multipart==0.0.20" in lock)
    c.true("libgles2" in dockerfile)
    c.true("libegl1" in dockerfile)
    c.true("64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff" in dockerfile)
    c.true("FaceLandmarker.create_from_options" in dockerfile)
    c.true("USER 1001:1001" in dockerfile)
    c.true("serve_faces_shared_v9_gateway.py" in dockerfile)


def test_gateway_runtime_lock_is_a_closed_exact_environment(c: Check):
    lock_lines = {
        line.strip()
        for line in _text("environment/faces_shared_v9_gateway_v1.lock").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    expected = {
        "absl-py==2.5.0",
        "annotated-types==0.7.0",
        "anyio==4.9.0",
        "certifi==2026.7.22",
        "cffi==2.1.1",
        "click==8.1.8",
        "contourpy==1.3.3",
        "cycler==0.12.1",
        "fastapi==0.116.1",
        "flatbuffers==25.12.19",
        "fonttools==4.63.0",
        "h11==0.14.0",
        "idna==3.7",
        "joblib==1.5.1",
        "kiwisolver==1.5.0",
        "matplotlib==3.11.1",
        "mediapipe==0.10.35",
        "numpy==1.26.4",
        "opencv-contrib-python==4.11.0.86",
        "packaging==26.3",
        "pillow==12.3.0",
        "pycparser==3.0",
        "pydantic==2.11.7",
        "pydantic-core==2.33.2",
        "pyparsing==3.3.2",
        "python-dateutil==2.9.0.post0",
        "python-multipart==0.0.20",
        "scikit-learn==1.6.1",
        "scipy==1.13.1",
        "six==1.17.0",
        "sniffio==1.2.0",
        "sounddevice==0.5.6",
        "starlette==0.47.2",
        "threadpoolctl==3.6.0",
        "typing-extensions==4.14.1",
        "typing-inspection==0.4.1",
        "uvicorn==0.35.0",
    }
    c.eq(lock_lines, expected)


def test_dataset_package_keeps_torch_out_of_gateway_import_path(c: Check):
    package_init = _text("src/datasets/__init__.py")
    dynamic_landmark = _text("src/datasets/dynamic_landmark.py")
    c.true("from .patient_videos import" not in package_init)
    c.true("def __getattr__" in package_init)
    c.true("patient_multistream" not in dynamic_landmark)
    c.true("from .feature_schema import MP_FEATURE_NAMES_BY_SCHEMA" in dynamic_landmark)


def test_web_image_is_static_same_origin_proxy(c: Check):
    dockerfile = _text("facial_paralysis_web/Dockerfile")
    nginx = _text("facial_paralysis_web/nginx.conf")
    ignored = _text(".dockerignore")
    c.true(len(re.findall(r"^ARG (?:NODE|NGINX)_BASE=.+@sha256:[0-9a-f]{64}$", dockerfile, re.MULTILINE)) == 2)
    c.true("FROM ${NODE_BASE} AS build" in dockerfile)
    c.true("FROM ${NGINX_BASE}" in dockerfile)
    c.true("pnpm install --frozen-lockfile" in dockerfile)
    c.true("pnpm-workspace.yaml" in dockerfile)
    c.true("pnpm build" in dockerfile)
    c.true("location = /api/v1/facial-paralysis/infer" in nginx)
    c.true("proxy_pass http://gateway:8081" in nginx)
    c.true("client_max_body_size 512m" in nginx)
    c.true("Content-Security-Policy" in nginx)
    c.true("frame-ancestors 'none'" in nginx)
    c.true("Permissions-Policy" in nginx)
    c.true("camera=(self)" in nginx)
    c.true("facial_paralysis_web/node_modules/" in ignored)
    c.true("facial_paralysis_web/dist/" in ignored)


def test_compose_exposes_only_web_and_pins_shared_v9(c: Check):
    compose = _text("deploy/facial-process-shared-v9/compose.yaml")
    c.true("image: facial-attention-lab-shared-v9:script-flex-v1" in compose)
    c.true("dockerfile: environment/shared_v9_public_v1.Dockerfile" in compose)
    c.true(
        "81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64"
        in _text("environment/shared_v9_public_v1.Dockerfile")
    )
    c.true(all(name in compose for name in ("shared-v9:", "gateway:", "web:")))
    c.true(compose.count("ports:") == 1)
    c.true(
        '"127.0.0.1:8080:8080"' in compose,
        "the local research UI must not bind to every host interface",
    )
    c.true(compose.count("read_only: true") == 3)
    c.true(compose.count("no-new-privileges:true") == 3)
    c.true(compose.count("platform: linux/amd64") == 2)
    c.true(
        "/tmp:rw,noexec,nosuid,nodev,size=1258291200,mode=1777" in compose,
        "the gateway must hold the spooled upload and its exact decode copy",
    )
    c.true("volumes:" not in compose, "raw video must not have persistent storage")
    c.eq(compose.count('driver: "json-file"'), 3)
    c.eq(compose.count('max-size: "10m"'), 3)
    c.eq(compose.count('max-file: "3"'), 3)


if __name__ == "__main__":
    run_all("test_facial_process_shared_v9_deployment", dict(globals()))
