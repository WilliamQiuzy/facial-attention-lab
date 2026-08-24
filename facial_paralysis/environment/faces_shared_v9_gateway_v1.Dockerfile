ARG PYTHON_BASE=python:3.11-slim-bookworm@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91
FROM ${PYTHON_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/facial_paralysis \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    FACE_LANDMARKER_MODEL=/models/face_landmarker.task \
    SHARED_V9_URL=http://shared-v9:8080 \
    FACES_GATEWAY_PORT=8081

RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY environment/faces_shared_v9_gateway_v1.lock /tmp/runtime.lock
RUN python -m pip install --no-cache-dir -r /tmp/runtime.lock \
    && python -m pip check \
    && groupadd --gid 1001 app \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /usr/sbin/nologin app \
    && rm -f /tmp/runtime.lock

RUN mkdir -p /models \
    && python -c "import hashlib, pathlib, urllib.request; p=pathlib.Path('/models/face_landmarker.task'); p.write_bytes(urllib.request.urlopen('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', timeout=60).read()); assert hashlib.sha256(p.read_bytes()).hexdigest() == '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'"
RUN python -c "from mediapipe.tasks import python as mpp; from mediapipe.tasks.python import vision as mpv; options=mpv.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path='/models/face_landmarker.task'), running_mode=mpv.RunningMode.IMAGE, num_faces=1); detector=mpv.FaceLandmarker.create_from_options(options); detector.close()"

WORKDIR /app/facial_paralysis
COPY src/ /app/facial_paralysis/src/
COPY scripts/serve_faces_shared_v9_gateway.py /app/facial_paralysis/scripts/serve_faces_shared_v9_gateway.py
RUN find /app/facial_paralysis /models -type d -exec chmod 0555 {} + \
    && find /app/facial_paralysis /models -type f -exec chmod 0444 {} +

USER 1001:1001
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/readyz', timeout=2).read()"]
ENTRYPOINT ["python", "scripts/serve_faces_shared_v9_gateway.py"]
