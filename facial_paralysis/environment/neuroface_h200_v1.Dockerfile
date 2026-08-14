FROM python:3.10.2-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      libegl1 libgl1 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/facial_paralysis
COPY environment/neuroface_h200_v1.requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r /tmp/requirements.txt

CMD ["python", "--version"]
