FROM pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/WilliamQiuzy/facial-attention-lab" \
      org.opencontainers.image.version="shared-v9-research-v1" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.description="Public Shared V9 research inference service"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/facial_paralysis \
    SHARED_V9_RELEASE=/model \
    SHARED_V9_DEVICE=cpu \
    SHARED_V9_PORT=8080

COPY environment/shared_v8_runtime_v1.lock /tmp/shared_v9_runtime_v1.lock
RUN python -m pip install --no-cache-dir --no-deps \
      -r /tmp/shared_v9_runtime_v1.lock \
    && python -m pip check \
    && groupadd --gid 1001 app \
    && useradd --uid 1001 --gid 1001 --no-create-home \
      --shell /usr/sbin/nologin app \
    && rm -f /tmp/shared_v9_runtime_v1.lock

WORKDIR /app/facial_paralysis
COPY src/ /app/facial_paralysis/src/
COPY scripts/serve_shared_v9.py /app/facial_paralysis/scripts/serve_shared_v9.py
COPY --chown=1001:1001 \
  releases/shared-v9-research-v1/manifest.json \
  releases/shared-v9-research-v1/weights-seed0.npz \
  releases/shared-v9-research-v1/weights-seed1.npz \
  releases/shared-v9-research-v1/weights-seed2.npz \
  /model/
RUN echo "c4fdaf054f3076a2e31b0e1ae93d1e91a45212817eb39d1c4a53620a4007b18f  /model/manifest.json" \
      | sha256sum --check --strict \
    && echo "7befb2853b89a11ebf904483b027098d042e36d12891d65d93ffc4766ad3fc96  /model/weights-seed0.npz" \
      | sha256sum --check --strict \
    && echo "9f27e5d3535a472c09cb9cfd94cadd432d8373242b73a67e6d533e287d76760f  /model/weights-seed1.npz" \
      | sha256sum --check --strict \
    && echo "b97ef723ee8a2fdb6c90d04c1f5c1adb0b090292a76313feddc46f3b9a68fdf7  /model/weights-seed2.npz" \
      | sha256sum --check --strict \
    && find /app/facial_paralysis -type d -exec chmod 0555 {} + \
    && find /app/facial_paralysis -type f -exec chmod 0444 {} + \
    && chmod 0555 /model \
    && chmod 0444 /model/manifest.json /model/weights-seed*.npz

USER 1001:1001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).read()"]
ENTRYPOINT ["python", "scripts/serve_shared_v9.py"]
