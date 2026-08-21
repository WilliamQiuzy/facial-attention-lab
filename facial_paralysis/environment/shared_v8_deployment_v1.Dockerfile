FROM pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/facial_paralysis \
    SHARED_V8_RELEASE=/model \
    SHARED_V8_DEVICE=cuda \
    SHARED_V8_PORT=8080

COPY environment/shared_v8_runtime_v1.lock /tmp/shared_v8_runtime_v1.lock
RUN python -m pip install --no-cache-dir --no-deps \
      -r /tmp/shared_v8_runtime_v1.lock \
    && python -m pip check \
    && groupadd --gid 1001 app \
    && useradd --uid 1001 --gid 1001 --no-create-home \
      --shell /usr/sbin/nologin app \
    && rm -f /tmp/shared_v8_runtime_v1.lock

WORKDIR /app/facial_paralysis
COPY src/ /app/facial_paralysis/src/
COPY scripts/serve_shared_v8.py /app/facial_paralysis/scripts/serve_shared_v8.py
RUN find /app/facial_paralysis -type d -exec chmod 0555 {} + \
    && find /app/facial_paralysis -type f -exec chmod 0444 {} +

USER 1001:1001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).read()"]
ENTRYPOINT ["python", "scripts/serve_shared_v8.py"]
