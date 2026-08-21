ARG RUNTIME_IMAGE=ghcr.io/williamqiuzy/facial-attention-lab-shared-v8-runtime@sha256:59c25d4c8cdd56e7468a099b16772880a5a5bc7f9fb16c0cd315392154861c55
FROM ${RUNTIME_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/WilliamQiuzy/facial-attention-lab" \
      org.opencontainers.image.version="shared-v8-deployment-v1" \
      org.opencontainers.image.description="Private weight-bundled Shared V8 research deployment"

USER root
COPY --chown=1001:1001 manifest.json weights.npz /model/
RUN echo "72e40ea7b127b6768e931665df622550f06cc5a1bbad20070a42614c5b9901ab  /model/weights.npz" \
      | sha256sum --check --strict \
    && chmod 0555 /model \
    && chmod 0444 /model/manifest.json /model/weights.npz
USER 1001:1001
