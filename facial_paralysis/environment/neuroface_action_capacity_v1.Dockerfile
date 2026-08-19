FROM facial-paralysis-neuroface:v1.3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/facial_paralysis

COPY src /workspace/facial_paralysis/src
COPY scripts/run_neuroface_action_capacity_v1.py /workspace/facial_paralysis/scripts/run_neuroface_action_capacity_v1.py
COPY scripts/run_mirror_invariant_110d.py /workspace/facial_paralysis/scripts/run_mirror_invariant_110d.py
COPY scripts/launch_neuroface_action_capacity_v1.py /workspace/facial_paralysis/scripts/launch_neuroface_action_capacity_v1.py
COPY environment/neuroface_h200_v1.lock /workspace/facial_paralysis/environment/neuroface_h200_v1.lock
COPY environment/neuroface_action_capacity_host_audit_ed25519_public.pem /workspace/facial_paralysis/environment/neuroface_action_capacity_host_audit_ed25519_public.pem

RUN chmod -R a-w /workspace/facial_paralysis \
    && find /workspace/facial_paralysis -type d -exec chmod 0555 {} + \
    && find /workspace/facial_paralysis -type f -exec chmod 0444 {} +

USER 1001:1001

CMD ["python", "scripts/run_neuroface_action_capacity_v1.py", "--help"]
