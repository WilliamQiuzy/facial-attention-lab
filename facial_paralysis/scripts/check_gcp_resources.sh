#!/usr/bin/env bash
# Read-only discovery of GCP resources in the active project.
# Does NOT create / start / modify anything — all `list` and `describe` calls.
#
# Prerequisites:
#   gcloud auth login                          # browser OAuth, refreshes the CLI token
#   gcloud auth application-default login      # browser OAuth, refreshes ADC for SDKs
#
# Usage:
#   bash scripts/check_gcp_resources.sh
#   bash scripts/check_gcp_resources.sh > outputs/gcp_inventory.txt 2>&1   # save it

set -u   # surface unset vars, but keep going on individual API errors

PROJECT="$(gcloud config get-value project 2>/dev/null)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null)"

hdr() { printf '\n========== %s ==========\n' "$*"; }
sub() { printf '\n--- %s ---\n' "$*"; }

hdr "Identity & project"
echo "account: $ACCOUNT"
echo "project: $PROJECT"
gcloud projects describe "$PROJECT" --format='value(name, projectNumber, lifecycleState)' 2>&1 \
    | sed 's/^/  /'

hdr "All projects this account can see"
gcloud projects list --format='table(projectId, name, projectNumber)' 2>&1 | head -50

hdr "Enabled APIs (compute-relevant subset)"
gcloud services list --enabled --format='value(config.name)' 2>&1 \
    | grep -E '(compute|aiplatform|notebooks|bigquery|storage|container|run|batch|dataproc|tpu|deeplearning)' \
    | sed 's/^/  /'

hdr "IAM permissions on this project (what you can do)"
gcloud projects get-iam-policy "$PROJECT" \
    --flatten='bindings[].members' \
    --filter="bindings.members:$ACCOUNT" \
    --format='value(bindings.role)' 2>&1 \
    | sort -u | sed 's/^/  /'

# ----- Vertex AI Workbench (the most common GPU notebook surface) -----
hdr "Vertex AI Workbench INSTANCES (new generation, user-managed VMs)"
gcloud workbench instances list --format='table(name, state, machineType.basename(), location, gceSetup.acceleratorConfigs[0].type, gceSetup.acceleratorConfigs[0].coreCount)' 2>&1

hdr "AI Platform Notebooks (legacy)"
gcloud notebooks instances list --format='table(name, state, machineType.basename(), location, acceleratorConfig.type, acceleratorConfig.coreCount)' 2>&1 | head -30

# ----- Vertex AI training resources -----
hdr "Vertex AI custom-training jobs (recent)"
gcloud ai custom-jobs list --region=us-central1 --format='table(displayName, state, createTime.date())' --limit=10 2>&1 | head -20
sub "(switch --region= if your project is in a different one; check 'Quotas' section below for hints)"

# ----- Compute Engine -----
hdr "Compute Engine VMs (any zone)"
gcloud compute instances list --format='table(name, zone.basename(), machineType.basename(), status, accelerators[0].type.basename(), accelerators[0].count)' 2>&1 | head -50

hdr "Compute Engine GPU quotas (per region) — non-zero rows"
gcloud compute regions list --format='value(name)' 2>&1 | while read region; do
    [ -z "$region" ] && continue
    gcloud compute regions describe "$region" --format='value(quotas[].metric,quotas[].limit,quotas[].usage)' 2>/dev/null \
        | tr ';' '\n' \
        | grep -iE 'GPU|TPU|NVIDIA' \
        | awk -v r="$region" -F: '{print r, $0}'
done | awk 'NF && $0 !~ /[[:space:]]0$/' | head -40

# ----- Storage -----
hdr "GCS buckets in this project"
gsutil ls -p "$PROJECT" 2>&1 | head -30

hdr "BigQuery datasets in this project"
bq ls --project_id="$PROJECT" --format=prettyjson 2>&1 | head -40

# ----- Cloud Run / Batch / Dataproc (other compute) -----
hdr "Cloud Run services"
gcloud run services list --format='table(metadata.name, status.url, status.conditions[0].type, status.conditions[0].status)' 2>&1 | head -20

hdr "Batch jobs"
gcloud batch jobs list --format='table(name, status.state)' 2>&1 | head -20

hdr "Dataproc clusters"
gcloud dataproc clusters list --region=us-central1 --format='table(clusterName, status.state, config.workerConfig.numInstances)' 2>&1 | head -20

# ----- Kubernetes -----
hdr "GKE clusters"
gcloud container clusters list --format='table(name, location, currentNodeCount, nodeConfig.machineType, status)' 2>&1 | head -20

echo
echo "===== done ====="
