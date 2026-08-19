#!/usr/bin/env bash
# Download the public, no-friction datasets / pretrained models referenced in
# Technical_Report.docx §1.5 and the baseline shortlist.
#
# This script ONLY downloads sources that are publicly accessible WITHOUT
# registration, EULA acceptance, or institutional application. Restricted
# datasets (MEEI, CK+, YFP, Toronto NeuroFace) need a human in the loop —
# see `data/external/README_RESTRICTED.md` (created by this script if missing).
#
# Usage:
#   bash scripts/download_public_datasets.sh           # all three
#   bash scripts/download_public_datasets.sh palsynet  # one
#   bash scripts/download_public_datasets.sh marlin farl
#
# Cost: ~3-4 GB local disk. Re-running is idempotent (skips existing files).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="${ROOT}/data/external"
mkdir -p "${EXTERNAL}"

WHICH="${*:-palsynet marlin farl}"

hdr() { printf '\n========== %s ==========\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Pick a python that has huggingface_hub. We use the project's `dev` conda env
# because system python on macOS is often PEP 668-managed and rejects pip installs.
HF_PY="${HF_PY:-conda run -n dev python}"

download_palsynet() {
    hdr "PalsyNet (HuggingFace, CC-BY-4.0)"
    local dst="${EXTERNAL}/palsynet"
    mkdir -p "${dst}"
    if [ -n "$(ls -A "${dst}" 2>/dev/null)" ]; then
        echo "  ${dst} already populated — skip. To force re-download: rm -rf ${dst}"
        return 0
    fi
    KMP_DUPLICATE_LIB_OK=TRUE $HF_PY -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='jasir/palsynet-data', repo_type='dataset', local_dir='${dst}')
"
    echo "  -> ${dst}/"
    du -sh "${dst}/" 2>/dev/null
}

download_marlin() {
    hdr "MARLIN ViT-Base (HuggingFace, MIT)"
    local dst="${EXTERNAL}/marlin_vit_base_ytf"
    mkdir -p "${dst}"
    if [ -n "$(ls -A "${dst}" 2>/dev/null)" ]; then
        echo "  ${dst} already populated — skip."
        return 0
    fi
    KMP_DUPLICATE_LIB_OK=TRUE $HF_PY -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='ControlNet/marlin_vit_base_ytf', repo_type='model', local_dir='${dst}')
"
    echo "  -> ${dst}/"
    du -sh "${dst}/" 2>/dev/null
}

download_farl() {
    hdr "FaRL ViT-B/16 (GitHub Releases, MIT)"
    local dst="${EXTERNAL}/farl"
    mkdir -p "${dst}"
    local url="https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
    local out="${dst}/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
    if [ -f "${out}" ]; then
        echo "  ${out} already present — skip."
        return 0
    fi
    curl -fL -o "${out}" "${url}"
    echo "  -> ${out}"
    du -sh "${out}" 2>/dev/null
}

write_restricted_note() {
    local f="${EXTERNAL}/README_RESTRICTED.md"
    if [ -f "${f}" ]; then return 0; fi
    cat > "${f}" <<'EOF'
# Datasets requiring human action

These are NOT downloaded by `scripts/download_public_datasets.sh`. How to get each:

## 1. MEEI Facial Palsy Standard Set (Greene 2020, Laryngoscope)
**Why we want it:** the ONLY public dataset with HB grades on real palsy patients (60 subjects, also has SFGS + eFACE). Field-standard HB benchmark and our planned supervised warm-start before Mayo fine-tuning (Tech Report §1.5).
- **Access:** http://www.sircharlesbell.com/ — register a free account on the Sir Charles Bell Society site. Look for "Standard Datasets" / "MEEI Standard Set" after login.
- **Wait time:** immediate to a few days
- **License:** research/education use; no commercial
- **Format:** photos + videos with per-subject HB / SFGS / eFACE scores

## 2. YouTube Facial Palsy (YFP) — Hsu et al. 2018
**Why we want it:** 32 videos, 21 patients, ~2,246 labeled frames with 4-class HB-derived severity. External validation per Tech Report §1.5.
- **Access:** email request via https://github.com/AvLab-CV/YouTube-Facial-Palsy-Database
  - Contact: Prof. Gee-Sern "Jison" Hsu, NTUST. Email is on the repo README.
  - Mention Mayo Clinic, research purpose (HB grading evaluation), research-only use.
- **Wait time:** days to ~2 weeks
- **License:** research use, by request

## 3. CK+ (Extended Cohn-Kanade) — Lucey 2010
**Why we want it:** 593 sequences, 123 healthy subjects with 7 emotion labels. Used by Oo et al. as the "healthy" class for binary FP. Not needed if our Mayo cohort includes healthy controls.
- **Access:**
  - Original: email Prof. Jeffrey Cohn (jeffcohn@pitt.edu) with EULA acknowledgment.
  - Zenodo mirror: https://zenodo.org/records/11221351 — register a free Zenodo account, accept the academic EULA, then direct download.
- **Wait time:** immediate (Zenodo) or days (Pitt email)
- **License:** academic-only, non-commercial

## 4. Toronto NeuroFace (Bandini 2021)
**Why we want it:** 261 videos, 36 subjects (ALS + post-stroke + controls), 68 landmarks + perceptual scores. Trained Shagdar 2024 GNN. Not directly HB-relevant.
- **Access:** apply via https://slp.utoronto.ca/faculty/yana-yunusova/speech-production-lab/datasets/
  - Requires REB approval and signed Data Use Agreement.
- **Wait time:** weeks
- **License:** research-only DUA

## Priority for our HB project

1. **MEEI** — highest priority. Register at sircharlesbell.com today.
2. **YFP** — secondary, email Prof. Hsu today.
3. **CK+** — only if we need a healthy cohort (likely not).
4. **Toronto NeuroFace** — low priority. Not needed for HB.
EOF
    echo "  wrote ${f}"
}

write_restricted_note

for x in $WHICH; do
    case "$x" in
        palsynet) download_palsynet ;;
        marlin)   download_marlin ;;
        farl)     download_farl ;;
        *) echo "unknown target: $x  (valid: palsynet marlin farl)" >&2; exit 1 ;;
    esac
done

hdr "Summary"
echo "data/external/ contents:"
ls -la "${EXTERNAL}/" 2>/dev/null
du -sh "${EXTERNAL}"/* 2>/dev/null
echo
echo "See ${EXTERNAL}/README_RESTRICTED.md for datasets that need application/registration."
