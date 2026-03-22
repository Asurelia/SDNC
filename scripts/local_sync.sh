#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SDNC Local Sync — synchronise le dernier checkpoint depuis GCS
#
# Usage : bash scripts/local_sync.sh
# Prérequis : git, gsutil (Google Cloud SDK), python3
# ─────────────────────────────────────────────────────────────
set -e

REPO_URL="${SDNC_GITHUB_REPO:-}"
LOCAL_DIR="${SDNC_LOCAL_DIR:-$(pwd)}"
BUCKET="${SDNC_GCS_BUCKET:-sdnc-models}"

echo "=== SDNC Local Sync ==="
echo "  Repo   : ${REPO_URL:-'(pas configuré)'}"
echo "  Local  : ${LOCAL_DIR}"
echo "  Bucket : ${BUCKET}"
echo ""

# 1. Git pull
if [ -d "${LOCAL_DIR}/.git" ]; then
    echo "Git pull..."
    cd "${LOCAL_DIR}"
    git pull origin main
else
    if [ -n "${REPO_URL}" ]; then
        echo "Git clone..."
        git clone "${REPO_URL}" "${LOCAL_DIR}"
        cd "${LOCAL_DIR}"
    else
        echo "Pas de repo configuré — skip git."
        cd "${LOCAL_DIR}"
    fi
fi

# 2. Lire releases/latest.json
LATEST_JSON="releases/latest.json"
if [ ! -f "${LATEST_JSON}" ]; then
    echo "Pas de ${LATEST_JSON} — rien à synchroniser."
    exit 0
fi

echo ""
echo "Lecture ${LATEST_JSON}..."
GCS_URI=$(python3 -c "import sys,json; print(json.load(sys.stdin).get('gcs_uri',''))" < "${LATEST_JSON}")
STEP=$(python3 -c "import sys,json; print(json.load(sys.stdin).get('step','?'))" < "${LATEST_JSON}")

echo "  Step    : ${STEP}"
echo "  GCS URI : ${GCS_URI}"

if [ -z "${GCS_URI}" ]; then
    echo "Pas de gcs_uri — rien à télécharger."
    exit 0
fi

# 3. Télécharger si nécessaire
FILENAME=$(basename "${GCS_URI}")
mkdir -p checkpoints

if [ -f "checkpoints/${FILENAME}" ]; then
    echo ""
    echo "Checkpoint déjà présent : checkpoints/${FILENAME}"
    echo "Synchronisation terminée."
else
    echo ""
    echo "Téléchargement nouveau checkpoint depuis GCS..."
    if command -v gsutil &> /dev/null; then
        gsutil cp "${GCS_URI}" "checkpoints/${FILENAME}"
        echo "Checkpoint téléchargé : checkpoints/${FILENAME}"
    else
        echo "gsutil non disponible — téléchargement impossible."
        echo "Installez Google Cloud SDK : https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
fi

# 4. Lancer le runner interactif
echo ""
echo "=== Lancement du runner local ==="
python3 -c "
from pipeline.local_runner import LocalRunner
runner = LocalRunner()
model = runner.load_student('checkpoints/${FILENAME}')
runner.interactive_chat(model)
"
