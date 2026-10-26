#!/bin/bash
set -e

# ==============================================================================
# Publish a trained Kaldi/Vosk model bundle to the HuggingFace Hub
# ==============================================================================
# Uploads the bundled model .tar.gz (produced by the recipe's
# copy_final_result.sh) to a HuggingFace model repo, creating the repo if it
# does not exist. Uploading under the same filename replaces the file in place.
#
# Usage:
#   ./publish.sh <model-repo-id> [model.tar.gz]
#
#   <model-repo-id>  HF repo to create/update, e.g. techiaith/kaldi-cy
#   [model.tar.gz]   path to the bundle (default: models/model.tar.gz).
#
# HF_TOKEN is read from the environment or this repo's .env (needed to push).
# ==============================================================================

cd "$(dirname "$0")"

if [ -f .env ] && grep -q "^HF_TOKEN=" .env; then
    export $(grep "^HF_TOKEN=" .env | xargs)
fi

MODEL_REPO=$1
MODEL_TAR=${2:-models/model.tar.gz}

if [ -z "${MODEL_REPO}" ]; then
    echo "Usage: $0 <model-repo-id> [model.tar.gz]" >&2
    exit 1
fi

if [ ! -f "${MODEL_TAR}" ]; then
    echo "Error: model bundle not found at '${MODEL_TAR}'." >&2
    echo "Run the recipe's copy_final_result.sh first (it writes model.tar.gz)." >&2
    exit 1
fi

echo "Publishing ${MODEL_TAR} -> ${MODEL_REPO} (HF Hub) ..."
uv run huggingface-cli upload "${MODEL_REPO}" "${MODEL_TAR}" "$(basename "${MODEL_TAR}")" --repo-type model

echo
echo "Upload complete: https://huggingface.co/${MODEL_REPO}"
