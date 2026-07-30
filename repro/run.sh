#!/usr/bin/env bash
set -euo pipefail

START_EPOCH=$(date +%s)
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="${PWD}/.cache/huggingface"
export ONLINE_CODEC_CACHE_DIR="${PWD}/.cache/codec"

echo "ORX_REPRO_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ORX_REPRO_COMMIT=$(git rev-parse HEAD)"
echo "ORX_REPRO_COMMAND=bash repro/run.sh"
echo "ORX_REPRO_BACKEND=kubernetes"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

apt-get update -qq
apt-get install -y -qq ffmpeg libarchive-tools curl > /tmp/apt-install.log
python -m pip install -q \
  "transformers>=5.7" "accelerate>=1.0" "huggingface_hub>=0.30" \
  "opencv-python-headless>=4.10" "pillow>=10" "decord>=0.6" \
  "safetensors>=0.4" "codec-video-prep>=0.2.5"

python repro/run_reproduction.py --config repro/config.json

END_EPOCH=$(date +%s)
echo "ORX_REPRO_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ORX_REPRO_WALL_SECONDS=$((END_EPOCH - START_EPOCH))"
