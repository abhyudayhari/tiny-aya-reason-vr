#!/bin/bash
# One-time setup on a RunPod pod. Run from the repo root after cloning it onto the network volume:
#   cd /workspace && git clone <this repo> && cd tiny-aya-reason-vr && bash runpod/setup.sh
# Everything heavy (venvs, HF cache, uv cache) goes to /workspace, so a stopped/restarted pod keeps it.
set -eo pipefail
WS=${WORKSPACE:-/workspace}; VENVS=${VENVS:-$WS/venvs}
export UV_CACHE_DIR=$WS/.uv-cache UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 HF_HOME=${HF_HOME:-$WS/hf_cache}
[ -d "$WS" ] || { echo "$WS not found: attach a network volume (or set WORKSPACE)"; exit 1; }

# system tools (container disk; re-run this script after a pod is recreated)
if command -v apt-get >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux git curl build-essential > /dev/null
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH

# Python envs on the volume (skipped if already built)
if [ ! -x $VENVS/train/bin/python ]; then
  uv venv -q -p 3.12 $VENVS/train
  uv pip install -q -p $VENVS/train/bin/python -r requirements-train.txt
fi
if [ ! -x $VENVS/eval/bin/python ]; then
  uv venv -q -p 3.12 $VENVS/eval
  uv pip install -q -p $VENVS/eval/bin/python -r requirements-eval.txt
fi
$VENVS/train/bin/python -c "import torch,trl,vllm,aligntune,curatorkit,math_verify;print('TRAIN_OK torch',torch.__version__,'trl',trl.__version__,'vllm',vllm.__version__,'gpus',torch.cuda.device_count())"
$VENVS/eval/bin/python  -c "import torch,vllm,auditkit,math_verify;print('EVAL_OK torch',torch.__version__,'vllm',vllm.__version__)"

# Hugging Face login: set HF_TOKEN as a RunPod secret / pod env var (tiny-aya-global is gated)
if [ -n "$HF_TOKEN" ]; then
  $VENVS/train/bin/python -c "import os;from huggingface_hub import login;login(token=os.environ['HF_TOKEN'],add_to_git_credential=False)"
else
  echo "WARNING: HF_TOKEN not set; run/02_base.sh needs it to download CohereLabs/tiny-aya-global"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
df -h $WS | tail -1
echo "SETUP DONE"
