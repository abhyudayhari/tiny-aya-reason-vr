# Sourced by every run/*.sh script. Defaults suit a RunPod pod with the repo cloned onto the network volume (/workspace).
P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$P"
WS=${WORKSPACE:-/workspace}                       # RunPod network volume: survives pod stop/restart
VENVS=${VENVS:-$WS/venvs}
TRAIN=$VENVS/train                                # AlignTune + CuratorKIT
EVAL=$VENVS/eval                                  # vLLM + AuditKIT
NG=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
export HF_HOME=${HF_HOME:-$WS/hf_cache} TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p logs results runs data models
BASE=models/tiny-aya-global-eos3
