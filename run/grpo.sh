#!/bin/bash
# GRPO (DAPO loss) with AlignTune on all GPUs of the pod.
# usage: run/grpo.sh MODEL DATA OUT_DIR [train_grpo.py args]      add --resume OUT_DIR/checkpoint-N to resume
set -eo pipefail; source "$(dirname "$0")/common.sh"
MODEL=$1; DATA=$2; OUT=$3; shift 3
export ALIGNTUNE_ENABLE_VLLM=1
"$TRAIN/bin/accelerate" launch --multi_gpu --num_processes $NG --mixed_precision bf16 \
  scripts/train_grpo.py --model "$MODEL" --data "$DATA" --out "$OUT" "$@"
