#!/bin/bash
# Optional SFT stage (not used by the released model). usage: run/sft.sh DATA(sft_messages.jsonl) OUT_DIR [train_sft.py args]
set -eo pipefail; source "$(dirname "$0")/common.sh"
DATA=$1; OUT=$2; shift 2
"$TRAIN/bin/accelerate" launch --multi_gpu --num_processes $NG --mixed_precision bf16 \
  scripts/train_sft.py --data "$DATA" --out "$OUT" "$@"
