#!/bin/bash
# Stage 1: math-only GRPO from the base model.
#   1) keep GSM8K/MATH train prompts the base solves 1-7 times out of 8  -> data/rl_train_base.jsonl
#   2) 250 GRPO steps with LoRA, then merge                              -> runs/grpo-math-merged
set -eo pipefail; source "$(dirname "$0")/common.sh"
[ -s data/rl_train_base.jsonl ] || CUDA_VISIBLE_DEVICES=0 "$EVAL/bin/python" scripts/filter_rl.py --model $BASE --out data/rl_train_base.jsonl --k 8
run/grpo.sh $BASE data/rl_train_base.jsonl runs/grpo-math \
  --steps 250 --batch 4 --gens 8 --grad-accum 8 --max-completion 1536 --lora --lr 2e-5 "$@"
run/merge_lora.sh $BASE runs/grpo-math runs/grpo-math-merged
