#!/bin/bash
# Stage 2: mixed-domain GRPO (maths, multilingual maths, finance, logic, code) on top of the stage-1 model.
# The LR schedule spans 1200 steps; the released model is checkpoint-1000, merged -> runs/grpo-mix-merged.
# Resume after an interruption: run/05_grpo_mix.sh --resume runs/grpo-mix/checkpoint-N
set -eo pipefail; source "$(dirname "$0")/common.sh"
run/grpo.sh runs/grpo-math-merged data/rl_mix.jsonl runs/grpo-mix \
  --steps 1200 --lora --lr 1e-5 --max-prompt 1536 --max-completion 2048 --batch 4 --grad-accum 4 --save-steps 100 "$@"
run/merge_lora.sh runs/grpo-math-merged runs/grpo-mix/checkpoint-${CKPT:-1000} runs/grpo-mix-merged
