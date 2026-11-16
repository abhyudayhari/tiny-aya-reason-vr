#!/bin/bash
# Probe every mixed-domain source with the stage-1 model (k=8 samples, verified rewards), keep learnable prompts
# (0 < correct < 8) and write data/rl_mix.jsonl. Sources run in parallel, one GPU each.
set -eo pipefail; source "$(dirname "$0")/common.sh"
M=${1:-runs/grpo-math-merged}; N=${N_PROBE:-12000}
SRC=(deepscaler dapo17k gsm8k_hi_fr_de gsm8k_sw indicfin_pool finqa_train tatqa_arith kk_train fld2 mbpp_train taco kodcode)
mkdir -p results/probe; pids=()
for i in "${!SRC[@]}"; do
  g=$(( i % NG )); [ $i -ge $NG ] && wait ${pids[$((i - NG))]}
  CUDA_VISIBLE_DEVICES=$g "$EVAL/bin/python" scripts/probe_pool.py --model $M --sources ${SRC[$i]} --dump --n $N \
    --out results/probe/d_${SRC[$i]}.json > logs/probe-${SRC[$i]}.log 2>&1 &
  pids[$i]=$!
done
for p in "${pids[@]}"; do wait $p; done
"$EVAL/bin/python" scripts/make_mix.py
