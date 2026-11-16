#!/bin/bash
# Full eval suite (19 benchmarks) for one model. Jobs are spread over the pod's GPUs, one GPU each.
# usage: run/eval_all.sh MODEL TAG          results -> results/TAG*/, logs -> logs/eval-TAG-*.log
set -eo pipefail; source "$(dirname "$0")/common.sh"
M=$1; T=$2; [ "$NG" -ge 1 ] || { echo "no GPU found"; exit 1; }
JOBS=(
  "scripts/eval_math.py     --model $M --tag $T        --sets gsm8k,math500,mgsm"
  "scripts/eval_math.py     --model $M --tag $T-extra  --sets svamp,asdiv,gsm_hard,amc23,aime24,aime25"
  "scripts/eval_math.py     --model $M --tag $T-extra2 --sets gsm_plus,minerva,olympiad"
  "scripts/eval_math.py     --model $M --tag $T-indic  --sets gsm8k_indic"
  "scripts/eval_indicfin.py --model $M --tag $T"
  "scripts/eval_reason.py   --model $M --tag $T-reason --sets humaneval,mbpp,kk"
  "scripts/eval_reason.py   --model $M --tag $T-reason --sets bbh"
  "scripts/eval_reason.py   --model $M --tag $T-reason --sets finqa"
)
pids=(); fail=0
for i in "${!JOBS[@]}"; do
  g=$(( i % NG ))
  # with fewer GPUs than jobs, wait for the previous job on this GPU
  [ $i -ge $NG ] && { wait ${pids[$((i - NG))]} || fail=1; }
  CUDA_VISIBLE_DEVICES=$g "$EVAL/bin/python" ${JOBS[$i]} > logs/eval-$T-$i.log 2>&1 &
  pids[$i]=$!; echo "job $i on GPU $g: ${JOBS[$i]}"
done
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] && echo "EVALS DONE -> results/$T*" || { echo "some evals failed, see logs/eval-$T-*.log"; exit 1; }
