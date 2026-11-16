#!/bin/bash
# Download every eval set and RL source, build eval files and the decontaminated RL pool (CPU + network, ~1-2 h).
set -eo pipefail; source "$(dirname "$0")/common.sh"
PY=$EVAL/bin/python
mkdir -p data/cands data/raw_extra data/raw_reason
$PY scripts/data/fetch_data.py eval rl              # GSM8K, MATH-500, MGSM; GSM8K/MATH train for stage 1
$PY scripts/data/fetch_extra_evals.py               # SVAMP, ASDiv, GSM-Hard, Minerva, OlympiadBench, AMC23, AIME, GSM-Plus, IndicFin
$PY scripts/data/build_extra.py
$PY scripts/data/fetch_reason.py                    # HumanEval, MBPP, BBH, Knights & Knaves, FinQA eval sets
$PY scripts/data/fetch_cands.py math fin logic code # RL candidate sources
$PY scripts/data/fetch_ml.py                        # multilingual GSM8K (incl. GSM8K-Indic eval)
$PY scripts/data/fetch_lla.py
$PY scripts/data/build_indic_eval.py
$PY scripts/data/build_pool.py                      # data/pool/<source>.jsonl, 13-gram decontaminated against all evals
$PY scripts/data/make_rl_math.py
echo "DATA READY"
