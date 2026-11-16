#!/bin/bash
# Everything end to end on one 8-GPU pod: data -> base -> stage 1 -> mix -> stage 2 -> evals of base and final model.
set -eo pipefail; D=$(dirname "$0")
$D/01_data.sh
$D/02_base.sh
$D/03_grpo_math.sh
$D/04_build_mix.sh
$D/05_grpo_mix.sh
$D/eval_all.sh models/tiny-aya-global base
$D/eval_all.sh runs/grpo-mix-merged reason-vr
