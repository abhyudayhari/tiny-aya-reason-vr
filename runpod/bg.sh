#!/bin/bash
# Run a long step detached from the terminal, logging to logs/<name>.log.
#   bash runpod/bg.sh stage2 run/05_grpo_mix.sh
#   tail -f logs/stage2.log          (tmux attach -t stage2 to watch it live)
set -e
cd "$(dirname "$0")/.."; mkdir -p logs
NAME=$1; shift
tmux new-session -d -s "$NAME" "bash -lc '$* 2>&1 | tee logs/$NAME.log'"
echo "started '$*' in tmux session '$NAME' -> logs/$NAME.log"
