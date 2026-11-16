#!/bin/bash
# usage: run/merge_lora.sh BASE_MODEL ADAPTER_DIR OUT_DIR
set -eo pipefail; source "$(dirname "$0")/common.sh"
"$TRAIN/bin/python" scripts/merge_lora.py --base "$1" --adapter "$2" --out "$3"
