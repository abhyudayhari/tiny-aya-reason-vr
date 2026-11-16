#!/bin/bash
# Download tiny-aya-global (gated: accept the licence on the Hub, export HF_TOKEN) and build the training base.
set -eo pipefail; source "$(dirname "$0")/common.sh"
"$TRAIN/bin/python" scripts/data/prep_base.py
