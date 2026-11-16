#!/bin/bash
# Optional: SFT data with CuratorKIT (schema/length gates, exact + MinHash dedup, text cleaning) -> data/sft_mix_messages.jsonl
set -eo pipefail; source "$(dirname "$0")/common.sh"; export PATH=$TRAIN/bin:$PATH
python scripts/data/fetch_data.py sft
python scripts/data/prep_sft.py                    # English: OpenMathInstruct-2, filtered + decontaminated
python scripts/data/prep_multilingual.py           # 10 MGSM languages: mCoT-MATH
python scripts/data/split_sft.py 8
rm -rf data/sft_curated
for f in data/clean/*.jsonl; do
  n=$(basename $f .jsonl); mt=900; case $n in mcot-*) mt=1400;; esac
  sed -e "s|^    path: .*|    path: $P/$f|" -e "s|max_tokens: 900 |max_tokens: $mt |" configs/curate_sft.yaml > data/clean/$n.yaml
  curatorkit run data/clean/$n.yaml --output-dir data/sft_curated/$n > logs/curate-$n.log 2>&1 &
done
wait
for d in data/sft_curated/*/; do echo "$(basename $d): $(wc -l < $d/sft_messages.jsonl) kept"; done
cat data/sft_curated/*/sft_messages.jsonl | shuf --random-source=<(yes) > data/sft_mix_messages.jsonl
wc -l data/sft_mix_messages.jsonl
