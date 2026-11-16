"""Prepare SFT data for CuratorKIT: keep instruction/output only (the alpaca reader would otherwise use an `answer`
column as the response) and split into shards: data/clean/en<i>.jsonl (English) and data/clean/<lang>.jsonl (multilingual)."""
import json, os, sys
os.makedirs("data/clean", exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
def row(r):
    return json.dumps({"instruction": r["instruction"], "input": "", "output": r["output"],
                       "source_dataset": r.get("source_dataset", "")}, ensure_ascii=False) + "\n"
fs = [open(f"data/clean/en{i}.jsonl", "w") for i in range(N)]
for i, l in enumerate(open("data/sft_stage1_alpaca.jsonl")):
    fs[i % N].write(row(json.loads(l)))
ml = {}
if os.path.exists("data/sft_ml_stage1_alpaca.jsonl"):
    for l in open("data/sft_ml_stage1_alpaca.jsonl"):
        r = json.loads(l); k = r.get("source_dataset") or "ml"
        ml.setdefault(k, open(f"data/clean/{k}.jsonl", "w")).write(row(r))
for f in fs + list(ml.values()):
    f.close()
print("shards:", N, "english +", len(ml), "multilingual")
