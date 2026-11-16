"""Download candidate extra benchmarks + IndicFin dev; print schema and a sample row for each."""
import json, os
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
OUT = "data/raw_extra"; os.makedirs(OUT, exist_ok=True)
C = [("svamp", "ChilleD/SVAMP", None, "test"), ("asdiv", "EleutherAI/asdiv", None, "validation"),
     ("gsm_hard", "reasoning-machines/gsm-hard", None, "train"), ("minerva", "math-ai/minervamath", None, "test"),
     ("olympiad", "math-ai/olympiadbench", None, "test"), ("amc23", "math-ai/amc23", None, "test"),
     ("aime24", "HuggingFaceH4/aime_2024", None, "train"), ("aime25", "math-ai/aime25", None, "test"),
     ("gsm_plus", "qintongli/GSM-Plus", None, "testmini"), ("college_math", "math-ai/collegemath", None, "test")]
for name, rid, cfg, split in C:
    try:
        ds = load_dataset(rid, cfg, split=split)
        ds.to_json(f"{OUT}/{name}.jsonl", force_ascii=False)
        print("OK", name, rid, split, len(ds), ds.column_names, json.dumps(ds[0], ensure_ascii=False, default=str)[:400])
    except Exception as e:
        print("FAIL", name, rid, type(e).__name__, str(e)[:200])
try:
    files = list_repo_files("tanmay891/indicfin-data", repo_type="dataset"); print("INDICFIN files", files)
    p = hf_hub_download("tanmay891/indicfin-data", "data/dev.jsonl", repo_type="dataset", local_dir=OUT + "/indicfin")
    rows = [json.loads(l) for l in open(p)]
    print("INDICFIN dev rows", len(rows), "keys", list(rows[0].keys()))
    for r in rows[:3]: print("INDICFIN ROW", json.dumps(r, ensure_ascii=False)[:1200])
    from collections import Counter
    for k in rows[0]:
        v = [str(r.get(k))[:40] for r in rows]
        if len(set(v)) < 30: print("INDICFIN field", k, Counter(v).most_common(12))
except Exception as e:
    print("INDICFIN FAIL", type(e).__name__, str(e)[:300])
