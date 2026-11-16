"""Multilingual math candidates: list configs/splits and save train splits to data/cands/."""
import json, itertools
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import list_repo_files
C = "data/cands"
for rid in ["sarvamai/gsm8k-indic", "math-across-languages/gsm8k-translated", "llama-lang-adapt/Translated-GSM8K-balanced",
            "llama-lang-adapt/Translated-GSM8K", "OpenLLM-France/mgsm-rev2-with-train", "sam2ai/reasoning-multilingual-R1-indic-train",
            "uncoalesced-org/indic-reasoning-eval", "sarvamai/mmlu-indic"]:
    try:
        cfgs = get_dataset_config_names(rid)
    except Exception as e:
        print("!!!", rid, repr(e)[:200]); continue
    print("###", rid, "configs", cfgs[:40], flush=True)
    rows = []
    for c in cfgs[:40]:
        try:
            sps = get_dataset_split_names(rid, c)
        except Exception as e:
            print("  !!", c, repr(e)[:120]); continue
        print("  cfg", c, "splits", sps, flush=True)
        for sp in sps:
            try:
                ds = load_dataset(rid, c, split=sp, streaming=True)
                got = list(itertools.islice(iter(ds), 20000))
                print(f"    {c}/{sp}: {len(got)} cols={list(got[0].keys()) if got else None}", flush=True)
                if got and c == cfgs[0] and sp == sps[0]:
                    print("    EX", json.dumps(got[0], ensure_ascii=False, default=str)[:700], flush=True)
                rows += [dict(r, _cfg=c, _split=sp) for r in got]
            except Exception as e:
                print("    !!", c, sp, repr(e)[:150])
    if rows:
        name = rid.split("/")[1].lower().replace("-", "_")
        with open(f"{C}/ml_{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print("saved", name, len(rows), flush=True)
