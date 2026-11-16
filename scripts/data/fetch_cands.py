"""Download candidate RL training datasets (raw) to data/cands/ and search the Hub for more.

For each candidate: load (streaming, capped), save raw rows, print columns + one example so formats can be checked.
"""
import json, os, sys, itertools, traceback
from datasets import load_dataset
from huggingface_hub import HfApi, list_repo_files, hf_hub_download

C = "data/cands"
os.makedirs(C, exist_ok=True)
CAP = 60000


def save(name, it, cap=CAP):
    n = 0
    ex = None
    with open(f"{C}/{name}.jsonl", "w") as f:
        for r in itertools.islice(it, cap):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            ex = ex or r
            n += 1
    print(f"### {name}: {n} rows; cols={list(ex.keys()) if ex else None}", flush=True)
    if ex:
        print("EXAMPLE", json.dumps(ex, ensure_ascii=False, default=str)[:1500], flush=True)


def try_load(name, *args, cap=CAP, **kw):
    try:
        ds = load_dataset(*args, streaming=True, **kw)
        save(name, iter(ds), cap)
    except Exception as e:
        print(f"!!! {name} {args} {kw}: {repr(e)[:300]}", flush=True)


def configs(rid):
    try:
        from datasets import get_dataset_config_names
        return get_dataset_config_names(rid)
    except Exception as e:
        return f"ERR {repr(e)[:150]}"


which = sys.argv[1:] or ["math", "fin", "logic", "code", "search"]
api = HfApi()

if "math" in which:
    try_load("deepscaler", "agentica-org/DeepScaleR-Preview-Dataset", split="train")
    try_load("dapo17k", "BytedTsinghua-SIA/DAPO-Math-17k", split="train", cap=1800000)  # heavily duplicated; dedup later
    try_load("bigmath", "SynthLabsAI/Big-Math-RL-Verified", split="train", cap=260000)
    try_load("openr1math", "open-r1/OpenR1-Math-220k", "default", split="train", cap=100000)
    try_load("numina15", "AI-MO/NuminaMath-1.5", split="train", cap=900000)
    try_load("orz", "Open-Reasoner-Zero/orz_math_57k_collection", split="train")
    try_load("skywork_math", "Skywork/Skywork-OR1-RL-Data", split="math")

if "fin" in which:
    for fn in ["data/rl_pool.jsonl", "data/sft.jsonl"]:
        try:
            p = hf_hub_download("tanmay891/indicfin-data", fn, repo_type="dataset")
            rows = [json.loads(l) for l in open(p)]
            save("indicfin_" + os.path.basename(fn)[:-6], iter(rows))
        except Exception as e:
            print("!!! indicfin", fn, repr(e)[:200])
    print("finqa configs", configs("ibm-research/finqa"))
    try_load("finqa_train", "ibm-research/finqa", split="train")
    print("flare-finqa files", list_repo_files("ChanceFocus/flare-finqa", repo_type="dataset"))
    try_load("flare_finqa_train", "ChanceFocus/flare-finqa", split="train")
    print("tatqa files", list_repo_files("next-tat/TAT-QA", repo_type="dataset"))
    try_load("tatqa_train", "next-tat/TAT-QA", split="train")
    try_load("convfinqa", "AdaptLLM/ConvFinQA", split="train")
    try_load("fino1_finqa", "TheFinAI/Fino1_Reasoning_Path_FinQA", split="train")
    try_load("finqa_r1", "TheFinAI/FinQA-R1", split="train")
    try_load("fin_r1_data", "SUFE-AIFLM-Lab/Fin-R1-Data", split="train")

if "logic" in which:
    print("synlogic configs", configs("MiniMaxAI/SynLogic"))
    for cfg in ["easy", "hard"]:
        try_load(f"synlogic_{cfg}", "MiniMaxAI/SynLogic", cfg, split="train")
    print("kk configs", configs("K-and-K/knights-and-knaves"))
    rows = []
    for sp in ["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"]:
        try:
            ds = load_dataset("K-and-K/knights-and-knaves", "train", split=sp)
            rows += [dict(r, split=sp) for r in ds]
        except Exception as e:
            print("!!! kk", sp, repr(e)[:150])
    if rows:
        save("kk_train", iter(rows))
    try_load("logiqa_train", "lucasmccabe/logiqa", split="train")
    try_load("fld2", "hitachi-nlp/FLD.v2", "default", split="train", cap=20000)
    try_load("zebra_grid", "allenai/ZebraLogicBench-private", "grid_mode", split="test", cap=2000)
    try_load("skywork_code_logic", "Skywork/Skywork-OR1-RL-Data", split="code", cap=20000)

if "code" in which:
    try_load("oro1_code", "open-r1/verifiable-coding-problems-python", split="train")
    try_load("taco_verified", "likaixin/TACO-verified", split="train")
    try_load("kodcode_rl10k", "KodCode/KodCode-Light-RL-10K", split="train")
    try_load("mbpp_train", "google-research-datasets/mbpp", "full", split="train")
    try_load("eurus2", "PRIME-RL/Eurus-2-RL-Data", split="train", cap=500000)

if "search" in which:
    seen = set()
    for q in ["hindi math", "indic math", "gsm8k hindi", "multilingual gsm8k", "mgsm train", "hindi gsm8k", "bengali math",
              "multilingual math reasoning", "indic reasoning", "hinglish", "finance reasoning", "financial math",
              "fin r1", "logic reasoning rl", "logic puzzles", "verifiable rewards", "rlvr", "reasoning gym", "zebra puzzle",
              "multilingual reasoning rl", "gsm8k translated", "math hindi", "sarvam", "ai4bharat"]:
        try:
            for d in api.list_datasets(search=q, sort="downloads", limit=12):
                if d.id in seen:
                    continue
                seen.add(d.id)
                print(f"SEARCH[{q}] {d.id} dl={d.downloads} likes={d.likes} gated={getattr(d, 'gated', None)}", flush=True)
        except Exception as e:
            print("!!! search", q, repr(e)[:150])
