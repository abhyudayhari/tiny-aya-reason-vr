"""Download code / logic / finance eval sets for PS6 coverage and probe candidate training datasets on the Hub.

Writes raw rows to data/raw_reason/<name>.jsonl and a probe report to data/raw_reason/probe.json.
"""
import json, os, traceback
from datasets import load_dataset
from huggingface_hub import HfApi, list_repo_files

R = "data/raw_reason"
os.makedirs(R, exist_ok=True)


def dump(rows, name):
    with open(f"{R}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print("saved", name, len(rows), flush=True)


def first_ok(name, tries):
    for args, kw in tries:
        try:
            ds = load_dataset(*args, **kw)
            print("loaded", name, "from", args, kw, "cols", ds.column_names, flush=True)
            return ds
        except Exception as e:
            print("fail", name, args, kw, repr(e)[:200], flush=True)
    return None


ds = first_ok("humaneval", [(("openai/openai_humaneval",), {"split": "test"})])
if ds: dump(list(ds), "humaneval")
ds = first_ok("mbpp", [(("google-research-datasets/mbpp", "sanitized"), {"split": "test"}), (("mbpp", "sanitized"), {"split": "test"})])
if ds: dump(list(ds), "mbpp")

BBH = ["boolean_expressions", "formal_fallacies", "logical_deduction_five_objects", "logical_deduction_seven_objects",
       "web_of_lies", "navigate", "tracking_shuffled_objects_five_objects", "multistep_arithmetic_two"]
rows = []
for t in BBH:
    ds = first_ok("bbh-" + t, [(("lukaemon/bbh", t), {"split": "test"}), (("maveriq/bigbenchhard", t), {"split": "train"})])
    if ds:
        rows += [dict(r, task=t) for r in ds]
if rows: dump(rows, "bbh")

ds = first_ok("finqa", [(("TheFinAI/flare-finqa",), {"split": "test"}), (("ChanceFocus/flare-finqa",), {"split": "test"}),
                        (("ibm-research/finqa",), {"split": "test"}), (("dreamerdeo/finqa",), {"split": "test"})])
if ds: dump(list(ds), "finqa")

ds = first_ok("kk", [(("K-and-K/knights-and-knaves", "test"), {"split": "5ppl"}), (("K-and-K/knights-and-knaves", "test"), {"split": "4ppl"})])
if ds: dump(list(ds), "kk_test")

# IndicFin training pool (same repo as the dev set)
try:
    fs = list_repo_files("tanmay891/indicfin-data", repo_type="dataset")
    print("indicfin files", fs, flush=True)
except Exception as e:
    print("indicfin list fail", repr(e)[:200])

CANDS = {
    "math": ["BytedTsinghua-SIA/DAPO-Math-17k", "agentica-org/DeepScaleR-Preview-Dataset", "SynthLabsAI/Big-Math-RL-Verified",
             "open-r1/OpenR1-Math-220k", "AI-MO/NuminaMath-1.5", "nvidia/OpenMathReasoning", "PRIME-RL/Eurus-2-RL-Data"],
    "finance": ["TheFinAI/flare-finqa", "ibm-research/finqa", "next-tat/TAT-QA", "TheFinAI/Fino1_Reasoning_Path_FinQA",
                "AdaptLLM/finance-tasks", "sujet-ai/Sujet-Finance-Instruct-177k", "tanmay891/indicfin-data"],
    "code": ["open-r1/verifiable-coding-problems-python", "KodCode/KodCode-Light-RL-10K",
             "KodCode/KodCode-V1", "likaixin/TACO-verified", "deepmind/code_contests"],
    "logic": ["K-and-K/knights-and-knaves", "MiniMaxAI/SynLogic", "allenai/ZebraLogicBench", "lucasmccabe/logiqa", "hitachi-nlp/FLD.v2",
              "open-thought/reasoning-gym"],
    "multilingual": ["juletxara/mgsm", "openai/MMMLU", "CohereLabs/Global-MMLU"],
}
api = HfApi()
probe = {}
for cat, ids in CANDS.items():
    for rid in ids:
        try:
            info = api.dataset_info(rid)
            probe[rid] = {"cat": cat, "ok": True, "downloads": info.downloads, "likes": info.likes, "gated": info.gated,
                          "license": next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None),
                          "size": next((t for t in (info.tags or []) if t.startswith("size_categories:")), None)}
        except Exception as e:
            probe[rid] = {"cat": cat, "ok": False, "err": repr(e)[:150]}
        print(rid, probe[rid], flush=True)
json.dump(probe, open(f"{R}/probe.json", "w"), indent=1)
