"""Stage 1 of SFT data prep: filter + decontaminate OpenMathInstruct-2, write alpaca jsonl for CuratorKIT.

- keep solutions that end in a \\boxed{} answer and are short enough for a 3B model (<= 2400 chars)
- drop any problem sharing a 13-gram with GSM8K test, MATH-500 or MGSM (English and all languages)
- balanced sample across OMI-2 sources (gsm8k-derived vs math-derived)
"""
import json, random, re, sys
from collections import Counter, defaultdict

D = "data"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 240_000
random.seed(0)


def grams(text, n=13):
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


bad = set()
for name in ("gsm8k", "math500", "mgsm"):
    for l in open(f"{D}/eval_{name}.jsonl"):
        bad |= grams(json.loads(l)["question"])
print("eval 13-grams", len(bad))

by_src, stats = defaultdict(list), Counter()
for l in open(f"{D}/sft_raw_omi2_1M.jsonl"):
    r = json.loads(l)
    stats["seen"] += 1
    sol = r["solution"]
    if "\\boxed" not in sol[-300:]:
        stats["no_boxed"] += 1; continue
    if len(sol) > 2400 or len(r["question"]) > 1500:
        stats["too_long"] += 1; continue
    if grams(r["question"]) & bad:
        stats["contaminated"] += 1; continue
    by_src[r["source"]].append(r)
print(dict(stats), {k: len(v) for k, v in by_src.items()})

# OMI-2 sources: gsm8k / augmented_gsm8k / math / augmented_math. Half GSM-style, half MATH-style.
gsm = by_src.get("gsm8k", []) + by_src.get("augmented_gsm8k", [])
mth = by_src.get("math", []) + by_src.get("augmented_math", [])
random.shuffle(gsm); random.shuffle(mth)
pick = gsm[: N_TARGET * 2 // 5] + mth[: N_TARGET * 3 // 5]
random.shuffle(pick)
with open(f"{D}/sft_stage1_alpaca.jsonl", "w") as f:
    for r in pick:
        f.write(json.dumps({"instruction": r["question"] + SUFFIX, "input": "", "output": r["solution"],
                            "source_dataset": r["source"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
print("wrote", len(pick), Counter(r["source"] for r in pick))
