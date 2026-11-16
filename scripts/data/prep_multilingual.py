"""Build a multilingual math CoT SFT slice from mCoT-MATH for the 10 MGSM languages.

mCoT-MATH rows are machine-translated MetaMathQA solutions; `lang-id` = "<lang>-<MetaMathQA index>".
Gold answers come from MetaMathQA ("The answer is: X"). A row is kept only if the gold answer
string appears near the end of the translated solution (guards against misaligned rows), the
problem is GSM-style, and it shares no 13-gram with any eval set. Output: alpaca jsonl with a
final \\boxed{} line so the format matches the English data and the eval prompt.
"""
import json, random, re, sys
from collections import Counter, defaultdict
from datasets import load_dataset
from huggingface_hub import hf_hub_download

D = "data"
LANGS = ["bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
PER_LANG = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
random.seed(0)


def grams(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def cgrams(text, n=25):  # character n-grams for scripts written without spaces
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


bad_w, bad_c = set(), set()
for name in ("gsm8k", "math500", "mgsm"):
    for l in open(f"{D}/eval_{name}.jsonl"):
        q = json.loads(l)["question"]
        bad_w |= grams(q); bad_c |= cgrams(q)

mm = load_dataset("meta-math/MetaMathQA", split="train")
gold, mtype = {}, {}
for i, r in enumerate(mm):
    m = re.search(r"The answer is:?\s*(.+?)\s*$", r["response"])
    if m:
        gold[i] = m.group(1).strip()
        mtype[i] = r["type"]
print("metamath answers", len(gold), flush=True)

pools, stats = defaultdict(list), Counter()
for part in ("train_part_01.jsonl", "train_part_02.jsonl", "train_part_03.jsonl"):
    p = hf_hub_download("laihuiyuan/mCoT-MATH", part, repo_type="dataset")
    for l in open(p):
        r = json.loads(l)
        lang, _, idx = r["lang-id"].partition("-")
        stats["seen"] += 1
        if lang not in LANGS or not idx.isdigit():
            continue
        idx = int(idx)
        g = gold.get(idx)
        if g is None or not mtype.get(idx, "").startswith("GSM"):
            stats["not_gsm_or_no_gold"] += 1; continue
        q, a = r["question"].strip(), r["answer"].strip()
        if g.replace(",", "") not in a[-250:].replace(",", ""):
            stats["gold_not_in_tail"] += 1; continue
        if len(a) > 2000:
            stats["too_long"] += 1; continue
        if grams(q) & bad_w or cgrams(q) & bad_c:
            stats["contaminated"] += 1; continue
        pools[lang].append((q, a, g))
    print(part, dict(stats), {k: len(v) for k, v in pools.items()}, flush=True)

out = []
for lang in LANGS:
    rows = pools[lang]; random.shuffle(rows)
    for q, a, g in rows[:PER_LANG]:
        out.append({"instruction": q + SUFFIX, "input": "",
                    "output": a + f"\n\n$\\boxed{{{g}}}$", "source_dataset": f"mcot-{lang}", "answer": g})
random.shuffle(out)
with open(f"{D}/sft_ml_stage1_alpaca.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(out), Counter(r["source_dataset"] for r in out))
