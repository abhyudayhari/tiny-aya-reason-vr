"""Build the multi-domain GRPO training file from pass-rate dumps (results/probe/d*.<src>.dump.jsonl).

Keeps only learnable prompts (0 < correct < k), caps/upsamples each source to its target, drops long prompts,
and writes data/rl_mix.jsonl rows {id, source, domain, lang, prompt:[{role,content}], answer} where answer is
the plain reference for maths and a JSON verifier spec for every other domain (see train_grpo.math_verify_reward).
"""
import glob, json, random, sys
from collections import Counter, defaultdict

D = "."
random.seed(0)
# source -> target number of training prompts (upsampled with repetition if fewer learnable prompts exist, max 2x)
TARGET = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
    "deepscaler": 5000, "dapo17k": 3000,
    "gsm8k_hi_fr_de": 4200, "gsm8k_sw": 900,
    "indicfin_pool": 2400, "finqa_train": 1100, "tatqa_arith": 1500,
    "kk_train": 2700, "fld2": 1800,
    "mbpp_train": 400, "taco": 600, "kodcode": 1600,
}
MAX_PROMPT_CHARS = 5000  # ~1.4k tokens

by = defaultdict(list)
seen_ids = set()
for p in glob.glob(f"{D}/results/probe/d*.dump.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if r["id"] in seen_ids:
            continue
        seen_ids.add(r["id"])
        if 0 < r["correct"] < r["k"] and len(r["prompt"]) <= MAX_PROMPT_CHARS:
            by[r["source"]].append(r)

rows, report = [], {}
for src, tgt in TARGET.items():
    pool = by.get(src, [])
    if src == "gsm8k_hi_fr_de":  # favour Hindi: up to 3/4 of this source
        hi = [r for r in pool if r["lang"] == "hi"]; ot = [r for r in pool if r["lang"] != "hi"]
        random.shuffle(hi); random.shuffle(ot)
        pool = hi[:int(tgt * 0.75)] + ot[:tgt - min(len(hi), int(tgt * 0.75))]
    random.shuffle(pool)
    take = pool[:tgt]
    if take and len(take) < tgt:  # upsample small high-value sources (IndicFin, MBPP) at most 2x
        take = take + take[:min(len(take), tgt - len(take))]
    report[src] = {"learnable": len(by.get(src, [])), "used": len(take)}
    for r in take:
        ans = r["answer"] if r["verifier"] == "math" else json.dumps({"verifier": r["verifier"], "answer": r["answer"], "meta": r["meta"]}, ensure_ascii=False)
        rows.append({"id": r["id"], "source": src, "domain": r["domain"], "lang": r["lang"],
                     "prompt": [{"role": "user", "content": r["prompt"]}], "answer": ans, "pass_rate": r["correct"] / r["k"]})
random.shuffle(rows)
with open(f"{D}/data/rl_mix.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=1))
print("total", len(rows), "by domain", Counter(r["domain"] for r in rows), "by lang", Counter(r["lang"] for r in rows).most_common(8))
