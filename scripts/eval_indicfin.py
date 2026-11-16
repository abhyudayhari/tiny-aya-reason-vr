"""Evaluate a model on IndicFin dev (150 Indian-finance word problems, en/hi/hinglish) with AuditKIT.

Correct = last \\boxed{} value within the item's own tolerance (|pred-gold| <= max(tol_abs, tol_rel*|gold|)),
or \\boxed{INSUFFICIENT} for items marked unanswerable. Devanagari digits, commas, Rs./INR/% are normalised.
Usage: eval_indicfin.py --model PATH --tag NAME
"""
import argparse, json, os, re, time
from collections import defaultdict

import auditkit as ak
from auditkit.model.vllm_gen import VLLMModel
from auditkit.types import Direction

D = "."
DEV = f"{D}/data/raw_extra/indicfin/data/dev.jsonl"
SUFFIX = ("\n\nPlease reason step by step, and put your final answer within \\boxed{}. "
          "If the question does not give enough information to compute the answer, reply with \\boxed{INSUFFICIENT}.")
DEVA = str.maketrans("०१२३४५६७८९", "0123456789")


def last_boxed(text):
    i = (text or "").rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def to_number(s):
    s = s.translate(DEVA)
    s = re.sub(r"\\text\{[^}]*\}|\\mathrm\{[^}]*\}|Rs\.?|INR|₹|rupees?|रुपये|%|\\%|\$|\\,|,|\s", "", s, flags=re.I)
    s = s.replace("\\frac", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def score(output, item):
    b = last_boxed(output)
    if b is None:
        return False, None
    t = item["target"]
    if t["type"] == "insufficient":
        return "INSUFFICIENT" in b.upper(), b
    if "INSUFFICIENT" in b.upper():
        return False, b
    p = to_number(b)
    if p is None:
        return False, b
    v = float(t["value"])
    tol = max(float(t.get("tol_abs") or 0), float(t.get("tol_rel") or 0) * abs(v))
    return abs(p - v) <= tol + 1e-9, b


@ak.scorer(direction=Direction.MAXIMIZE)
def indicfin_acc(sample, output):
    return 1.0 if score(output, sample.metadata["item"])[0] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(DEV)]
    model = VLLMModel(model=a.model, max_model_len=8192, gpu_memory_utilization=0.85, dtype="bfloat16")
    cfg = ak.RunConfig(temperature=0.0, max_tokens=4096, seed=0)
    samples = [ak.Sample(input=it["question"] + SUFFIX, target=str(it["target"]["value"]), id=it["id"],
                         metadata={"item": it}) for it in items]
    t0 = time.time()
    res = ak.evaluate(samples, model, [indicfin_acc], config=cfg, cache=False)
    by_id = {it["id"]: it for it in items}
    recs, groups = [], defaultdict(list)
    for p in res.predictions:
        it = by_id[p.sample_id]
        ok, b = score(p.raw_output or "", it)
        recs.append({"id": p.sample_id, "language": it["language"], "task_family": it["task_family"], "tier": it["tier"],
                     "answerable": it["answerable"], "gold": it["target"]["value"], "pred": b, "correct": ok,
                     "output": p.raw_output})
        groups["lang:" + it["language"]].append(ok)
        groups["task:" + it["task_family"]].append(ok)
        groups["tier:" + str(it["tier"])].append(ok)
        groups["answerable" if it["answerable"] else "unanswerable"].append(ok)
    out = f"{D}/results/{a.tag}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/indicfin.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(r["correct"] for r in recs) / len(recs)
    summ = {"model": a.model, "tag": a.tag, "n": len(recs), "accuracy": round(100 * acc, 2), "prompt_suffix": SUFFIX,
            "breakdown": {k: {"n": len(v), "accuracy": round(100 * sum(v) / len(v), 2)} for k, v in sorted(groups.items())},
            "auditkit_headline": res.headline, "seconds": round(time.time() - t0, 1)}
    json.dump(summ, open(f"{out}/indicfin_summary.json", "w"), indent=2, ensure_ascii=False)
    print("indicfin", json.dumps({k: summ[k] for k in ("n", "accuracy")}), json.dumps(summ["breakdown"], ensure_ascii=False))


if __name__ == "__main__":
    main()
