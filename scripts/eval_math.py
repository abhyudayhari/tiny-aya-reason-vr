"""Evaluate a model on GSM8K, MATH-500 and MGSM with AuditKIT (vLLM backend).

Greedy decoding, one shared prompt for every model, answers checked with math-verify.
Usage: eval_math.py --model PATH --tag NAME [--sets gsm8k,math500,mgsm] [--limit N]
"""
import argparse, json, os, re, time
from collections import defaultdict

import auditkit as ak
from auditkit.model.vllm_gen import VLLMModel
from auditkit.types import Direction
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

D = "."
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def load(name, limit):
    rows = [json.loads(l) for l in open(f"{D}/data/eval_{name}.jsonl")]
    return rows[:limit] if limit else rows


def gold_parse(ans):
    return parse(f"${ans}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])


def pred_parse(text):
    # Prefer the last \boxed{}, fall back to the last number/expression in the output.
    return parse(text, extraction_config=[
        LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])


def is_correct(output, ans):
    try:
        g, p = gold_parse(ans), pred_parse(output or "")
        return bool(p) and verify(g, p)
    except Exception:
        return False


@ak.scorer(direction=Direction.MAXIMIZE)
def math_verify_acc(sample, output):
    return 1.0 if is_correct(output, sample.target) else 0.0


@ak.scorer(direction=Direction.MAXIMIZE)
def has_boxed(sample, output):
    return 1.0 if "\\boxed" in (output or "") else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="gsm8k,math500,mgsm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()

    model = VLLMModel(model=a.model, max_model_len=8192, gpu_memory_utilization=0.85,
                      dtype="bfloat16")
    cfg = ak.RunConfig(temperature=0.0, max_tokens=a.max_tokens, seed=0)
    out_dir = f"{D}/results/{a.tag}"
    os.makedirs(out_dir, exist_ok=True)
    summary = {"model": a.model, "tag": a.tag, "prompt_suffix": SUFFIX, "decoding": "greedy",
               "max_tokens": a.max_tokens, "sets": {}}
    for name in a.sets.split(","):
        rows = load(name, a.limit)
        samples = [ak.Sample(input=r["question"] + SUFFIX, target=r["answer"], id=r["id"],
                             metadata={k: v for k, v in r.items() if k not in ("question", "answer")})
                   for r in rows]
        t0 = time.time()
        res = ak.evaluate(samples, model, [math_verify_acc, has_boxed], config=cfg, cache=False)
        dt = time.time() - t0
        preds = res.predictions
        by_id = {r["id"]: r for r in rows}
        recs, groups = [], defaultdict(list)
        for p in preds:
            sid, out = p.sample_id, p.raw_output or ""
            r = by_id.get(sid, {})
            ok = is_correct(out, r.get("answer", ""))
            recs.append({"id": sid, "answer": r.get("answer"), "correct": ok, "output": out})
            groups[r.get("lang") or r.get("level") or "all"].append(ok)
        with open(f"{out_dir}/{name}.jsonl", "w") as f:
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        acc = sum(x["correct"] for x in recs) / max(1, len(recs))
        summary["sets"][name] = {
            "n": len(recs), "accuracy": round(100 * acc, 2),
            "auditkit_headline": {k: (round(100 * v, 2) if isinstance(v, float) else v)
                                  for k, v in (res.headline or {}).items()},
            "breakdown": {k: round(100 * sum(v) / len(v), 2) for k, v in sorted(groups.items())},
            "seconds": round(dt, 1), "auditkit_fingerprint": getattr(res, "fingerprint", None)}
        print(name, json.dumps(summary["sets"][name]), flush=True)
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2)
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
