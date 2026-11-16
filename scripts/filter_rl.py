"""Keep RL prompts the policy solves sometimes but not always (0 < pass@k-rate < 1).

GRPO gets zero advantage when all k samples agree, so those prompts waste rollouts.
usage: filter_rl.py --model PATH --k 8 --out data/rl_train.jsonl
"""
import argparse, json, random
from vllm import LLM, SamplingParams
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

D = "."


def ok(text, ans):
    try:
        g = parse(f"${ans}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        p = parse(text, extraction_config=[LatexExtractionConfig(boxed_match_priority=0)])
        return bool(p) and verify(g, p)
    except Exception:
        return False


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=1536)
a = ap.parse_args()

rows = [json.loads(l) for l in open(f"{D}/data/rl_all.jsonl")]
llm = LLM(a.model, max_model_len=4096, gpu_memory_utilization=0.9, seed=0)
sp = SamplingParams(n=a.k, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=0)
outs = llm.chat([r["prompt"] for r in rows], sp)
keep, hist = [], {}
for r, o in zip(rows, outs):
    c = sum(ok(x.text, r["answer"]) for x in o.outputs)
    hist[c] = hist.get(c, 0) + 1
    r["pass_rate"] = c / a.k
    if 0 < c < a.k:
        keep.append(r)
random.seed(0); random.shuffle(keep)
with open(a.out, "w") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("histogram(correct of k):", dict(sorted(hist.items())), "kept", len(keep), "of", len(rows))
