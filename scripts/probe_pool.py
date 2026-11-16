"""Measure how learnable each pool source is for the current policy: k samples per problem, verified reward.

GRPO only learns from prompts where some but not all samples are right, so the useful number per source is
`learnable` = share of problems with 0 < correct < k. Also reports mean pass rate and per-lang/task slices.
usage: probe_pool.py --model PATH --sources a,b,c --n 160 --k 8 --out results/probe/<name>.json
"""
import argparse, json, os, random, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from rewards import score

D = "."


def _cap():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))


def _score(args):
    item, texts = args
    return [score(item, t) for t in texts]


def sample(src, n):
    rows = [json.loads(l) for l in open(f"{D}/data/pool/{src}.jsonl")]
    random.Random(0).shuffle(rows)
    if src in ("synlogic", "gsm8k_translated_multi", "gsm8k_hi_fr_de", "kk_train", "indicfin_pool"):  # stratify by slice
        key = lambda r: r["meta"].get("task") or r["meta"].get("n") or r["lang"]
        by = defaultdict(list)
        for r in rows:
            by[key(r)].append(r)
        per = max(4, n // len(by))
        return [r for v in by.values() for r in v[:per]]
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", action="store_true", help="write every item with its correct-count (for building the RL mix)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--lang", default="", help="dump mode: comma list of langs to keep (prefix ! to exclude)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, max_model_len=8192, gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=0)
    report = {}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for src in a.sources.split(","):
        if a.dump:
            items = [json.loads(l) for l in open(f"{D}/data/pool/{src}.jsonl")]
            if a.lang:
                neg = a.lang.startswith("!"); langs = set(a.lang.lstrip("!").split(","))
                items = [it for it in items if (it["lang"] in langs) != neg]
            items = items[a.offset:a.offset + a.n]
        else:
            items = sample(src, a.n)
        items = [it for it in items if len(tok(it["prompt"])["input_ids"]) < 8192 - a.max_tokens - 64]
        outs = llm.chat([[{"role": "user", "content": it["prompt"]}] for it in items], sp, use_tqdm=False)
        texts = [[o.text for o in out.outputs] for out in outs]
        trunc = sum(o.finish_reason == "length" for out in outs for o in out.outputs) / max(1, len(items) * a.k)
        with ProcessPoolExecutor(10, initializer=_cap) as ex:
            scores = list(ex.map(_score, zip(items, texts), chunksize=4))
        slices = defaultdict(list)
        for it, s in zip(items, scores):
            c = sum(s)
            for key in ("all", "lang:" + it["lang"], "task:" + str(it["meta"].get("task") or it["meta"].get("n") or "")):
                slices[key].append(c)
        summ = lambda cs: {"n": len(cs), "mean_pass": round(sum(cs) / (a.k * len(cs)), 4),
                           "learnable": round(sum(0 < c < a.k for c in cs) / len(cs), 4),
                           "all_wrong": round(sum(c == 0 for c in cs) / len(cs), 4),
                           "all_right": round(sum(c == a.k for c in cs) / len(cs), 4)}
        report[src] = {"domain": items[0]["domain"] if items else None, "truncated": round(trunc, 4),
                       **summ(slices["all"]), "slices": {k: summ(v) for k, v in slices.items() if k != "all" and len(v) >= 4}}
        print(src, json.dumps({k: v for k, v in report[src].items() if k != "slices"}), flush=True)
        if a.dump:
            with open(a.out.replace(".json", f".{src}.dump.jsonl"), "w") as f:
                for it, s in zip(items, scores):
                    f.write(json.dumps(dict(it, correct=int(sum(s)), k=a.k), ensure_ascii=False) + "\n")
        with open(a.out.replace(".json", f".{src}.samples.jsonl"), "w") as f:
            for it, t, s in zip(items[:40], texts[:40], scores[:40]):
                f.write(json.dumps({"id": it["id"], "answer": it["answer"], "scores": s, "sample": t[0][-1500:]}, ensure_ascii=False) + "\n")
        json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
