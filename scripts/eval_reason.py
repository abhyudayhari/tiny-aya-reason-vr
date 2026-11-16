"""PS6 coverage beyond math: code (HumanEval, MBPP), logic (BBH logic tasks, Knights & Knaves), finance (FinQA).

Same setup as eval_math.py: AuditKIT + vLLM, greedy, 4096 new tokens, zero-shot step-by-step prompt.
Code is scored by running the official tests in a subprocess (10 s timeout, 2 GB memory cap, temp dir).
Usage: eval_reason.py --model PATH --tag NAME --sets humaneval,mbpp,bbh,kk,finqa
"""
import argparse, json, os, re, resource, subprocess, sys, tempfile, time
from collections import defaultdict

import auditkit as ak
from auditkit.model.vllm_gen import VLLMModel
from auditkit.types import Direction

D = "."
R = f"{D}/data/raw_reason"
BOX = "\n\nThink step by step, and put your final answer within \\boxed{}."
CODE = ("\n\nThink step by step about the approach first. Then write the complete solution (signature, imports and body) "
        "in a single ```python code block.")


def rows(name):
    return [json.loads(l) for l in open(f"{R}/{name}.jsonl")]


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


def code_block(text):
    bl = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return bl[-1] if bl else None


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    os.setsid()


def run_py(src):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True, timeout=10, preexec_fn=_limits)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def norm(s):
    s = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", lambda m: m.group(1) or m.group(2), s or "")
    return s.strip().strip("$. ").strip("()").strip().lower()


def num(s):
    s = re.sub(r"\\%|%|\$|,|\\text\{[^}]*\}|\\,", "", s or "")
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    if not m:
        return None
    t = m.group(0)
    if "/" in t:
        a, b = t.split("/")
        return float(a) / float(b) if float(b) else None
    return float(t)


# ---- per-set builders: return list of (id, prompt, item, group) and a scorer(output, item)->bool ----
def build(name):
    if name == "humaneval":
        items = rows("humaneval")
        def sc(o, it):
            c = code_block(o)
            if c is None:
                return False
            if f"def {it['entry_point']}" not in c:
                c = it["prompt"] + c
            return run_py(c + "\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n")
        return [(it["task_id"], "Complete the following Python function.\n\n```python\n" + it["prompt"] + "```" + CODE, it, "humaneval")
                for it in items], sc
    if name == "mbpp":
        items = rows("mbpp")
        def sc(o, it):
            c = code_block(o)
            if c is None:
                return False
            return run_py("\n".join(it["test_imports"]) + "\n" + c + "\n\n" + "\n".join(it["test_list"]) + "\n")
        return [(f"mbpp-{it['task_id']}", it["prompt"] + "\nYour code should pass these tests:\n" + "\n".join(it["test_list"]) + CODE,
                 it, "mbpp") for it in items], sc
    if name == "bbh":
        items = rows("bbh")
        def sc(o, it):
            b = last_boxed(o)
            if b is None:
                return False
            t = it["target"]
            if it["task"] == "multistep_arithmetic_two":
                v = num(b)
                return v is not None and abs(v - float(t)) < 1e-6
            nb, nt = norm(b), norm(t)
            if re.fullmatch(r"\([a-g]\)", t.lower()):
                return nb[:1] == nt and (len(nb) == 1 or not nb[1].isalpha())
            return nb == nt
        return [(f"bbh-{it['task']}-{i}", it["input"] + BOX, it, it["task"]) for i, it in enumerate(items)], sc
    if name == "kk":
        items = rows("kk_test")
        def sc(o, it):
            tail = (last_boxed(o) or o[-600:]).lower()
            for nm, kn in zip(it["names"], it["solution"]):
                m = re.search(re.escape(nm.lower()) + r"\W+(?:is\W+)?(?:a\W+)?(knight|knave)", tail)
                if not m or (m.group(1) == "knight") != bool(kn):
                    return False
            return True
        pr = ("\n\nThink step by step. At the end, give the identity of every inhabitant inside \\boxed{}, "
              "e.g. \\boxed{Alice is a knight, Bob is a knave}.")
        return [(f"kk-{i}", it["quiz"] + pr, it, "kk-5ppl") for i, it in enumerate(items)], sc
    if name == "finqa":
        items = rows("finqa")
        def sc(o, it):
            b = last_boxed(o)
            if b is None:
                return False
            g = it["answer"].strip().lower()
            if g in ("yes", "no"):
                return norm(b).startswith(g)
            v, p = float(g), num(b)
            if p is None:
                return False
            cands = [p, p / 100, p * 100]
            return any(abs(c - v) <= max(0.01 * abs(v), 5e-4) for c in cands)
        def prompt(q):
            q = re.sub(r"\nAnswer:\s*$", "", q.rstrip())
            return q + ("\n\nThink step by step, and put your final numeric answer within \\boxed{}. "
                        "Give ratios and percentage changes as decimals or percentages.")
        return [(it["id"], prompt(it["query"]), it, "finqa") for it in items], sc
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="humaneval,mbpp,bbh,kk,finqa")
    a = ap.parse_args()
    model = VLLMModel(model=a.model, max_model_len=8192, gpu_memory_utilization=0.85, dtype="bfloat16")
    cfg = ak.RunConfig(temperature=0.0, max_tokens=4096, seed=0)
    out = f"{D}/results/{a.tag}"
    os.makedirs(out, exist_ok=True)
    for name in a.sets.split(","):
        spec, sc = build(name)
        by_id = {sid: (it, grp) for sid, _, it, grp in spec}

        @ak.scorer(direction=Direction.MAXIMIZE)
        def acc(sample, output):
            it, _ = by_id[sample.metadata["sid"]]
            return 1.0 if sc(output, it) else 0.0

        samples = [ak.Sample(input=p, target="", id=sid, metadata={"sid": sid}) for sid, p, _, _ in spec]
        t0 = time.time()
        res = ak.evaluate(samples, model, [acc], config=cfg, cache=False)
        groups = defaultdict(list)
        with open(f"{out}/{name}.jsonl", "w") as f:
            for pr in res.predictions:
                it, grp = by_id[pr.sample_id]
                ok = sc(pr.raw_output or "", it)
                groups[grp].append(ok)
                f.write(json.dumps({"id": pr.sample_id, "group": grp, "correct": ok, "output": pr.raw_output}, ensure_ascii=False) + "\n")
        allv = [x for v in groups.values() for x in v]
        summ = {"model": a.model, "set": name, "n": len(allv), "accuracy": round(100 * sum(allv) / len(allv), 2),
                      "breakdown": {k: {"n": len(v), "accuracy": round(100 * sum(v) / len(v), 2)} for k, v in sorted(groups.items())},
                      "seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(summ), flush=True)
        json.dump(summ, open(f"{out}/{name}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
