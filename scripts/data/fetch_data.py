"""Download eval benchmarks and training sources to data/ as jsonl."""
import json, os, sys
from datasets import load_dataset
D = "data"
def dump(rows, name):
    with open(f"{D}/{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), flush=True)

which = sys.argv[1:] or ["eval", "rl", "sft"]
if "eval" in which:
    g = load_dataset("openai/gsm8k", "main", split="test")
    dump([{"id": f"gsm8k-{i}", "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(g)], "eval_gsm8k")
    m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dump([{"id": r["unique_id"], "question": r["problem"], "answer": r["answer"], "level": r["level"], "subject": r["subject"]} for r in m], "eval_math500")
    rows = []
    for lang in ["bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]:
        ds = load_dataset("juletxara/mgsm", lang, split="test")
        rows += [{"id": f"mgsm-{lang}-{i}", "lang": lang, "question": r["question"], "answer": str(r["answer_number"])} for i, r in enumerate(ds)]
    dump(rows, "eval_mgsm")
if "rl" in which:
    g = load_dataset("openai/gsm8k", "main", split="train")
    rows = [{"id": f"gsm8k-train-{i}", "source": "gsm8k", "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(g)]
    from math_verify import parse  # noqa
    m = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    import re
    def boxed(s):
        i = s.rfind("\\boxed")
        if i < 0: return None
        j = s.find("{", i); depth = 0
        for k in range(j, len(s)):
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            if depth == 0: return s[j+1:k]
    for i, r in enumerate(m):
        a = boxed(r["solution"])
        if a: rows.append({"id": f"math-train-{i}", "source": "math", "level": r["level"], "question": r["problem"], "answer": a})
    dump(rows, "rl_pool")
if "sft" in which:
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    dump([{"question": r["problem"], "solution": r["generated_solution"], "answer": r["expected_answer"], "source": r["problem_source"]} for r in ds], "sft_raw_omi2_1M")
