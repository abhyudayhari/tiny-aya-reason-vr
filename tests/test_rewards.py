"""Sanity check: gold solutions must score 1.0 under rewards.score."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from rewards import score
C = "data/cands"
def first(n, k=30):
    out = []
    for l in open(f"{C}/{n}.jsonl"):
        out.append(json.loads(l))
        if len(out) >= k: break
    return out
ok = 0; rows = first("kodcode_rl10k")
for r in rows:
    if "from solution import" not in r["test"]: continue
    s = score({"verifier": "code_pytest", "meta": {"test": r["test"]}}, "```python\n" + r["solution"] + "\n```")
    ok += s
    if not s: print("KOD FAIL", r["test"][:200].replace("\n", " | "))
print("kodcode gold pass", ok, "of", len(rows))
ok = 0; rows = first("mbpp_train")
for r in rows:
    ok += score({"verifier": "code_assert", "meta": {"tests": r["test_list"], "setup": r["test_setup_code"]}}, "```python\n" + r["code"] + "\n```")
print("mbpp gold pass", ok, "of", len(rows))
ok = n = 0
for r in first("oro1_code", 30):
    v = r["verification_info"]; v = json.loads(v) if isinstance(v, str) else v
    if v["test_cases"][0]["type"] != "stdin_stdout": continue
    n += 1; ok += score({"verifier": "code_io", "meta": {"tests": v["test_cases"]}}, r["gold_standard_solution"])
print("oro gold pass", ok, "of", n)
