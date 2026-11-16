"""Normalise every candidate RL source into data/pool/<source>.jsonl with a verifier, after decontamination.

Row: {id, source, domain, lang, prompt (user message), answer, verifier, meta}
Decontamination: drop any row sharing a 13-word n-gram with any eval question (all math, code, logic, finance evals).
"""
import glob, json, os, random, re
from collections import Counter, defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rewards import BOX_SUFFIX, CODE_SUFFIX, FIN_SUFFIX

D = "."
C, O = f"{D}/data/cands", f"{D}/data/pool"
os.makedirs(O, exist_ok=True)
random.seed(0)


def rd(name, cap=None):
    p = f"{C}/{name}.jsonl"
    if not os.path.exists(p):
        print("missing", name); return
    with open(p) as f:
        for i, l in enumerate(f):
            if cap and i >= cap:
                break
            yield json.loads(l)


# ---------- decontamination index ----------
def grams(t, n=13):
    w = re.findall(r"\w+", (t or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


EVC = Counter()
class _Acc:
    def __ior__(self, g):
        EVC.update(g); return self
EV = _Acc()
for p in glob.glob(f"{D}/data/eval_*.jsonl"):
    for l in open(p):
        EV |= grams(json.loads(l)["question"])
for name, key in [("humaneval", "prompt"), ("mbpp", "prompt"), ("bbh", "input"), ("kk_test", "quiz"), ("finqa", "query")]:
    for l in open(f"{D}/data/raw_reason/{name}.jsonl"):
        r = json.loads(l)
        q = r[key]
        if name == "finqa":  # question part + context of the test documents
            q = q[-3000:]
        EV |= grams(q)
for l in open(f"{D}/data/raw_extra/indicfin/data/dev.jsonl"):
    EV |= grams(json.loads(l)["question"])
for r in rd("ml_gsm8k_indic") or []:  # GSM8K test translated: everything there is eval
    for k in ("question", "original_question"):
        EV |= grams(r.get(k))
# n-grams shared by 3+ eval items are template boilerplate (K&K intro, IndicFin phrasing), not leakage
EV = {g for g, c in EVC.items() if c < 3}
print("eval 13-grams:", len(EV), "boilerplate dropped:", sum(1 for c in EVC.values() if c >= 3), flush=True)


def clean(q):
    return not (grams(q) & EV)


stats = Counter()
def out(source, rows):
    seen, keep = set(), []
    for r in rows:
        if r["verifier"] == "code_io" and not r["meta"]["tests"]:
            stats[source + ":no_small_tests"] += 1; continue
        key = re.sub(r"\W+", " ", r["prompt"].lower())[:400]
        if key in seen:
            stats[source + ":dup"] += 1; continue
        seen.add(key)
        if not clean(r["prompt"]):
            stats[source + ":contaminated"] += 1; continue
        keep.append(r)
    random.shuffle(keep)
    with open(f"{O}/{source}.jsonl", "w") as f:
        for i, r in enumerate(keep):
            r.setdefault("id", f"{source}-{i}")
            r["source"] = source
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats[source] = len(keep)
    print(source, len(keep), {k: v for k, v in stats.items() if k.startswith(source + ":")}, flush=True)


def M(q, a, dom="math", lang="en", **meta):
    return {"domain": dom, "lang": lang, "prompt": q.strip() + BOX_SUFFIX, "answer": str(a).strip(), "verifier": "math", "meta": meta}


def ok_ans(a):
    a = str(a or "").strip()
    return a and len(a) < 60 and a.lower() not in ("proof", "notfound", "none", "") and "\\text{" not in a[:7]


# ---------- math ----------
out("deepscaler", (M(r["problem"], r["answer"]) for r in rd("deepscaler") if ok_ans(r["answer"])))
DAPO_PRE = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
out("dapo17k", (M(r["prompt"][0]["content"].replace(DAPO_PRE, "").split("\n\nRemember to put your answer")[0], r["reward_model"]["ground_truth"])
                for r in rd("dapo17k")))
def sky():
    for r in rd("skywork_math"):
        g = json.loads(r["reward_model"]["ground_truth"])
        g = g[0] if isinstance(g, list) else g
        if ok_ans(g):
            md = r["extra_info"].get("model_difficulty") or {}
            yield M(r["prompt"][-1]["content"], g, r1_7b=md.get("DeepSeek-R1-Distill-Qwen-7B"), r1_1_5b=md.get("DeepSeek-R1-Distill-Qwen-1.5B"))
out("skywork_math", sky())
out("openr1math", (M(r["problem"], r["answer"]) for r in rd("openr1math") if r.get("question_type") == "math-word-problem" and ok_ans(r["answer"])))
def numina():
    for r in rd("numina15"):
        if r.get("question_type") != "math-word-problem" or r.get("problem_is_valid") != "Yes" or r.get("solution_is_valid") != "Yes":
            continue
        if r.get("synthetic") or not ok_ans(r["answer"]):
            continue
        if r["source"] in ("olympiads", "amc_aime", "aops_forum", "cn_contest", "number_theory", "inequalities", "olympiads_ref"):
            yield M(r["problem"], r["answer"], numina_src=r["source"])
out("numina_comp", numina())

# ---------- multilingual math (GSM8K *train* translations only) ----------
def gsm_ans(a):
    m = re.search(r"####\s*([-\d.,]+)", a or "")
    return m.group(1).replace(",", "") if m else None
def mal():
    for r in rd("ml_gsm8k_translated"):
        if r["_split"] != "train":
            continue
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", r["answer"] or "")
        if nums:
            yield M(r["question"], nums[-1].replace(",", "").rstrip("."), lang=r["_cfg"])
out("gsm8k_hi_fr_de", mal())
def lla():
    for r in rd("ml_translated_gsm8k"):
        if r["_split"] != "train":
            continue
        q = r.get("target_question")
        a = gsm_ans(r.get("source_answer"))
        lang = r.get("lang") or "xx"
        if q and a:
            yield M(q, a, lang=lang)
out("gsm8k_translated_multi", lla())

# ---------- finance ----------
def indicfin():
    for r in rd("indicfin_rl_pool"):
        t = r["target"]
        yield {"domain": "finance", "lang": r["language"], "prompt": r["question"] + FIN_SUFFIX, "answer": str(t["value"]), "verifier": "fin",
               "meta": {"tol_abs": t.get("tol_abs"), "tol_rel": t.get("tol_rel"), "task": r["task_family"], "tier": r["tier"]}}
out("indicfin_pool", indicfin())
def finqa():
    for r in rd("flare_finqa_train"):
        q = re.sub(r"\nAnswer:\s*$", "", r["query"].rstrip())
        yield {"domain": "finance", "lang": "en", "prompt": q + "\n\nThink step by step, and put your final numeric answer within \\boxed{}. Give ratios and percentage changes as decimals or percentages.",
               "answer": r["answer"], "verifier": "fin", "meta": {}}
out("finqa_train", finqa())
def tatqa():
    for r in rd("tatqa_train"):
        tbl = "\n".join(" | ".join(map(str, row)) for row in r["table"]["table"])
        par = "\n".join(p["text"] for p in sorted(r["paragraphs"], key=lambda p: p["order"]))
        for q in r["questions"]:
            if q["answer_type"] not in ("arithmetic", "count"):
                continue
            a = q["answer"]
            try:
                float(str(a).replace(",", ""))
            except ValueError:
                continue
            sc = f" (answer in {q['scale']})" if q.get("scale") else ""
            yield {"domain": "finance", "lang": "en",
                   "prompt": f"Table:\n{tbl}\n\nText:\n{par}\n\nQuestion: {q['question']}{sc}" + BOX_SUFFIX,
                   "answer": str(a), "verifier": "fin", "meta": {"tol_rel": 0.01 if q["scale"] != "percent" else 0.005, "tol_abs": 0.01}}
out("tatqa_arith", tatqa())

# ---------- logic ----------
def kk():
    for r in rd("kk_train"):
        yield {"domain": "logic", "lang": "en", "prompt": r["quiz"] + "\n\nThink step by step. At the end, give the identity of every inhabitant inside \\boxed{}, e.g. \\boxed{Alice is a knight, Bob is a knave}.",
               "answer": r["solution_text"], "verifier": "kk", "meta": {"names": r["names"], "solution": r["solution"], "n": r["split"]}}
out("kk_train", kk())
SYN_OK = {"boolean_expressions", "web_of_lies", "word_sorting", "word_sorting_mistake", "dyck_language", "dyck_language_errors",
          "dyck_language_reasoning_errors", "object_counting", "object_properties", "operation", "space_reasoning", "buggy_tables",
          "time_sequence", "futoshiki", "sudoku", "survo", "calcudoko", "space_reasoning_tree", "cipher"}
GRID = {"futoshiki", "sudoku", "survo", "calcudoko", "time_sequence", "cipher"}
INSTR = re.compile(r"^Solve the following problem step by step\..*?<answer> answer here</answer>\.\s*", re.S)
def syn():
    for f in ("synlogic_easy", "synlogic_hard"):
        for r in rd(f):
            t = r["data_source"]
            if t not in SYN_OK:
                continue
            g = json.loads(r["extra_info"]["game_data_str"])
            a = g.get("answer")
            if a in (None, "", []):
                continue
            q = INSTR.sub("", g["question"])
            zh = bool(re.search(r"[一-鿿]", q))
            yield {"domain": "logic", "lang": "zh" if zh else "en", "prompt": q.strip() + "\n\nThink step by step, and put your final answer within \\boxed{}.",
                   "answer": a if isinstance(a, str) else json.dumps(a), "verifier": "exact", "meta": {"task": t, "grid": t in GRID, "level": f[9:]}}
out("synlogic", syn())
def fld():
    for r in rd("fld2"):
        q = ("Given the facts below, decide whether the hypothesis is PROVED, DISPROVED, or UNKNOWN (cannot be decided from the facts).\n\n"
             f"Facts:\n{r['context']}\n\nHypothesis: {r['hypothesis']}")
        yield {"domain": "logic", "lang": "en", "prompt": q + "\n\nThink step by step, and put your final answer (PROVED, DISPROVED or UNKNOWN) within \\boxed{}.",
               "answer": r["proof_label"], "verifier": "label", "meta": {"labels": ["PROVED", "DISPROVED", "UNKNOWN"], "depth": r.get("depth")}}
out("fld2", fld())

# ---------- code ----------
def code_io(q, tests, dom="code"):
    tests = [t for t in tests if len(t["input"]) < 4000 and len(t["output"]) < 4000][:8]
    return {"domain": dom, "lang": "en", "prompt": q.strip() + CODE_SUFFIX + " Read from standard input and write to standard output.",
            "answer": "", "verifier": "code_io", "meta": {"tests": tests}}
def oro():
    for r in rd("oro1_code"):
        v = r["verification_info"]; v = json.loads(v) if isinstance(v, str) else v
        tc = [t for t in v["test_cases"] if t["type"] == "stdin_stdout"]
        if tc and len(tc) == len(v["test_cases"]):
            q = r["problem_statement"].replace("Solve the following coding problem using the programming language python:\n\n", "")
            q = re.split(r"\n\nThe input will be (?:given via stdin|stdin)", q)[0]
            yield code_io(q, [{"input": t["input"], "output": t["output"]} for t in tc])
out("oro1_code", oro())
def taco():
    for r in rd("taco_verified"):
        io = json.loads(r["input_output"])
        ins, outs = io.get("inputs") or [], io.get("outputs") or []
        if not ins or "fn_name" in io:
            continue
        if all(isinstance(x, str) for x in ins) and all(isinstance(x, str) for x in outs):
            row = code_io(r["question"], [{"input": a, "output": b} for a, b in zip(ins, outs)])
            row["meta"]["difficulty"] = r["difficulty"]
            yield row
out("taco", taco())
def mbpp():
    for r in rd("mbpp_train"):
        yield {"domain": "code", "lang": "en", "prompt": r["text"] + "\nYour code should pass these tests:\n" + "\n".join(r["test_list"]) + CODE_SUFFIX,
               "answer": "", "verifier": "code_assert", "meta": {"tests": r["test_list"], "setup": r.get("test_setup_code") or ""}}
out("mbpp_train", mbpp())

def kod():
    for r in rd("kodcode_rl10k"):
        if "from solution import" not in r["test"] or "pytest.fixture" in r["test"] or "capsys" in r["test"] or "monkeypatch" in r["test"]:
            continue
        ti = r.get("test_info") or []
        sig = "".join(f"\n- {t.get('function_declaration', '')}" for t in ti if isinstance(t, dict) and t.get("function_declaration"))
        q = r["question"] + (f"\nYour solution should define:{sig}" if sig else "")
        yield {"domain": "code", "lang": "en", "prompt": q + CODE_SUFFIX, "answer": "", "verifier": "code_pytest",
               "meta": {"test": r["test"], "subset": r.get("subset"), "gpt_pass": r.get("gpt_pass_percentage")}}
out("kodcode", kod())
def swh():
    for r in rd("ml_lla_sel"):
        if r["lang"] != "swh_Latn":
            continue
        a = gsm_ans(r.get("source_answer"))
        if r.get("target_question") and a:
            yield M(r["target_question"], a, lang="sw")
out("gsm8k_sw", swh())

json.dump(stats, open(f"{O}/stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
