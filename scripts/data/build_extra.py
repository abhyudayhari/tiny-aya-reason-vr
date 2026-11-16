"""Turn the downloaded extra benchmarks into data/eval_<name>.jsonl (id, question, answer[, level/lang])."""
import json, re

R = "data/raw_extra"
D = "data"


def rows(name):
    return [json.loads(l) for l in open(f"{R}/{name}.jsonl")]


def num(s):
    s = str(s).strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return s


def dump(name, items):
    with open(f"{D}/eval_{name}.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(name, len(items))


dump("svamp", [{"id": r["ID"], "question": (r["Body"].strip() + " " + r["Question"].strip()).strip(),
                "answer": num(r["Answer"])} for r in rows("svamp")])

asd = []
for i, r in enumerate(rows("asdiv")):
    a = re.sub(r"\s*\(.*?\)\s*$", "", str(r["answer"])).strip()
    if not re.fullmatch(r"-?[\d.,/]+", a):  # skip ratio/text answers like "3:2" or "red"
        continue
    asd.append({"id": f"asdiv-{i}", "question": (r["body"].strip() + " " + r["question"].strip()).strip(), "answer": a.replace(",", "")})
dump("asdiv", asd)

dump("gsm_hard", [{"id": f"gsmhard-{i}", "question": r["input"], "answer": num(r["target"])} for i, r in enumerate(rows("gsm_hard"))])

gp = []
for i, r in enumerate(rows("gsm_plus")):
    a = str(r["answer"]).strip()
    if a.lower() in ("none", "", "nan"):  # "critical thinking" items have no numeric answer
        continue
    gp.append({"id": f"gsmplus-{i}", "question": r["question"], "answer": num(a), "level": r["perturbation_type"]})
dump("gsm_plus", gp)

dump("minerva", [{"id": f"minerva-{i}", "question": r["question"], "answer": r["answer"]} for i, r in enumerate(rows("minerva"))])

ol = []
for r in rows("olympiad"):
    if r.get("image_1") or (r.get("is_multiple_answer") is True):
        continue
    fa = r["final_answer"][0] if isinstance(r["final_answer"], list) else r["final_answer"]
    ol.append({"id": f"olympiad-{r['id']}", "question": r["question"], "answer": str(fa).strip("$ ")})
dump("olympiad", ol)

dump("amc23", [{"id": f"amc23-{r['id']}", "question": r["question"], "answer": num(r["answer"])} for r in rows("amc23")])
dump("aime24", [{"id": f"aime24-{r['id']}", "question": r["problem"], "answer": num(r["answer"])} for r in rows("aime24")])
dump("aime25", [{"id": f"aime25-{r['id']}", "question": r["problem"], "answer": num(r["answer"])} for r in rows("aime25")])
