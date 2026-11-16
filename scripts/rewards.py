"""Verifiable rewards for every PS6 domain. score(item, completion) -> 1.0 / 0.0.

item: {"verifier": ..., "answer": ..., "meta": {...}}
  math      math-verify on the last \\boxed{}
  fin       numeric within tolerance (meta tol_abs/tol_rel, or 1% with %/decimal equivalence); INSUFFICIENT; yes/no
  kk        every inhabitant's knight/knave status named correctly
  exact     normalised string match of the last \\boxed{} (grids compared by their digit/token sequence)
  label     one of a fixed label set (meta labels)
  code_io   python program in last ```python block, stdin/stdout test cases (meta tests)
  code_assert  python code + assert lines (meta tests, meta setup)
  code_fn   call-based tests: fn_name(*inputs) == output (meta fn_name, tests)
"""
import json, os, re, resource, signal, subprocess, sys, tempfile, threading

BOX_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE_SUFFIX = ("\n\nThink step by step about the approach first. Then write the complete solution in Python "
               "in a single ```python code block.")
FIN_SUFFIX = ("\n\nPlease reason step by step, and put your final answer within \\boxed{}. "
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


def _math(text, ans):
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    if "\\boxed" not in (text or ""):
        return False
    import threading
    to = 5 if threading.current_thread() is threading.main_thread() else None  # signal timeouts only work in main thread
    try:
        g = parse(f"${ans}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()], parsing_timeout=to)
        p = parse(text, extraction_config=[LatexExtractionConfig(boxed_match_priority=0)], parsing_timeout=to)
        return bool(p) and verify(g, p, timeout_seconds=to)
    except Exception:
        return False


def to_num(s):
    s = (s or "").translate(DEVA)
    s = re.sub(r"\\text\{[^}]*\}|\\mathrm\{[^}]*\}|Rs\.?|INR|₹|rupees?|रुपये|\\%|%|\$|\\,|,|\s|\\!", "", s, flags=re.I)
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", s.replace("\\frac{", "").replace("}{", "/"))
    if not m:
        return None
    t = m.group(0)
    if "/" in t:
        a, b = t.split("/")
        return float(a) / float(b) if float(b) else None
    return float(t)


def _fin(text, ans, meta):
    b = last_boxed(text)
    if b is None:
        return False
    a = str(ans).strip()
    if a.upper() == "INSUFFICIENT":
        return "INSUFFICIENT" in b.upper()
    if "INSUFFICIENT" in b.upper():
        return False
    if a.lower() in ("yes", "no"):
        return re.sub(r"[^a-z]", "", b.lower()).startswith(a.lower())
    try:
        v = float(a.replace(",", ""))
    except ValueError:
        return _norm(b) == _norm(a)
    p = to_num(b)
    if p is None:
        return False
    if meta.get("tol_abs") is not None or meta.get("tol_rel") is not None:
        tol = max(float(meta.get("tol_abs") or 0), float(meta.get("tol_rel") or 0) * abs(v))
        return abs(p - v) <= tol + 1e-9
    return any(abs(c - v) <= max(0.01 * abs(v), 5e-4) for c in (p, p / 100, p * 100))


def _kk(text, names, sol):
    tail = (last_boxed(text) or (text or "")[-800:]).lower()
    for nm, kn in zip(names, sol):
        m = re.search(re.escape(nm.lower()) + r"\W+(?:is\W+)?(?:a\W+)?(knight|knave)", tail)
        if not m or (m.group(1) == "knight") != bool(kn):
            return False
    return True


def _norm(s):
    s = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}|\\textbf\{([^}]*)\}", lambda m: next(g for g in m.groups() if g is not None), s or "")
    s = s.replace("\\", "").replace("$", "")
    return re.sub(r"[\s\"'`]", "", s).strip(".").lower()


def _exact(text, ans, meta):
    b = last_boxed(text)
    if b is None:
        m = re.findall(r"<answer>(.*?)</answer>", text or "", re.S)
        b = m[-1] if m else None
    if b is None:
        return False
    a = ans if isinstance(ans, str) else json.dumps(ans)
    if meta.get("grid"):
        toks = lambda s: re.findall(r"[A-Za-z0-9.\-]+|[^\s\[\]\(\),;'\"]", s)
        return toks(b) == toks(a)
    if _norm(b) == _norm(a):
        return True
    x, y = to_num(b), to_num(a)
    return x is not None and y is not None and re.fullmatch(r"-?[\d.,]+", a.strip()) is not None and abs(x - y) < 1e-6


def _label(text, ans, meta):
    b = last_boxed(text)
    if b is None:
        return False
    b = _norm(b)
    hits = [l for l in meta["labels"] if _norm(l) in b]
    return len(hits) == 1 and _norm(hits[0]) == _norm(ans) or b == _norm(ans)


def code_block(text):
    bl = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text or "", re.S)
    return bl[-1] if bl else None


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    os.setsid()


def _exec(cmd, cwd, stdin=b"", timeout=6):
    """Run cmd in its own session; on timeout kill the whole process group (forked children can hold the pipes)."""
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         preexec_fn=_limits)
    try:
        out, _ = p.communicate(stdin, timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            p.communicate(timeout=2)
        except Exception:
            pass
        return -9, b""


def _run(src, stdin="", timeout=6):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py")
        open(p, "w").write(src)
        rc, out = _exec([sys.executable, "-I", p], td, stdin.encode(), timeout)
        return rc == 0, out.decode(errors="replace")


def _same_out(a, b):
    ta, tb = a.split(), b.split()
    if ta == tb:
        return True
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) > 1e-6 * max(1, abs(float(y))):
                return False
        except ValueError:
            return False
    return True


def _code(text, verifier, meta):
    c = code_block(text)
    if not c:
        return False
    tests = (meta.get("tests") or [])[:8]
    if verifier == "code_io":
        for t in tests:
            ok, out = _run(c, t["input"])
            if not ok or not _same_out(out, t["output"]):
                return False
        return True
    if verifier == "code_assert":
        ok, _ = _run((meta.get("setup") or "") + "\n" + c + "\n\n" + "\n".join(tests) + "\n", timeout=10)
        return ok
    if verifier == "code_pytest":  # KodCode style: tests import from `solution`
        runner = ("import sys, types, inspect\nsys.path.insert(0, '.')\nimport test_sol as T\n"
                  "fs = [f for n, f in vars(T).items() if n.startswith('test') and callable(f)]\n"
                  "assert fs\nfor f in fs:\n    if not inspect.signature(f).parameters: f()\n")
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "solution.py"), "w").write(c)
            open(os.path.join(td, "test_sol.py"), "w").write(meta["test"])
            open(os.path.join(td, "run.py"), "w").write(runner)
            rc, _ = _exec([sys.executable, "-I", "run.py"], td, b"", 10)
            return rc == 0
    if verifier == "code_fn":
        fn = meta["fn_name"]
        harness = c + "\n\nimport json\n_T = json.loads(" + repr(json.dumps(tests)) + ")\n"
        harness += ("_f = Solution()." + fn if "class Solution" in c else "_f = " + fn) + "\n"
        harness += ("for _i, _o in _T:\n    _r = _f(*_i)\n    if isinstance(_r, tuple): _r = list(_r)\n"
                    "    if _r != _o and [_r] != _o and _r != (_o[0] if isinstance(_o, list) and len(_o) == 1 else object()):\n"
                    "        raise SystemExit(1)\n")
        ok, _ = _run(harness, timeout=10)
        return ok
    return False


class _Hard(Exception):
    pass


def _code_guarded(text, v, meta, limit=90):
    """Backstop: a code item can never block the caller for more than `limit` s (main thread only)."""
    if threading.current_thread() is not threading.main_thread():
        return _code(text, v, meta)
    def h(*_):
        raise _Hard()
    old = signal.signal(signal.SIGALRM, h)
    signal.alarm(limit)
    try:
        return _code(text, v, meta)
    except _Hard:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def score(item, text):
    v, a, meta = item["verifier"], item.get("answer"), item.get("meta") or {}
    try:
        if v == "math":
            return float(_math(text, a))
        if v == "fin":
            return float(_fin(text, a, meta))
        if v == "kk":
            return float(_kk(text, meta["names"], meta["solution"]))
        if v == "exact":
            return float(_exact(text, a, meta))
        if v == "label":
            return float(_label(text, a, meta))
        if v.startswith("code"):
            return float(_code_guarded(text, v, meta))
    except Exception:
        return 0.0
    return 0.0
