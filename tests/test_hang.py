import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from rewards import score
fork = "```python\nimport os, time\nif os.fork() == 0:\n    time.sleep(300)\nprint(1)\n```"
t = time.time(); print("fork-holding-pipe", score({"verifier": "code_io", "meta": {"tests": [{"input": "", "output": "1"}]}}, fork), round(time.time() - t, 1), "s")
loop = "```python\nwhile True: pass\n```"
t = time.time(); print("infinite loop", score({"verifier": "code_io", "meta": {"tests": [{"input": "", "output": "1"}]}}, loop), round(time.time() - t, 1), "s")
ok = "```python\nprint(int(input())*2)\n```"
print("correct code", score({"verifier": "code_io", "meta": {"tests": [{"input": "3", "output": "6"}]}}, ok))
print("pytest", score({"verifier": "code_pytest", "meta": {"test": "from solution import f\ndef test_a():\n    assert f(2)==4\n"}}, "```python\ndef f(x): return x*2\n```"))
print("math", score({"verifier": "math", "answer": "\\frac{1}{2}"}, "so \\boxed{0.5}"))
sys.path.insert(0, "scripts")
