"""Turn data/rl_pool.jsonl (GSM8K train + MATH train, from fetch_data.py rl) into chat-format RL rows: data/rl_all.jsonl.
filter_rl.py then keeps the prompts the base model solves sometimes but not always -> data/rl_train_base.jsonl.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rewards import BOX_SUFFIX

n = 0
with open("data/rl_all.jsonl", "w") as f:
    for l in open("data/rl_pool.jsonl"):
        r = json.loads(l)
        f.write(json.dumps({"id": r["id"], "source": r["source"], "level": r.get("level"),
                            "prompt": [{"role": "user", "content": r["question"] + BOX_SUFFIX}],
                            "answer": r["answer"]}, ensure_ascii=False) + "\n")
        n += 1
print("rl_all rows", n)
