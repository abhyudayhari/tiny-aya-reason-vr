import json, collections
from datasets import load_dataset
ds = load_dataset("llama-lang-adapt/Translated-GSM8K", split="train", streaming=True)
c = collections.Counter(); keep = []
WANT = {"hin_Deva","ben_Beng","tel_Telu","tam_Taml","mar_Deva","guj_Gujr","kan_Knda","mal_Mlym","pan_Guru","ory_Orya","urd_Arab",
        "deu_Latn","spa_Latn","fra_Latn","jpn_Jpan","rus_Cyrl","swh_Latn","tha_Thai","zho_Hans","arb_Arab","ind_Latn","vie_Latn"}
for r in ds:
    c[r["lang"]] += 1
    if r["lang"] in WANT: keep.append(r)
print(sorted(c.items(), key=lambda x: -x[1])[:80])
with open("data/cands/ml_lla_sel.jsonl", "w") as f:
    for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("kept", len(keep), collections.Counter(r["lang"] for r in keep))
