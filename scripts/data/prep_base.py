"""Download CohereLabs/tiny-aya-global and make the training base: same weights, generation stops on
<|END_OF_TURN_TOKEN|> (id 3) and <EOS_TOKEN> (id 6), tokenizer eos = <EOS_TOKEN>.
Writes models/tiny-aya-global (as downloaded) and models/tiny-aya-global-eos3 (symlinked weights + edited configs).
The model is gated: accept its licence on the Hub and set HF_TOKEN first.
"""
import json, os
from huggingface_hub import snapshot_download

SRC, DST = "models/tiny-aya-global", "models/tiny-aya-global-eos3"
snapshot_download("CohereLabs/tiny-aya-global", local_dir=SRC)
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f.endswith(".safetensors") or f in ("model.safetensors.index.json", "tokenizer.json", "config.json", "special_tokens_map.json"):
        dst = os.path.join(DST, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(SRC, f)), dst)
gc = json.load(open(f"{SRC}/generation_config.json"))
gc.pop("_from_model_config", None); gc.pop("transformers_version", None)
gc["eos_token_id"] = [3, 6]; gc["pad_token_id"] = 0
json.dump(gc, open(f"{DST}/generation_config.json", "w"), indent=2)
tc = json.load(open(f"{SRC}/tokenizer_config.json"))
tc["eos_token"] = "<EOS_TOKEN>"
json.dump(tc, open(f"{DST}/tokenizer_config.json", "w"), indent=2, ensure_ascii=False)
print("base ready:", DST)
