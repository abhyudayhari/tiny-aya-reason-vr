"""Merge a GRPO LoRA adapter into its base model so vLLM can serve it.
usage: merge_lora.py --base BASE_MODEL --adapter OUTPUT_DIR[/checkpoint-N] --out MERGED_DIR
Picks <adapter>/final, else <adapter> itself, else its latest checkpoint-* that has an adapter.
"""
import argparse, glob, os, shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def has_adapter(d):
    return os.path.exists(os.path.join(d, "adapter_config.json"))


cands = [os.path.join(a.adapter, "final"), a.adapter] + sorted(
    glob.glob(os.path.join(a.adapter, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]), reverse=True)
src = next((c for c in cands if has_adapter(c)), None)
assert src, f"no adapter_config.json under {a.adapter}"
print("adapter", src)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, src).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
gc = os.path.join(a.base, "generation_config.json")
if os.path.exists(gc):
    shutil.copy(gc, os.path.join(a.out, "generation_config.json"))
print("merged ->", a.out)
