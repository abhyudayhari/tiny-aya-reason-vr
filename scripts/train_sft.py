"""Stage 1: full-parameter SFT with AlignTune (TRL backend) on the CuratorKIT-curated set.

Data is conversational prompt/completion, so TRL applies the Aya chat template and
computes loss on the assistant solution only.
Launch: accelerate launch --num_processes N scripts/train_sft.py --data FILE --out DIR
"""
import argparse, json, os

from aligntune.core.backend_factory import create_sft_trainer


def to_prompt_completion(src, dst, limit=0, model=None):
    """Render the Aya chat template into plain prompt/completion strings.

    AlignTune's loader flattens chat-format columns to raw text, which would drop the
    template. Pre-rendering keeps train and inference formats identical; TRL then masks
    the prompt (completion_only_loss). BOS is stripped because the tokenizer adds it.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    n = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            r = json.loads(line)
            msgs = r["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            p = [{"role": "user", "content": user}]
            ps = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            fs = tok.apply_chat_template(p + [{"role": "assistant", "content": asst}], tokenize=False)
            assert fs.startswith(ps) and ps.startswith(tok.bos_token)
            comp = fs[len(ps):]
            # Aya natively ends a reply with <|END_RESPONSE|><EOS_TOKEN> (id 3), not the
            # template's <|END_OF_TURN_TOKEN|>; train on what the model emits at inference.
            assert comp.endswith("<|END_RESPONSE|><|END_OF_TURN_TOKEN|>") and tok.eos_token == "<EOS_TOKEN>"
            comp = comp[: -len("<|END_OF_TURN_TOKEN|>")] + tok.eos_token
            g.write(json.dumps({"prompt": ps[len(tok.bos_token):], "completion": comp},
                               ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tiny-aya-global-eos3")
    ap.add_argument("--data", required=True)          # CuratorKIT sft_messages.jsonl
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)   # per device
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pc = os.path.join(a.out, "train_pc.jsonl")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and int(os.environ.get("RANK", "0")) == 0:
        print("rows", to_prompt_completion(a.data, pc + ".tmp", a.limit, a.model))
        os.replace(pc + ".tmp", pc)
    else:
        import time
        while not os.path.exists(pc):
            time.sleep(5)

    trainer = create_sft_trainer(
        model_name=a.model,
        dataset_name=pc,
        backend="trl",
        device_map=None,  # let DDP place the model
        task_type="instruction_following",
        output_dir=a.out,
        num_epochs=a.epochs,
        max_steps=a.max_steps,
        batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_seq_length=2048,
        completion_only_loss=True,
        packing=False,
        padding_free=False,
        use_peft=False,
        peft_enabled=False,
        bf16=True,
        precision="bf16",
        use_gradient_checkpointing=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )
    res = trainer.train()
    print("TRAIN_RESULT", json.dumps({k: v for k, v in res.items() if k != "metrics"}, default=str))


if __name__ == "__main__":
    main()
