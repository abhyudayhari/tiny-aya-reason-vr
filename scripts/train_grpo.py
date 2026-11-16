"""Stage 2: RLVR with AlignTune (TRL backend), DAPO-style GRPO loss, vLLM rollouts.

Reward = 1 if the final \\boxed{} answer is math-verify equivalent to the reference, else 0.
Launch: accelerate launch --num_processes N scripts/train_grpo.py --model PATH --data FILE --out DIR
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("ALIGNTUNE_ENABLE_VLLM", "1")
from aligntune.core.backend_factory import create_rl_trainer  # noqa: E402
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig  # noqa: E402

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def math_verify_reward(text, answer=None, **kwargs):
    """1.0 if the completion is verified correct, else 0.0.

    A plain reference string is checked with math-verify. A JSON reference ({"verifier": ...}) is a multi-domain
    item (finance / logic / code / multilingual maths) and is routed to rewards.score.
    """
    ref = answer if answer is not None else kwargs.get("reference")
    if isinstance(ref, str) and ref.startswith('{"verifier"'):
        from rewards import score
        return score(json.loads(ref), text or "")
    if ref is None or "\\boxed" not in (text or ""):
        return 0.0
    try:
        g = parse(f"${ref}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
                  parsing_timeout=5)
        p = parse(text, extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                  parsing_timeout=5)
        return 1.0 if p and verify(g, p, timeout_seconds=5) else 0.0
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)      # jsonl: prompt, answer
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch", type=int, default=16)        # prompts*gens per device per step
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--max-prompt", type=int, default=768)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from (optimizer, scheduler, data position)")
    a = ap.parse_args()

    kw = dict(
        model_name=a.model,
        dataset_name=a.data,
        column_mapping={"prompt": "prompt", "answer": "answer"},
        algorithm="grpo",
        backend="trl",
        device_map="DDP",  # AlignTune: place each rank's copy on its own GPU
        output_dir=a.out,
        num_epochs=1,
        max_steps=a.steps,
        batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_generations=a.gens,
        learning_rate=a.lr,
        warmup_ratio=0.03,
        lr_scheduler="constant_with_warmup",
        weight_decay=0.0,
        max_grad_norm=1.0,
        # DAPO-style objective: token-level loss, no KL, asymmetric clip
        loss_type="dapo",
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        scale_rewards="group",
        mask_truncated_completions=True,
        max_prompt_length=a.max_prompt,
        max_completion_length=a.max_completion,
        max_seq_length=a.max_prompt + a.max_completion,
        temperature=1.0,
        top_p=1.0,
        rollout_backend="vllm",
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        vllm_enable_sleep_mode=False,
        rewards=[{"type": math_verify_reward, "weight": 1.0}],
        bf16=True,
        use_gradient_checkpointing=True,
        logging_steps=1,
        save_steps=a.save_steps,
        save_total_limit=6,
        seed=42,
        log_completions=True,
        num_completions_to_print=2,
    )
    if a.lora:
        kw.update(use_peft=True, lora_r=64, lora_alpha=128, lora_dropout=0.0,
                  lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])
    else:
        kw.update(use_peft=False)
    trainer = create_rl_trainer(**kw)
    if a.resume:  # AlignTune builds the HF trainer inside train(); hook it so trainer.train resumes from the checkpoint
        _setup = trainer.setup_trainer

        def _setup_and_resume():
            _setup()
            hf_train = trainer.trainer.train
            trainer.trainer.train = lambda *x, **k: hf_train(resume_from_checkpoint=a.resume)
        trainer.setup_trainer = _setup_and_resume
    res = trainer.train()
    print("TRAIN_RESULT", res)
    trainer.save_model(a.out + "/final") if hasattr(trainer, "save_model") else None


if __name__ == "__main__":
    main()
