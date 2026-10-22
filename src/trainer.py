"""PEFT training for one cell of the experimental matrix (M1–M9).

    python src/trainer.py --bits 4 --rank 8 --output_dir results/runs/M1
    python src/trainer.py --bits 4 --rank 8 --output_dir results/runs/dry --max_steps 50   # dry run

--bits 4   4-bit NF4 base + double quantization, bf16 compute (QLoRA)
--bits 8   8-bit LLM.int8() base (bitsandbytes)
--bits 16  bf16 base (plain LoRA)

Everything except bits / rank / alpha is fixed by the research plan and identical
across runs, so differences between M1–M9 come only from the matrix factors.
Training is wrapped in the GPU profiler (peak VRAM, duration, throughput, energy),
and one row per run is appended to a CSV log.
"""

import argparse
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import DEFAULT_MODEL, IGNORE_INDEX, tokenize_for_training  # noqa: E402
from profiler import GPUProfiler, append_csv, cap_vram, trainable_parameters  # noqa: E402

# "All linear" layers of a LLaMA-style decoder (DeepSeek-Coder): attention + MLP
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Fixed hyperparameters (research plan §5); recorded in every run's config
FIXED = {
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # effective batch 16
    "num_train_epochs": 3,
    "max_length": 768,  # Megadiff pairs above 768 tokens are dropped when the splits are built (#39)
    "lora_dropout": 0.05,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "optim": "paged_adamw_32bit",  # same optimizer for every bit-width; paging only engages under memory pressure
    "weight_decay": 0.0,
    "max_grad_norm": 0.3,
}


def quantization_config(bits):
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(model_name, bits, gradient_checkpointing=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(bits),
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if bits in (4, 8):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def lora_config(rank, alpha, dropout):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_split(data_dir, subset, split, tokenizer, max_length, limit=None):
    ds = load_dataset("json", data_files=str(Path(data_dir) / f"{subset}_{split}.jsonl"), split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    tok = tokenize_for_training(ds, tokenizer, max_length)
    n_truncated = sum(tok["truncated"])
    n_tokens = sum(len(x) for x in tok["input_ids"])
    return tok.remove_columns(["truncated"]), n_truncated, n_tokens


def _warmup_kwargs(ratio):
    """transformers < 5 takes `warmup_ratio`; >= 5 removed it and accepts a float ratio in `warmup_steps`."""
    fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    return {"warmup_ratio": ratio} if "warmup_ratio" in fields else {"warmup_steps": ratio}


def _versions():
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    out["cuda"] = torch.version.cuda
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", type=int, choices=[4, 8, 16], required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=int, help="LoRA alpha (default: 2 × rank)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", help="e.g. M1 (default: output_dir name)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--subset", default="megadiff", help="data file prefix: <subset>_{train,validation}.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_vram_gib", type=float,
                   help="cap PyTorch GPU memory (e.g. 24 on a 48 GB card, to emulate a 24 GB GPU)")
    p.add_argument("--max_steps", type=int, default=-1, help="optimizer steps; > 0 overrides epochs (dry runs)")
    p.add_argument("--train_limit", type=int, help="use only the first N training examples (debugging)")
    p.add_argument("--eval_limit", type=int, help="use only the first N validation examples (debugging)")
    p.add_argument("--no_eval", action="store_true", help="skip per-epoch validation loss")
    p.add_argument("--log_csv", default="results/train_log.csv")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--save_steps", type=int, default=200,
                   help="checkpoint every N optimizer steps (full runs only), so an interrupted run resumes")
    p.add_argument("--no_resume", action="store_true", help="ignore existing checkpoints in output_dir")
    for k, v in FIXED.items():  # overridable for debugging; overrides are recorded in the run config
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args(argv)
    args.alpha = args.alpha or 2 * args.rank
    args.run_name = args.run_name or Path(args.output_dir).name
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    vram_fraction = cap_vram(args.max_vram_gib)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, n_truncated, n_train_tokens = load_split(
        args.data_dir, args.subset, "train", tokenizer, args.max_length, args.train_limit)
    eval_ds = None
    if not args.no_eval:
        eval_ds, _, _ = load_split(args.data_dir, args.subset, "validation", tokenizer, args.max_length, args.eval_limit)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model = load_model(args.model, args.bits, not args.no_gradient_checkpointing)
    model = get_peft_model(model, lora_config(args.rank, args.alpha, args.lora_dropout))
    load_s = time.perf_counter() - t_load
    params = trainable_parameters(model)
    static_vram_gib = round(torch.cuda.memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None

    training_args = TrainingArguments(
        output_dir=str(out),
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        **_warmup_kwargs(args.warmup_ratio),
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # already configured on the model in load_model()
        eval_strategy="no" if args.no_eval else ("steps" if args.max_steps > 0 else "epoch"),
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_strategy="steps" if args.max_steps <= 0 else "no",
        save_steps=args.save_steps,
        save_total_limit=1,  # only the latest checkpoint (adapter + optimizer + scheduler + RNG state)
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX, padding=True),
        processing_class=tokenizer,
    )

    # Resume after an interruption (lost pod, preemption): the latest checkpoint restores adapter,
    # optimizer, scheduler, RNG and data position. Timing and energy then cover only the resumed part,
    # so the run is flagged and its efficiency numbers are not used as official RQ3 values (#63).
    # Only complete checkpoints: trainer_state.json is written last, so a save cut off mid-way is skipped.
    checkpoints = sorted((c for c in out.glob("checkpoint-*") if (c / "trainer_state.json").exists()),
                         key=lambda c: int(c.name.split("-")[-1]))
    resume = str(checkpoints[-1]) if checkpoints and not args.no_resume else None
    if resume:
        print(f"resuming from {resume}")
    with GPUProfiler() as prof:
        train_output = trainer.train(resume_from_checkpoint=resume)
        steps = trainer.state.global_step
        eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        samples = min(steps * eff_batch, len(train_ds) * args.num_train_epochs) if args.max_steps > 0 \
            else len(train_ds) * args.num_train_epochs
        prof.add(samples=samples, tokens=round(samples * n_train_tokens / len(train_ds)))

    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))

    history = trainer.state.log_history
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    eval_runtime_s = sum(h.get("eval_runtime", 0.0) for h in history)
    p = prof.results
    row = {
        "run_name": args.run_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "bits": args.bits,
        "rank": args.rank,
        "alpha": args.alpha,
        "seed": args.seed,
        "epochs": args.num_train_epochs if args.max_steps <= 0 else None,
        "max_steps": args.max_steps,
        "global_steps": trainer.state.global_step,
        "n_train": len(train_ds),
        "n_train_truncated": n_truncated,
        **params,
        "model_load_s": round(load_s, 1),
        "static_vram_gib": static_vram_gib,
        "train_duration_s": p.get("duration_s"),
        "eval_runtime_s": round(eval_runtime_s, 1),
        "train_duration_excl_eval_s": round(p["duration_s"] - eval_runtime_s, 1) if p.get("duration_s") else None,
        "peak_vram_allocated_gib": p.get("peak_vram_allocated_gib"),
        "peak_vram_reserved_gib": p.get("peak_vram_reserved_gib"),
        "samples_per_s": p.get("samples_per_s"),
        "tokens_per_s": p.get("tokens_per_s"),
        "energy_wh": p.get("energy_wh"),
        "energy_method": p.get("energy_method"),
        "final_train_loss": round(train_output.training_loss, 4),
        "eval_loss_per_eval": json.dumps([round(x, 4) for x in eval_losses]),
        "gpu_name": p.get("gpu_name"),
        "vram_cap_gib": args.max_vram_gib if vram_fraction else None,
        "resumed_from": Path(resume).name if resume else "",
    }
    config = {"args": vars(args), "fixed_defaults": FIXED, "target_modules": TARGET_MODULES,
              "versions": _versions(), "result": row}
    (out / "run_config.json").write_text(json.dumps(config, indent=2, default=str))
    append_csv(args.log_csv, row)
    print(json.dumps(row, indent=2))
    return row


if __name__ == "__main__":
    main()
