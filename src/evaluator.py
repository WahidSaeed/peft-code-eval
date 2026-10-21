"""Inference for one model (M0 base or an M1–M9 adapter) on one benchmark.

    # M0: untuned base model (16-bit)
    python src/evaluator.py --bits 16 --run_name M0 --benchmark megadiff --score
    # M1: 4-bit base + its trained adapter
    python src/evaluator.py --bits 4 --adapter results/runs/M1/adapter --run_name M1 --benchmark humaneval_java

Decoding (RESEARCH_LOG #37/#39):
  megadiff        greedy, 1 sample: Exact Match / CodeBLEU / syntax on the held-out split
  humanevalfix,   sampled, n = 20, temperature 0.2, top-p 0.95 (OctoPack protocol):
  humaneval_java  pass@1 and pass@10 from the same samples, scored by execution.py

Two measurement phases, both on the same loaded model:
  1. Batched generation over the whole test set → predictions JSONL, generation
     throughput, peak inference VRAM and energy (RQ2/RQ3).
  2. Latency probe at batch size 1, always GREEDY, on the first --latency_samples
     prompts → Time-to-First-Token and decode tokens/s, as a single developer would see (RQ3).

The adapter is applied unmerged on top of the quantized base (how QLoRA models are
served). Each output row carries a `predictions` list (n entries) plus every field
metrics_se.py / execution.py need. Megadiff predictions can be scored inline (--score).
Test-based predictions run model-generated code, so score them separately with
execution.py on a disposable machine.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import DEFAULT_MODEL, HEF_LANGUAGES, chat_prompt  # noqa: E402
from profiler import GPUProfiler, append_csv, cap_vram, measure_generation  # noqa: E402
from trainer import quantization_config  # noqa: E402

# Longest reference fix + margin (DeepSeek tokenizer, RESEARCH_LOG #39):
# Megadiff test max 375 tokens, HumanEval-Java max 383, HumanEvalFix-Java max 506 (whole class)
MAX_NEW_TOKENS = {"megadiff": 512, "humaneval_java": 512, "humanevalfix": 768}
N_SAMPLES = {"megadiff": 1, "humaneval_java": 20, "humanevalfix": 20}


def load_records(benchmark, data_dir, languages, limit=None):
    if benchmark == "megadiff":
        paths = [Path(data_dir) / "megadiff_test.jsonl"]
    elif benchmark == "humaneval_java":
        paths = [Path(data_dir) / "humaneval_java.jsonl"]
    else:
        paths = [Path(data_dir) / f"humanevalfix_{lang}.jsonl" for lang in languages]
    records = []
    for path in paths:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        records += rows[:limit] if limit else rows
    return records


def load_model(model_name, bits, adapter=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(bits),
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    return model


def _gen_kwargs(tokenizer, max_new_tokens, n_samples=1, temperature=0.2, top_p=0.95):
    """Greedy for n_samples == 1, otherwise n_samples sampled sequences per prompt."""
    kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if n_samples > 1:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n_samples)
    return kwargs


def _n_generated(seq, eos_id, pad_id):
    """Generated tokens up to and including the first EOS (ignores trailing padding)."""
    n = 0
    for t in seq:
        n += 1
        if t == eos_id:
            break
    else:
        n = sum(1 for t in seq if t != pad_id)
    return n


@torch.no_grad()
def generate_all(model, tokenizer, prompts, batch_size, max_new_tokens, n_samples=1, temperature=0.2, top_p=0.95,
                 prof=None):
    """Batched generation; prompts are length-sorted to cut padding, output keeps input order.

    batch_size counts sequences: with n_samples > 1, batch_size // n_samples prompts (at least 1)
    go into each generate() call. Returns per prompt a list of n_samples outputs and token counts.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs, n_tokens = [None] * len(prompts), [None] * len(prompts)
    kwargs = _gen_kwargs(tokenizer, max_new_tokens, n_samples, temperature, top_p)
    per_call = max(1, batch_size // n_samples)
    device = model.device
    for start in range(0, len(order), per_call):
        idx = order[start:start + per_call]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1]:]  # rows: prompt 0 × n_samples, prompt 1 × n_samples, ...
        for k, i in enumerate(idx):
            rows = new[k * n_samples:(k + 1) * n_samples]
            outputs[i] = [tokenizer.decode(r, skip_special_tokens=True) for r in rows]
            n_tokens[i] = [_n_generated(r.tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id) for r in rows]
        if prof is not None:
            prof.add(samples=len(idx) * n_samples, tokens=sum(sum(n_tokens[i]) for i in idx))
        print(f"  generated {min(start + per_call, len(order))}/{len(order)} prompts", flush=True)
    return outputs, n_tokens


@torch.no_grad()
def latency_probe(model, tokenizer, prompts, max_new_tokens):
    """Batch-size-1 TTFT / decode speed, plus peak VRAM for a single request."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    kwargs = _gen_kwargs(tokenizer, max_new_tokens)
    rows = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(model.device)
        _, timing = measure_generation(model, enc["input_ids"], enc["attention_mask"], **kwargs)
        rows.append(timing)
    ttft = sorted(r["ttft_s"] for r in rows)
    decode = [r["decode_tokens_per_s"] for r in rows if r["decode_tokens_per_s"]]
    return {
        "latency_n": len(rows),
        "ttft_median_s": round(statistics.median(ttft), 4) if ttft else None,
        "ttft_p90_s": round(ttft[math.ceil(0.9 * len(ttft)) - 1], 4) if ttft else None,  # nearest-rank
        "decode_tokens_per_s_median": round(statistics.median(decode), 2) if decode else None,
        "latency_peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if torch.cuda.is_available() else None,
    }


def _warmup(model, tokenizer, prompt):
    """One short generation so CUDA kernels / caches don't distort the first measurement."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    model.generate(**enc, **_gen_kwargs(tokenizer, 8))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", type=int, choices=[4, 8, 16], required=True, help="precision of the base model")
    p.add_argument("--adapter", help="trained LoRA adapter dir (omit for M0)")
    p.add_argument("--run_name", required=True, help="M0 … M9")
    p.add_argument("--benchmark", choices=list(MAX_NEW_TOKENS), required=True)
    p.add_argument("--languages", nargs="+", default=["java"], choices=HEF_LANGUAGES, help="HumanEvalFix languages")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/predictions")
    p.add_argument("--log_csv", default="results/eval_log.csv")
    p.add_argument("--batch_size", type=int, default=20, help="sequences per generate() call")
    p.add_argument("--max_new_tokens", type=int, help=f"default per benchmark: {MAX_NEW_TOKENS}")
    p.add_argument("--n_samples", type=int, help=f"samples per prompt; 1 = greedy. Default per benchmark: {N_SAMPLES}")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--latency_samples", type=int, default=50, help="batch-size-1 latency probe size (0 = skip)")
    p.add_argument("--limit", type=int, help="first N records per file (debugging)")
    p.add_argument("--score", action="store_true", help="score Megadiff inline (EM, CodeBLEU, syntax; first sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_vram_gib", type=float,
                   help="cap PyTorch GPU memory (e.g. 24 on a 48 GB card, to emulate a 24 GB GPU)")
    args = p.parse_args(argv)
    args.max_new_tokens = args.max_new_tokens or MAX_NEW_TOKENS[args.benchmark]
    args.n_samples = args.n_samples or N_SAMPLES[args.benchmark]
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    vram_fraction = cap_vram(args.max_vram_gib)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only batched generation

    records = load_records(args.benchmark, args.data_dir, args.languages, args.limit)
    prompts = [chat_prompt(tokenizer, r["prompt"]) for r in records]
    print(f"{args.run_name}: {len(records)} prompts from {args.benchmark}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = load_model(args.model, args.bits, args.adapter)
    load_s = time.perf_counter() - t0
    static_vram = round(torch.cuda.memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None
    _warmup(model, tokenizer, prompts[0])

    set_seed(args.seed)  # sampling starts from the same RNG state for every model
    with GPUProfiler() as prof:
        predictions, n_tokens = generate_all(model, tokenizer, prompts, args.batch_size, args.max_new_tokens,
                                             args.n_samples, args.temperature, args.top_p, prof)
    latency = latency_probe(model, tokenizer, prompts[:args.latency_samples], args.max_new_tokens) \
        if args.latency_samples > 0 else {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "humanevalfix_" + "-".join(args.languages) if args.benchmark == "humanevalfix" else args.benchmark
    pred_path = out_dir / f"{args.run_name}_{suffix}.jsonl"
    with pred_path.open("w") as f:
        for r, preds, n in zip(records, predictions, n_tokens):
            f.write(json.dumps({**r, "predictions": preds, "generated_tokens": n, "run_name": args.run_name}) + "\n")

    p = prof.results
    hit_limit = sum(n >= args.max_new_tokens for ns in n_tokens for n in ns)
    row = {
        "run_name": args.run_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": suffix,
        "n": len(records),
        "bits": args.bits,
        "adapter": args.adapter or "",
        "batch_size": args.batch_size,
        "n_samples": args.n_samples,
        "temperature": args.temperature if args.n_samples > 1 else None,
        "top_p": args.top_p if args.n_samples > 1 else None,
        "max_new_tokens": args.max_new_tokens,
        "hit_max_new_tokens": hit_limit,
        "model_load_s": round(load_s, 1),
        "static_vram_gib": static_vram,
        "gen_duration_s": p.get("duration_s"),
        "gen_tokens": p.get("tokens"),
        "gen_tokens_per_s": p.get("tokens_per_s"),
        "gen_samples_per_s": p.get("samples_per_s"),
        "gen_peak_vram_allocated_gib": p.get("peak_vram_allocated_gib"),
        "gen_peak_vram_reserved_gib": p.get("peak_vram_reserved_gib"),
        "gen_energy_wh": p.get("energy_wh"),
        **latency,
        "gpu_name": p.get("gpu_name"),
        "vram_cap_gib": args.max_vram_gib if vram_fraction else None,
        "predictions": str(pred_path),
    }
    if args.score and args.benchmark == "megadiff":
        from metrics_se import evaluate, extract_code
        summary, _ = evaluate([extract_code(x[0]) for x in predictions], [r["target"] for r in records])
        row.update({k: v for k, v in summary.items() if k != "n"})

    append_csv(args.log_csv, row)
    print(json.dumps(row, indent=2))
    return row


if __name__ == "__main__":
    main()
