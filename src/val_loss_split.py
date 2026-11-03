"""Validation loss split into changed-line and copied-line tokens, per saved epoch adapter (RESEARCH_LOG #95).

Diagnostic for the overconfidence reading of the rising validation loss (#84, #94): ~85% of the target tokens are
copied unchanged from the buggy input (#86), so the trainer's validation loss is dominated by copy tokens. This
script recomputes the loss on the same validation pairs, with the same prompt format and labels as training, and
reports it separately for
  changed  target tokens on lines that differ from the buggy method (line diff, difflib, as in #86),
  copied   target tokens on lines identical to a buggy-method line,
  eos      the end-of-sequence token,
plus the overall token-weighted loss and the trainer-style loss (mean of per-batch means, batch size 4), which
should reproduce the trainer's eval_loss up to numerical noise.

A token belongs to the target line its first character is on (a newline token belongs to the line it ends).
Runs after training, outside the timed section, so the RQ3 measurements are unaffected.

    python src/val_loss_split.py --bits 4 --run_dir results/runs/M2          # every adapter-epoch<N>
    python src/val_loss_split.py --bits 4 --adapter results/pilot/runs/M2_pilotA10/adapter --out x.json
"""

import argparse
import bisect
import difflib
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import DEFAULT_MODEL, IGNORE_INDEX, chat_prompt  # noqa: E402
from evaluator import load_model  # noqa: E402
from profiler import cap_vram  # noqa: E402

CATEGORIES = ("changed", "copied", "eos")


def changed_target_lines(buggy, target):
    """Indices of target lines (split on '\\n') that are not an unchanged copy of a buggy line."""
    b, f = buggy.split("\n"), target.split("\n")
    changed = set()
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, b, f, autojunk=False).get_opcodes():
        if op != "equal":
            changed.update(range(j1, j2))
    return changed


def encode(tokenizer, row, max_length):
    """Same input_ids/labels as data_loader.tokenize_for_training, plus a category per target token."""
    prompt_ids = tokenizer(chat_prompt(tokenizer, row["prompt"]), add_special_tokens=False)["input_ids"]
    enc = tokenizer(row["target"], add_special_tokens=False, return_offsets_mapping=True)
    target_ids = enc["input_ids"] + [tokenizer.eos_token_id]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        raise SystemExit(f"pair longer than max_length {max_length}; the splits are built without truncation (#39)")
    line_starts = [0] + [m.end() for m in re.finditer("\n", row["target"])]
    changed = changed_target_lines(row["buggy"], row["target"])
    cats = []
    for start, _ in enc["offset_mapping"]:
        line = bisect.bisect_right(line_starts, start) - 1
        cats.append("changed" if line in changed else "copied")
    cats.append("eos")
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    return input_ids, labels, [None] * len(prompt_ids) + cats


@torch.no_grad()
def split_loss(model, tokenizer, rows, max_length, batch_size):
    sums = {c: 0.0 for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}
    batch_means = []
    device = model.device
    for start in range(0, len(rows), batch_size):
        encoded = [encode(tokenizer, r, max_length) for r in rows[start:start + batch_size]]
        width = max(len(ids) for ids, _, _ in encoded)
        pad = tokenizer.pad_token_id
        ids = torch.tensor([x + [pad] * (width - len(x)) for x, _, _ in encoded], device=device)
        att = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _, _ in encoded], device=device)
        lab = torch.tensor([y + [IGNORE_INDEX] * (width - len(y)) for _, y, _ in encoded], device=device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        tok_loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), lab[:, 1:], ignore_index=IGNORE_INDEX,
                                   reduction="none").cpu()          # loss for predicting position t+1
        valid = lab[:, 1:].cpu() != IGNORE_INDEX
        batch_means.append(tok_loss[valid].mean().item())
        for k, (_, _, cats) in enumerate(encoded):
            for t, cat in enumerate(cats[1:]):                      # category of the predicted token
                if cat is not None:
                    sums[cat] += tok_loss[k, t].item()
                    counts[cat] += 1
    total_n = sum(counts.values())
    return {
        "loss_all_token_weighted": round(sum(sums.values()) / total_n, 5),
        "loss_trainer_style": round(sum(batch_means) / len(batch_means), 5),
        **{f"loss_{c}": round(sums[c] / counts[c], 5) if counts[c] else None for c in CATEGORIES},
        **{f"n_tokens_{c}": counts[c] for c in CATEGORIES},
        "share_changed_tokens": round(counts["changed"] / total_n, 4),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", type=int, choices=[4, 8, 16], required=True)
    p.add_argument("--run_dir", help="training output dir: evaluates every adapter-epoch<N>/ in it")
    p.add_argument("--adapter", nargs="+", help="explicit adapter dir(s) instead of --run_dir")
    p.add_argument("--out", help="output JSON (default <run_dir>/val_loss_split.json)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default="data/megadiff_validation.jsonl")
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=4, help="trainer's per-device eval batch size")
    p.add_argument("--limit", type=int, help="first N pairs (debugging)")
    p.add_argument("--max_vram_gib", type=float, help="cap PyTorch GPU memory (same value as for training)")
    args = p.parse_args(argv)
    cap_vram(args.max_vram_gib)
    if not (args.run_dir or args.adapter):
        p.error("give --run_dir or --adapter")

    if args.adapter:
        adapters = [Path(a) for a in args.adapter]
    else:
        adapters = sorted(Path(args.run_dir).glob("adapter-epoch*"), key=lambda d: int(d.name.removeprefix("adapter-epoch")))
    if not adapters:
        raise SystemExit(f"no adapter-epoch*/ in {args.run_dir}")
    out_path = Path(args.out or Path(args.run_dir) / "val_loss_split.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(l) for l in open(args.data) if l.strip()][:args.limit]

    trainer_losses = {}
    if args.run_dir and (Path(args.run_dir) / "best_epoch.json").exists():
        trainer_losses = json.loads((Path(args.run_dir) / "best_epoch.json").read_text())["val_loss_per_epoch"]

    result = {"data": args.data, "n_pairs": len(rows), "bits": args.bits, "batch_size": args.batch_size,
              "line_rule": "difflib line diff buggy vs target; token assigned to the line of its first character",
              "adapters": {}}
    for adapter in adapters:
        t0 = time.perf_counter()
        model = load_model(args.model, args.bits, str(adapter))
        r = split_loss(model, tokenizer, rows, args.max_length, args.batch_size)
        epoch = adapter.name.removeprefix("adapter-epoch")
        if epoch in trainer_losses:
            r["trainer_eval_loss"] = trainer_losses[epoch]
        r["seconds"] = round(time.perf_counter() - t0, 1)
        result["adapters"][adapter.name if args.run_dir else str(adapter)] = r
        print(adapter, json.dumps(r), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
