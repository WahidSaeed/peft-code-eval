"""CodeXGLUE Java Code Refinement loader.

Fetches the Medium split from the Hugging Face Hub, subsamples it to the sizes
fixed in the research plan (10k / 1k / 1k, seeded), and turns each buggy/fixed
pair into an instruction-formatted prompt-target pair for DeepSeek-Coder-Instruct.
"""

import argparse
import json
import statistics
from pathlib import Path

from datasets import DatasetDict, load_dataset

DATASET_ID = "google/code_x_glue_cc_code_refinement"
DEFAULT_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
IGNORE_INDEX = -100

INSTRUCTION = (
    "The following Java method contains a bug. Identifiers have been abstracted "
    "(e.g. VAR_1, METHOD_1). Return only the corrected Java method, using the "
    "same abstracted tokens, with no explanation.\n\n"
    "Buggy method:\n{buggy}"
)


def load_splits(subset="medium", train_size=10_000, val_size=1_000, test_size=1_000, seed=42):
    """Load CodeXGLUE refinement and draw a seeded random subsample per split.

    A size of None keeps the full split.
    """
    raw = load_dataset(DATASET_ID, subset)
    sizes = {"train": train_size, "validation": val_size, "test": test_size}
    out = {}
    for split, n in sizes.items():
        ds = raw[split]
        if n is not None and n < len(ds):
            ds = ds.shuffle(seed=seed).select(range(n))
        out[split] = ds.map(to_prompt_target, remove_columns=ds.column_names)
    return DatasetDict(out), {s: len(raw[s]) for s in raw}


def to_prompt_target(example):
    return {
        "id": example["id"],
        "prompt": INSTRUCTION.format(buggy=example["buggy"].strip()),
        "target": example["fixed"].strip(),
    }


def chat_prompt(tokenizer, prompt):
    """Render the user turn with the model's chat template, ending at the assistant cue."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def tokenize_for_training(dataset, tokenizer, max_length=512):
    """Tokenize prompt+target; prompt tokens are masked out of the loss.

    Adds a boolean `truncated` column so truncation can be reported.
    """

    def _tok(example):
        prompt_ids = tokenizer(chat_prompt(tokenizer, example["prompt"]), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(example["target"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        truncated = len(input_ids) > max_length
        input_ids, labels = input_ids[:max_length], labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "truncated": truncated,
        }

    return dataset.map(_tok, remove_columns=dataset.column_names)


def length_report(dataset, tokenizer, max_length=512):
    """Token-length statistics of the full chat-formatted sequence."""
    lengths = [
        len(tokenizer(chat_prompt(tokenizer, ex["prompt"]) + ex["target"], add_special_tokens=False)["input_ids"]) + 1
        for ex in dataset
    ]
    lengths.sort()
    return {
        "n": len(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
        "over_max_length": sum(l > max_length for l in lengths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="medium", choices=["small", "medium"])
    parser.add_argument("--train_size", type=int, default=10_000)
    parser.add_argument("--val_size", type=int, default=1_000)
    parser.add_argument("--test_size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer used for the length report")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    splits, full_sizes = load_splits(args.subset, args.train_size, args.val_size, args.test_size, args.seed)
    print(f"Full split sizes ({args.subset}): {full_sizes}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, ds in splits.items():
        path = out / f"{args.subset}_{split}.jsonl"
        with path.open("w") as f:
            for ex in ds:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(ds)} examples to {path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for split, ds in splits.items():
        print(f"{split} token lengths: {length_report(ds, tokenizer, args.max_length)}")


if __name__ == "__main__":
    main()
