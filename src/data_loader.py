"""Benchmark loaders.

CodeXGLUE Java Code Refinement (training + in-distribution test): fetches the
Medium split from the Hugging Face Hub, subsamples it to the sizes fixed in the
research plan (10k / 1k / 1k, seeded), and turns each buggy/fixed pair into an
instruction-formatted prompt-target pair for DeepSeek-Coder-Instruct.

HumanEvalFix (evaluation only, out-of-distribution): 164 hand-written bugs per
language from HumanEvalPack (Muennighoff et al., 2024), with unit tests for Pass@1.
The prompt shows the buggy function and the *example* tests; the full hidden
tests are kept for scoring only.
"""

import argparse
import json
import re
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

HUMANEVALPACK_ID = "bigcode/humanevalpack"
HEF_LANGUAGES = ("python", "java", "cpp", "go", "rust", "js")
# language -> (display name, markdown fence tag, what "complete code" means)
HEF_LANGUAGE_INFO = {
    "python": ("Python", "python", "imports and the whole function"),
    "java": ("Java", "java", "the whole `Solution` class, including imports"),
    "cpp": ("C++", "cpp", "#include lines and the whole function"),
    "go": ("Go", "go", "imports and the whole function"),
    "rust": ("Rust", "rust", "use statements and the whole function"),
    "js": ("JavaScript", "javascript", "the whole function"),
}
HEF_INSTRUCTION = (
    "The following {language} code contains a bug in `{entry_point}`. "
    "Fix the bug and return the complete corrected code ({scope}), with no explanation.\n\n"
    "Buggy code:\n```{language_tag}\n{buggy}\n```\n\n"
    "Example tests that the fixed code must pass:\n```{language_tag}\n{example_test}\n```"
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


_FUNC_NAME = {
    "python": re.compile(r"^def (\w+)\s*\(", re.M),
    "java": re.compile(r"^\s*public [\w<>\[\], ]+? (\w+)\s*\(", re.M),
    "cpp": re.compile(r"^[\w:<>,\* ]+?\b(\w+)\s*\([^;{]*\)\s*\{", re.M),
    "go": re.compile(r"^func (\w+)\s*\(", re.M),
    "rust": re.compile(r"^fn (\w+)\s*\(", re.M),
    "js": re.compile(r"^const (\w+)\s*=", re.M),
}


def function_name(declaration, language):
    """Name of the function to fix, read from the declaration itself.

    The dataset's `entry_point` field sometimes disagrees with the code
    (e.g. CPP/121 'solution' vs `solutions(`, JavaScript/114 casing).
    """
    names = _FUNC_NAME[language].findall(declaration)
    return names[-1] if names else None


def _go_helpers(ex):
    """Helper functions that Go's `prompt` defines but its `declaration` omits (e.g. Go/10 IsPalindrome)."""
    prompt, signature = ex["prompt"], ex["declaration"].strip()
    cut = prompt.find(signature)
    if cut < 0:
        return ""
    head = prompt[:cut].rstrip("\n").split("\n")
    while head and head[-1].lstrip().startswith("//"):  # drop the target function's doc comment
        head.pop()
    head = "\n".join(head).replace(ex["import"].strip(), "", 1)
    return head.strip() + "\n" if "func " in head else ""


def load_humanevalfix(languages=HEF_LANGUAGES):
    """HumanEvalFix records: prompt, target (canonical fix) and everything needed to run the tests."""
    records = []
    for language in languages:
        name, tag, scope = HEF_LANGUAGE_INFO[language]
        for ex in load_dataset(HUMANEVALPACK_ID, language, split="test"):
            helpers = _go_helpers(ex) if language == "go" else ""
            header = ex["import"] + helpers + ex["declaration"]
            records.append({
                "id": ex["task_id"],
                "language": language,
                "bug_type": ex["bug_type"],
                "entry_point": ex["entry_point"],
                "function_name": function_name(ex["declaration"], language) or ex["entry_point"],
                "prompt": HEF_INSTRUCTION.format(
                    language=name,
                    language_tag=tag,
                    entry_point=function_name(ex["declaration"], language) or ex["entry_point"],
                    scope=scope,
                    buggy=(header + ex["buggy_solution"]).strip(),
                    example_test=ex["example_test"].strip(),
                ),
                "buggy": (header + ex["buggy_solution"]).strip(),
                "target": (header + ex["canonical_solution"]).strip(),
                "declaration": header,
                "test_setup": ex["test_setup"],
                "test": ex["test"],
            })
    return records


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
    parser.add_argument("--benchmark", default="codexglue", choices=["codexglue", "humanevalfix"])
    parser.add_argument("--subset", default="medium", choices=["small", "medium"])
    parser.add_argument("--train_size", type=int, default=10_000)
    parser.add_argument("--val_size", type=int, default=1_000)
    parser.add_argument("--test_size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer used for the length report")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    if args.benchmark == "humanevalfix":
        main_humanevalfix(args)
        return

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


def main_humanevalfix(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_humanevalfix()
    for language in HEF_LANGUAGES:
        rows = [r for r in records if r["language"] == language]
        path = out / f"humanevalfix_{language}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows)} examples to {path}")
        print(f"{language} token lengths: {length_report(rows, tokenizer, args.max_length)}")


if __name__ == "__main__":
    main()
