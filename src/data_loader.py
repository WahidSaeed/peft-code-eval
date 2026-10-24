"""Benchmark loaders.

Megadiff single-function (training + in-task test, RESEARCH_LOG #38/#39): real Java
commit changes (Monperrus et al., 2021) from the Hugging Face Hub. Trivial and
duplicate pairs and pairs overlapping the test benchmarks are dropped, pairs longer
than --max_length tokens are dropped (never truncated), and the rest is split by
FILE PATH (every change to one file lands in one split): 1k test and 1k validation, then
every remaining pair for training (~18k; seed 42; RESEARCH_LOG #77). --train_size N gives
the earlier 10k subset (#39), whose validation and test splits are identical.

HumanEvalFix-Java (evaluation only): 164 hand-written bugs from HumanEvalPack
(Muennighoff et al., 2024), with unit tests. The prompt shows the buggy code and the
*example* tests; the full hidden tests are kept for scoring only. The other five
HumanEvalPack languages are still supported but not used (RESEARCH_LOG #39).

HumanEval-Java (evaluation only): 163 single-method bugs with JUnit tests (Jiang et
al., ICSE 2023; lin-tan/clm, BSD-3). The prompt is the Megadiff training format with the
original HumanEval problem description as a comment above the buggy method (#40);
the harness puts the predicted method back into the buggy class.
"""

import argparse
import inspect
import json
import random
import re
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

MEGADIFF_ID = "ASSERT-KTH/megadiff-single-function"
DEFAULT_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
IGNORE_INDEX = -100

INSTRUCTION = (
    "The following Java method contains a bug. Return only the corrected Java method, "
    "with no explanation.\n\nBuggy method:\n{buggy}"
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

# HumanEval-Java tasks excluded from the clean set (sanity check, RESEARCH_LOG #38)
HEJ_DEFECTIVE = {"ISCUBE": "buggy file is identical to the correct file (no bug)"}

_DIFF_PATH = re.compile(r"^diff --git a/(\S+) b/")
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_WORD_OR_SYMBOL = re.compile(r"\w+|[^\w\s]")


def _ws(s):
    return " ".join(s.split())


def _method(code):
    """Method text without the class-level indentation it had in its file."""
    return textwrap.dedent(code).strip()


def ngrams(code, k=20):
    t = _WORD_OR_SYMBOL.findall(code)
    return {" ".join(t[i:i + k]) for i in range(len(t) - k + 1)}


def sequence_length(tokenizer, prompt, target):
    """Tokens of the chat-formatted prompt + target + EOS, exactly as tokenize_for_training builds them."""
    return (len(tokenizer(chat_prompt(tokenizer, prompt), add_special_tokens=False)["input_ids"])
            + len(tokenizer(target, add_special_tokens=False)["input_ids"]) + 1)


def load_megadiff(tokenizer, test_code, max_length=768, sizes=(None, 1_000, 1_000), seed=42):
    """Clean, filter and split Megadiff. Returns ({split: records}, counts of what was dropped).

    test_code: code of the evaluation benchmarks; Megadiff pairs sharing a 20-token
    sequence with any of it are dropped (contamination guard).
    sizes: (train, validation, test); train None = every path not used by test or validation.
    """
    ds = load_dataset(MEGADIFF_ID, split="train")
    counts = {"rows": len(ds)}
    bench_grams = set().union(*(ngrams(c) for c in test_code))
    seen, rows = set(), []
    n_trivial = n_dup = n_contam = n_long = 0
    for i, ex in enumerate(ds):
        buggy, fixed = _method(ex["buggy_function"]), _method(ex["fixed_function"])
        if _ws(_COMMENT.sub(" ", buggy)) == _ws(_COMMENT.sub(" ", fixed)):
            n_trivial += 1  # identical, whitespace-only or comment-only change
            continue
        key = (_ws(buggy), _ws(fixed))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        if bench_grams & (ngrams(buggy) | ngrams(fixed)):
            n_contam += 1
            continue
        prompt = INSTRUCTION.format(buggy=buggy)
        if sequence_length(tokenizer, prompt, fixed) > max_length:
            n_long += 1
            continue
        rows.append({"id": f"megadiff/{i}", "path": _DIFF_PATH.match(ex["diff"]).group(1),
                     "prompt": prompt, "buggy": buggy, "target": fixed})
    counts.update(trivial=n_trivial, duplicate=n_dup, contaminated=n_contam,
                  over_max_length=n_long, pool=len(rows))

    # Split by file path: shuffle the paths, fill test, then validation, then train
    by_path = defaultdict(list)
    for r in rows:
        by_path[r["path"]].append(r)
    paths = sorted(by_path)
    random.Random(seed).shuffle(paths)
    names, targets = ("test", "validation", "train"), dict(zip(("train", "validation", "test"), sizes))
    targets["train"] = targets["train"] or len(rows)  # None: all remaining paths
    splits, k = {name: [] for name in names}, 0
    for name in names:
        while k < len(paths) and len(splits[name]) < targets[name]:
            splits[name] += by_path[paths[k]]
            k += 1
    rng = random.Random(seed)
    for name in names:  # a path group can overshoot the target size; trim by random sampling
        if len(splits[name]) > targets[name]:
            splits[name] = sorted(rng.sample(splits[name], targets[name]), key=lambda r: int(r["id"].split("/")[1]))
    counts["paths"] = len(paths)
    counts["paths_unused"] = len(paths) - k
    return splits, counts


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
                "benchmark": "humanevalfix",
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


_HEJ_PACKAGE = re.compile(r"^package humaneval\.\w+;", re.M)


def _comment_text(body):
    """Text of a /* */ or /** */ comment body, without leading '*' markers and common indentation."""
    return inspect.cleandoc("\n".join(re.sub(r"^\s*\*(?!/) ?", "", l).rstrip() for l in body.splitlines()).strip("\n"))


def humaneval_descriptions(hej_dir):
    """HumanEval-Java task name -> docstring of the original HumanEval task.

    Line i of humaneval_loc.txt is HumanEval/i (9 tasks were renamed, e.g. f -> FACTORIAL).
    CORRECT_BRACKETING occurs twice (HumanEval/56 and /61); the Java task is /56 (the "<>"
    version, as its source comment shows), so the first occurrence wins and /61 has no Java task.
    """
    root = Path(hej_dir)
    problems = [json.loads(l) for l in open(root / "HumanEval.jsonl") if l.strip()]
    names = [l.split()[0] for l in open(root / "src/main/java/humaneval/humaneval_loc.txt") if l.strip()]
    assert len(names) == len(problems) == 164
    out = {}
    for name, p in zip(names, problems):
        if name in out:
            continue
        m = re.search(rf"def {re.escape(p['entry_point'])}\(.*?(\"\"\"|\'\'\')(.*?)\1", p["prompt"], re.S)
        out[name] = inspect.cleandoc(m.group(2)).replace("*/", "* /")
    return out


def load_humaneval_java(hej_dir):
    """HumanEval-Java records from a checkout of lin-tan/clm's `humaneval-java` folder.

    `buggy_class` is the whole buggy file (package line normalised: GET_ROW's buggy and
    correct files have swapped packages); lines begin..end (1-based) are the buggy method
    that the prompt shows and the harness replaces with the prediction.
    """
    root = Path(hej_dir)
    meta = json.loads((root / "humaneval-java-sf.json").read_text())
    originals = humaneval_descriptions(root)
    records = []
    for name in sorted(meta):
        m = meta[name]
        buggy_class = _HEJ_PACKAGE.sub("package humaneval.buggy;",
                                       (root / "src/main/java/humaneval/buggy" / f"{name}.java").read_text(), count=1)
        method = "".join(buggy_class.splitlines(keepends=True)[m["begin"] - 1:m["end"]])
        assert method.expandtabs(4) == m["buggy"].expandtabs(4), name  # MAKE_A_PILE, WORDS_STRINGS: tabs in the file
        buggy = _method(method)
        # Description: the comment above the class where the Java file has one (90 tasks; the authors
        # adapted some to the Java tests, e.g. ADD_ELEMENTS, ANY_INT), else the HumanEval original (73)
        comment = re.search(r"/\*(.*?)\*/", "".join(buggy_class.splitlines(keepends=True)[:m["begin"] - 1]), re.S)
        description = _comment_text(comment.group(1)) if comment else originals[name]
        records.append({
            "id": f"HumanEval-Java/{name}",
            "language": "java",
            "benchmark": "humaneval_java",
            "name": name,
            "defective": HEJ_DEFECTIVE.get(name, ""),
            "description_source": "java_file" if comment else "humaneval",
            "prompt": INSTRUCTION.format(buggy=f"/* {description} */\n{buggy}"),
            "buggy": buggy,  # method only: the harness, the buggy sanity run and the contamination check use it
            "target": _method(m["fix"]),
            "buggy_class": buggy_class,
            "begin": m["begin"],
            "end": m["end"],
            "test": (root / "src/test/java/humaneval" / f"TEST_{name}.java").read_text(),
        })
    return records


def chat_prompt(tokenizer, prompt):
    """Render the user turn with the model's chat template, ending at the assistant cue."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def tokenize_for_training(dataset, tokenizer, max_length=768):
    """Tokenize prompt+target; prompt tokens are masked out of the loss.

    Adds a boolean `truncated` column so truncation can be reported (Megadiff pairs
    longer than max_length are dropped when the splits are built, so it should be 0).
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


def length_report(records, tokenizer, max_length):
    """Token-length statistics of the full chat-formatted sequence (prompt + target + EOS)."""
    lengths = sorted(sequence_length(tokenizer, r["prompt"], r["target"]) for r in records)
    return {
        "n": len(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
        "over_max_length": sum(l > max_length for l in lengths),
    }


def _write(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hej_dir", default="data/clm/humaneval-java",
                        help="humaneval-java folder of a lin-tan/clm checkout (see README)")
    parser.add_argument("--hef_languages", nargs="+", default=["java"], choices=HEF_LANGUAGES)
    parser.add_argument("--train_size", type=int, default=0,
                        help="0 = every pair not in test or validation (#77); 10000 = the earlier subset (#39)")
    parser.add_argument("--val_size", type=int, default=1_000)
    parser.add_argument("--test_size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer for the length filter and report")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Evaluation benchmarks first: their code is the contamination reference for Megadiff
    hef = load_humanevalfix(args.hef_languages)
    for language in args.hef_languages:
        rows = [r for r in hef if r["language"] == language]
        _write(out / f"humanevalfix_{language}.jsonl", rows)
        print(f"  token lengths: {length_report(rows, tokenizer, args.max_length)}")
    hej = load_humaneval_java(args.hej_dir)
    _write(out / "humaneval_java.jsonl", hej)
    print(f"  token lengths: {length_report(hej, tokenizer, args.max_length)}")

    test_code = [r[k] for r in hef + hej for k in ("buggy", "target")]
    splits, counts = load_megadiff(tokenizer, test_code, args.max_length,
                                   (args.train_size or None, args.val_size, args.test_size), args.seed)
    print(f"Megadiff: {json.dumps(counts)}")
    for split in ("train", "validation", "test"):
        _write(out / f"megadiff_{split}.jsonl", splits[split])
        print(f"  token lengths: {length_report(splits[split], tokenizer, args.max_length)}")
    paths = {s: {r["path"] for r in splits[s]} for s in splits}
    assert not (paths["train"] & paths["test"] or paths["train"] & paths["validation"]
                or paths["validation"] & paths["test"]), "a file path occurs in two splits"


if __name__ == "__main__":
    main()
