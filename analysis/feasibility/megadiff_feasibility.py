"""Feasibility statistics for Megadiff single-function as PEFT training data (RESEARCH_LOG #38)."""
import difflib
import json
import re
import statistics
import sys
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

REPO = sys.argv[1]
MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
INSTR = ("The following Java method contains a bug. Return only the corrected Java method, "
         "with no explanation.\n\nBuggy method:\n")

ds = load_dataset("ASSERT-KTH/megadiff-single-function", split="train")
tok = AutoTokenizer.from_pretrained(MODEL)
n = len(ds)
print(f"rows: {n}")
print("is_single_function:", Counter(ds["is_single_function"]))
print("is_single_chunk:", Counter(ds["is_single_chunk"]))

ws = lambda s: " ".join(s.split())
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
nocomment = lambda s: ws(_COMMENT.sub(" ", s))

buggy, fixed = ds["buggy_function"], ds["fixed_function"]
identical = sum(b == f for b, f in zip(buggy, fixed))
ws_only = sum(b != f and ws(b) == ws(f) for b, f in zip(buggy, fixed))
comment_only = sum(ws(b) != ws(f) and nocomment(b) == nocomment(f) for b, f in zip(buggy, fixed))
print(f"identical: {identical}, whitespace-only: {ws_only}, comment-only: {comment_only}")

pairs = Counter((ws(b), ws(f)) for b, f in zip(buggy, fixed))
dup_pairs = sum(c - 1 for c in pairs.values() if c > 1)
bug_keys = Counter(ws(b) for b in buggy)
dup_buggy = sum(c - 1 for c in bug_keys.values() if c > 1)
print(f"exact duplicate pairs (extra copies): {dup_pairs}; duplicate buggy functions: {dup_buggy}")

# changed lines per pair (line-level diff of the two functions)
changed = []
for b, f in zip(buggy, fixed):
    d = difflib.unified_diff(b.splitlines(), f.splitlines(), lineterm="", n=0)
    changed.append(sum(1 for l in d if (l.startswith("+") or l.startswith("-"))
                       and not l.startswith(("+++", "---"))))
q = statistics.quantiles(changed, n=20)
print(f"changed lines: median {statistics.median(changed)}, p75 {q[14]}, p95 {q[18]}, max {max(changed)}")
for lim in (2, 6, 10, 20):
    print(f"  <= {lim} changed lines: {100 * sum(c <= lim for c in changed) / n:.1f}%")

# tokens: chat-formatted prompt + target + EOS, as tokenize_for_training builds them
def n_tokens(b, f):
    prompt = tok.apply_chat_template([{"role": "user", "content": INSTR + b}], tokenize=False,
                                     add_generation_prompt=True)
    return len(tok(prompt, add_special_tokens=False)["input_ids"]) + len(tok(f)["input_ids"]) + 1

lengths = [n_tokens(b, f) for b, f in zip(buggy, fixed)]
q = statistics.quantiles(lengths, n=20)
print(f"tokens (prompt+target): mean {statistics.mean(lengths):.0f}, median {statistics.median(lengths)}, "
      f"p75 {q[14]}, p95 {q[18]}, max {max(lengths)}")
for lim in (512, 768, 1024, 2048):
    print(f"  <= {lim}: {sum(l <= lim for l in lengths)} ({100 * sum(l <= lim for l in lengths) / n:.1f}%)")

# usable pool after cleaning: drop identical / whitespace / comment-only, dedupe pairs
seen, clean = set(), []
for i, (b, f) in enumerate(zip(buggy, fixed)):
    if nocomment(b) == nocomment(f):
        continue
    key = (ws(b), ws(f))
    if key in seen:
        continue
    seen.add(key)
    clean.append(i)
print(f"clean (non-trivial, deduplicated): {len(clean)}")
for lim in (512, 1024):
    print(f"  clean and <= {lim} tokens: {sum(lengths[i] <= lim for i in clean)}")

# contamination: HumanEvalFix-Java function names and canonical bodies inside Megadiff
hef = [json.loads(l) for l in open(f"{REPO}/data/humanevalfix_java.jsonl")]
names = {r["function_name"] for r in hef if r.get("function_name")}
corpus = "\n".join(buggy) + "\n".join(fixed)
name_hits = sorted(nm for nm in names if re.search(rf"\b{re.escape(nm)}\s*\(", corpus))
print(f"HumanEvalFix-Java function names found in Megadiff: {len(name_hits)}/{len(names)} {name_hits[:30]}")

def grams(s, k=20):
    t = re.findall(r"\w+|[^\w\s]", s)
    return {" ".join(t[i:i + k]) for i in range(len(t) - k + 1)}

hef_grams = set().union(*(grams(r["target"]) for r in hef))
hits = [i for i, (b, f) in enumerate(zip(buggy, fixed)) if hef_grams & (grams(b) | grams(f))]
print(f"Megadiff rows sharing a 20-token sequence with a HumanEvalFix-Java fix: {len(hits)}")

json.dump({"lengths": lengths, "changed": changed, "clean": clean, "contam_rows": hits},
          open(sys.argv[2], "w"))
