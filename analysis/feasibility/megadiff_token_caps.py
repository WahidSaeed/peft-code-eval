"""Megadiff pairs under several token caps, with the cleaning of src/data_loader.py (RESEARCH_LOG #79).

Run from the repo root after src/data_loader.py has written data/humaneval*_java.jsonl.
"""
import json, sys, statistics
sys.path.insert(0, "src")
from data_loader import (MEGADIFF_ID, DEFAULT_MODEL, INSTRUCTION, _method, _ws, _COMMENT, ngrams,
                         sequence_length, _DIFF_PATH)
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
test_code = []
for f in ("data/humanevalfix_java.jsonl", "data/humaneval_java.jsonl"):
    for line in open(f):
        r = json.loads(line); test_code += [r["buggy"], r["target"]]
bench = set().union(*(ngrams(c) for c in test_code))
ds = load_dataset(MEGADIFF_ID, split="train")
seen, lens, paths = set(), [], []
c = dict(rows=len(ds), trivial=0, duplicate=0, contaminated=0)
for ex in ds:
    b, f = _method(ex["buggy_function"]), _method(ex["fixed_function"])
    if _ws(_COMMENT.sub(" ", b)) == _ws(_COMMENT.sub(" ", f)): c["trivial"] += 1; continue
    k = (_ws(b), _ws(f))
    if k in seen: c["duplicate"] += 1; continue
    seen.add(k)
    if bench & (ngrams(b) | ngrams(f)): c["contaminated"] += 1; continue
    lens.append(sequence_length(tok, INSTRUCTION.format(buggy=b), f))
    paths.append(_DIFF_PATH.match(ex["diff"]).group(1))
c["clean"] = len(lens)
print(json.dumps(c))
for cap in (768, 1024, 1536, 2048):
    sel = [(l, p) for l, p in zip(lens, paths) if l <= cap]
    L = [l for l, _ in sel]
    print(json.dumps(dict(cap=cap, pairs=len(L), paths=len({p for _, p in sel}),
                          train_if_2k_held_out=len(L) - 2000, total_tokens=sum(L),
                          mean=round(statistics.mean(L)), median=statistics.median(L))))
