"""Share of training-target tokens that sit on changed lines (RESEARCH_LOG #86). CPU only.
Line diff (difflib) between buggy input and fixed target; tokens counted with the DeepSeek tokenizer, + EOS.
    python analysis/feasibility/megadiff_edit_share.py
"""
import json, difflib, statistics as st, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
rows = [json.loads(l) for l in open("data/megadiff_train.jsonl")]
shares, ch_lines, tot = [], [], [0, 0]
for r in rows:
    b, f = r["buggy"].splitlines(), r["target"].splitlines()
    changed = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, f, autojunk=False).get_opcodes():
        if op != "equal": changed.update(range(j1, j2))
    n_all = sum(len(tok(l + "\n", add_special_tokens=False)["input_ids"]) for l in f) + 1  # + EOS
    n_ch = sum(len(tok(f[j] + "\n", add_special_tokens=False)["input_ids"]) for j in changed)
    shares.append(n_ch / n_all); ch_lines.append(len(changed)); tot[0] += n_ch; tot[1] += n_all
q = st.quantiles(shares, n=10)
print(f"pairs {len(rows)} | target tokens on changed lines: median {st.median(shares):.1%}, p10 {q[0]:.1%}, p90 {q[8]:.1%}, pooled {tot[0]/tot[1]:.1%}")
print(f"changed target lines per pair: median {st.median(ch_lines)}, pairs with 0 changed target lines (pure deletions) {sum(c==0 for c in ch_lines)}")
