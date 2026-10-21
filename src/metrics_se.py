"""Software-engineering correctness metrics for RQ1.

- Syntax Validity Rate: tree-sitter-java parse without ERROR/MISSING nodes.
  Megadiff samples are single methods, so each one is wrapped in a dummy class
  before parsing (a bare method is not a valid compilation unit).
- Exact Match: whitespace-normalised string equality with the developer fix.
- CodeBLEU (Ren et al., 2020): n-gram, weighted n-gram, AST and data-flow match.
"""

import argparse
import json
import re
from pathlib import Path

import tree_sitter_java
from codebleu import calc_codebleu
from tree_sitter import Language, Parser

JAVA = Language(tree_sitter_java.language())
_parser = Parser(JAVA)

WRAPPER_OPEN = "class __Wrapper__ {\n"
WRAPPER_CLOSE = "\n}"
_FENCE = re.compile(r"```[\w+#-]*[ \t]*\n(.*?)```", re.DOTALL)  # any language tag

def extract_code(text):
    """Strip a markdown code fence if the model produced one."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def is_syntax_valid(code):
    tree = _parser.parse((WRAPPER_OPEN + code + WRAPPER_CLOSE).encode("utf-8"))
    return not tree.root_node.has_error


def normalize(code):
    return " ".join(code.split())


def exact_match(prediction, target):
    return normalize(prediction) == normalize(target)


def evaluate(predictions, targets):
    """Corpus-level metrics plus per-sample syntax/EM flags."""
    assert len(predictions) == len(targets)
    syntax = [is_syntax_valid(p) for p in predictions]
    em = [exact_match(p, t) for p, t in zip(predictions, targets)]
    cb = calc_codebleu(targets, predictions, lang="java", weights=(0.25, 0.25, 0.25, 0.25))
    n = len(predictions)
    summary = {
        "n": n,
        "syntax_validity_pct": round(100 * sum(syntax) / n, 2),
        "exact_match_pct": round(100 * sum(em) / n, 2),
        "codebleu": round(cb["codebleu"], 4),
        "ngram_match": round(cb["ngram_match_score"], 4),
        "weighted_ngram_match": round(cb["weighted_ngram_match_score"], 4),
        "syntax_match": round(cb["syntax_match_score"], 4),
        "dataflow_match": round(cb["dataflow_match_score"], 4),
    }
    per_sample = [{"syntax_valid": s, "exact_match": e} for s, e in zip(syntax, em)]
    return summary, per_sample


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="JSONL with 'predictions' (first one is scored) and 'target' fields")
    parser.add_argument("--sanity", action="store_true",
                        help="score the targets against themselves (expects ~100%% validity, EM 100, CodeBLEU 1)")
    parser.add_argument("--per_sample_out", help="optional JSONL path for per-sample flags")
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    targets = [r["target"] for r in rows]
    preds = targets if args.sanity else [extract_code(r["predictions"][0]) for r in rows]
    summary, per_sample = evaluate(preds, targets)
    print(json.dumps(summary, indent=2))

    if args.per_sample_out:
        Path(args.per_sample_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_sample_out, "w") as f:
            for r, flags in zip(rows, per_sample):
                f.write(json.dumps({"id": r.get("id"), **flags}) + "\n")


if __name__ == "__main__":
    main()
