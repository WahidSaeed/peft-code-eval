"""DIAGNOSTIC ONLY (RESEARCH_LOG #53): how much do renamed methods cost on HumanEval-Java?

For every saved sample whose code does not declare the method the tests call, the first declared
method is classified:
  - a RENAME if it is the same name spelled differently (case/underscores ignored, similarity >= 0.8,
    e.g. get_closest_vowel vs get_closet_vowel, minSubArraySum vs min_subarray_sum): it is renamed back
    (declaration and calls) and the sample is re-run;
  - a DIFFERENT METHOD otherwise (e.g. the helper `poly` instead of `find_zero`): counted, not re-run.
This is never part of the scoring harness and never used for reported results; it only measures
whether option (a) "keep the benchmark's descriptions unchanged" biases the comparison.

    python analysis/diagnostics/rename_diagnostic.py <predictions.jsonl> <per_sample.jsonl> <out.json>

Runs model-generated code: use the Modal CPU `score` job, not a personal machine.
"""

import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from execution import execute, extract_code, pass_at_k  # noqa: E402

_DECL = re.compile(r"^[ \t]*(?:(?:public|private|protected|static|final|synchronized)\s+)+"
                   r"[\w<>\[\],.? ]+?\s+(\w+)\s*\(", re.M)


def declared(code):
    return [n for n in _DECL.findall(code) if n != "main"]


def same_name(a, b):
    norm = lambda x: x.replace("_", "").lower()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.8


def main(pred_path, per_sample_path, out_path):
    recs = {r["id"]: r for r in map(json.loads, open(pred_path))}
    status = {(s["id"], s["sample"]): s["status"] for s in map(json.loads, open(per_sample_path))}
    jobs, other = [], Counter()
    for rid, r in recs.items():
        expected = declared(r["buggy"])[0]
        for j, pred in enumerate(r["predictions"]):
            code = extract_code(pred)
            names = declared(code)
            if not names or expected in names:
                continue
            if same_name(names[0], expected):
                jobs.append((rid, j, names[0], re.sub(rf"\b{re.escape(names[0])}\b", expected, code)))
            else:
                other[f"{rid}: {names[0]} instead of {expected}"] += 1

    with ThreadPoolExecutor(8) as ex:
        outcomes = list(ex.map(lambda job: execute(job[3], recs[job[0]])[0], jobs))

    per_task = defaultdict(lambda: {"renamed": 0, "pass_after_rename_back": 0, "names": Counter()})
    for (rid, j, wrong, _), out in zip(jobs, outcomes):
        t = per_task[rid]
        t["renamed"] += 1
        t["pass_after_rename_back"] += out == "passed"
        t["names"][wrong] += 1

    # pass@k as scored (v2) vs. with rename-back passes added (diagnostic)
    passed = Counter(rid for (rid, _), st in status.items() if st == "passed" and rid in recs)
    extra = Counter(rid for (rid, _, _, _), out in zip(jobs, outcomes) if out == "passed")
    n = {rid: len(r["predictions"]) for rid, r in recs.items()}

    def pak(add, k):
        return round(100 * sum(pass_at_k(n[r], passed[r] + (extra[r] if add else 0), k) for r in recs) / len(recs), 2)

    summary = {
        "samples_total": sum(n.values()),
        "samples_with_renamed_method": len(jobs),
        "samples_returning_a_different_method": sum(other.values()),
        "different_method_cases": dict(other.most_common()),
        "of_which_pass_after_rename_back": sum(o == "passed" for o in outcomes),
        "outcomes_after_rename_back": dict(Counter(outcomes)),
        "pass_at_1_scored_v2": pak(False, 1), "pass_at_1_if_renames_forgiven": pak(True, 1),
        "pass_at_10_scored_v2": pak(False, 10), "pass_at_10_if_renames_forgiven": pak(True, 10),
        "per_task": {rid: {**t, "names": dict(t["names"])} for rid, t in
                     sorted(per_task.items(), key=lambda kv: -kv[1]["renamed"])},
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
