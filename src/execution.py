"""Test execution for HumanEvalFix (Pass@1).

Each candidate fix is combined with the benchmark's hidden unit tests and run
in a fresh temporary directory with a timeout. Python runs in an isolated
interpreter (`-I`); Java is compiled with javac and run with java.

Outcome per sample: passed | failed (assertion / wrong output) | compile_error
(syntax / javac error) | runtime_error | timeout.

SAFETY: this executes model-generated code. Run it on a disposable machine or
container (e.g. the GPU VM), never on a personal laptop with real data, except
for the canonical/buggy sanity checks shipped with the benchmark.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

TIMEOUT_S = 10
JAVA_IMPORTS = "import java.util.*;\nimport java.lang.*;\n"
_FENCE = re.compile(r"```[\w+#-]*[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """First fenced code block if the model produced one, else the raw text."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def build_program(code, record):
    """Make a runnable program from a candidate fix and the record's tests."""
    if record["language"] == "python":
        if f"def {record['entry_point']}" not in code:  # model returned only the body
            code = record["declaration"] + code
        return f"{record['test_setup']}\n{code}\n\n{record['test']}\n"
    if "class Solution" not in code:  # model returned only the method body
        code = record["declaration"] + code
    if "import java.util" not in code:
        code = JAVA_IMPORTS + code
    return f"{record['test_setup']}\n{code}\n\n{record['test']}\n"


def _run(cmd, cwd, timeout):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def run_python(program, timeout=TIMEOUT_S):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "prog.py").write_text(program)
        try:
            compile(program, "prog.py", "exec")
        except SyntaxError as e:
            return "compile_error", str(e)
        r = _run([sys.executable, "-I", "prog.py"], tmp, timeout)
        if r is None:
            return "timeout", ""
        if r.returncode == 0:
            return "passed", ""
        return ("failed" if "AssertionError" in r.stderr else "runtime_error"), r.stderr[-500:]


def run_java(program, timeout=TIMEOUT_S):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "Main.java").write_text(program)
        c = _run(["javac", "-nowarn", "Main.java"], tmp, 60)
        if c is None or c.returncode != 0:
            return "compile_error", (c.stderr[-500:] if c else "javac timeout")
        r = _run(["java", "-Xmx512m", "-cp", tmp, "Main"], tmp, timeout)
        if r is None:
            return "timeout", ""
        if r.returncode == 0:
            return "passed", ""
        return ("failed" if "AssertionError" in r.stderr else "runtime_error"), r.stderr[-500:]


def execute(code, record, timeout=TIMEOUT_S):
    program = build_program(code, record)
    runner = run_python if record["language"] == "python" else run_java
    return runner(program, timeout)


def pass_at_1(records, codes, timeout=TIMEOUT_S):
    """Greedy Pass@1 plus an outcome breakdown and per-sample results."""
    per_sample = []
    for rec, code in zip(records, codes):
        status, detail = execute(code, rec, timeout)
        per_sample.append({"id": rec["id"], "language": rec["language"], "bug_type": rec["bug_type"],
                           "status": status, "detail": detail})
    summary = {}
    for language in sorted({r["language"] for r in records}):
        rows = [s for s in per_sample if s["language"] == language]
        counts = Counter(s["status"] for s in rows)
        summary[language] = {
            "n": len(rows),
            "pass_at_1_pct": round(100 * counts["passed"] / len(rows), 2),
            "outcomes": dict(counts),
        }
    return summary, per_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="+", help="humanevalfix_<lang>.jsonl file(s); add 'prediction' fields to score a model")
    parser.add_argument("--sanity", choices=["canonical", "buggy"],
                        help="run the reference fixes (expect 100%%) or the buggy code (expect ~0%%) instead of predictions")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    parser.add_argument("--per_sample_out")
    args = parser.parse_args()

    records = [json.loads(l) for p in args.data for l in open(p) if l.strip()]
    if args.sanity == "canonical":
        codes = [r["target"] for r in records]
    elif args.sanity == "buggy":
        codes = [r["buggy"] for r in records]
    else:
        codes = [extract_code(r["prediction"]) for r in records]

    summary, per_sample = pass_at_1(records, codes, args.timeout)
    print(json.dumps(summary, indent=2))
    if args.per_sample_out:
        with open(args.per_sample_out, "w") as f:
            for s in per_sample:
                f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
