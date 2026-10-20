"""Test execution for HumanEvalFix (Pass@1) in Python, Java, C++, Go, Rust and JavaScript.

Each candidate fix is combined with the benchmark's hidden unit tests, built,
and run with a timeout. Outcome per sample:
    passed | failed (assertion / wrong output) | compile_error | runtime_error | timeout

Per-language harness:
    python  python -I prog.py                       (isolated interpreter)
    java    javac Main.java && java Main
    cpp     c++ -std=c++17 (Boost + OpenSSL for 3 tasks) && ./prog
    go      module with testify; go test -c, then run the test binary
    rust    one shared cargo project (rand, regex, md5); cargo test --no-run, then cargo test
    js      node, with console.assert patched to THROW (Node's console.assert only
            prints "Assertion failed" and exits 0, which would score failing code as passing)

SAFETY: this executes model-generated code. Run it on a disposable machine or
container (e.g. the GPU VM), never on a personal laptop with real data, except
for the canonical/buggy sanity checks shipped with the benchmark.

Toolchain setup on the GPU VM (Ubuntu):
    apt install -y default-jdk g++ libboost-dev libssl-dev golang nodejs cargo
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

TIMEOUT_S = 10
COMPILE_TIMEOUT_S = 120
CACHE_DIR = Path(os.environ.get("HEF_CACHE", Path.home() / ".cache" / "peft-code-eval"))

JAVA_IMPORTS = "import java.util.*;\nimport java.lang.*;\n"
JS_STRICT_ASSERT = (
    "console.assert = (cond, ...msg) => { if (!cond) { throw new Error('AssertionError ' + msg.join(' ')); } };\n"
)
JS_MD5_SHIM = (
    "const crypto = require('crypto');\n"
    "module.exports = (text) => crypto.createHash('md5').update(text).digest('hex');\n"
)
GO_MOD = "module hef\n\ngo 1.19\n\nrequire github.com/stretchr/testify v1.8.4\n"
CARGO_TOML = """[package]
name = "hef"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.7"  # the Rust tests use the 0.7 API gen_range(low, high)
regex = "1"
md5 = "0.7"
"""
# Tasks found defective in the harness sanity checks (see docs/RESEARCH_LOG.md #22).
# Pass@1 is reported on all 164 tasks (comparable with the literature) AND on the
# clean subset without these.
DEFECTIVE_TASKS = {
    "CPP/38": "reference fix does not compile (missing closing brace in the dataset)",
    "CPP/23": "buggy code passes the hidden tests (tests do not detect the bug)",
    "Go/95": "flaky: reference fix depends on Go's random map iteration order (failed 1 of 5 runs)",
    "Rust/32": "flaky: unseeded random test; reference Newton solver sometimes never converges (timeout)",
}
_FENCE = re.compile(r"```[\w+#-]*[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """First fenced code block if the model produced one, else the raw text."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def _run(cmd, cwd, timeout, env=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None


def _outcome(result, fail_markers):
    """Classify a finished test run."""
    if result is None:
        return "timeout", ""
    if result.returncode == 0:
        return "passed", ""
    out = result.stdout + result.stderr
    status = "failed" if any(m in out for m in fail_markers) else "runtime_error"
    return status, out[-500:]


def _with_header(code, record, marker):
    """Prepend the original declaration if the model returned only a function body."""
    return code if marker in code else record["declaration"] + code


# ----------------------------------------------------------------------------- python

def run_python(code, record, timeout):
    code = _with_header(code, record, f"def {record['function_name']}")
    program = f"{record['test_setup']}\n{code}\n\n{record['test']}\n"
    try:
        compile(program, "prog.py", "exec")
    except SyntaxError as e:
        return "compile_error", str(e)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "prog.py").write_text(program)
        return _outcome(_run([sys.executable, "-I", "prog.py"], tmp, timeout), ["AssertionError"])


# ----------------------------------------------------------------------------- java

def run_java(code, record, timeout):
    code = _with_header(code, record, "class Solution")
    if "import java.util" not in code:
        code = JAVA_IMPORTS + code
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "Main.java").write_text(f"{code}\n\n{record['test']}\n")
        c = _run(["javac", "-nowarn", "Main.java"], tmp, COMPILE_TIMEOUT_S)
        if c is None or c.returncode != 0:
            return "compile_error", (c.stderr[-500:] if c else "javac timeout")
        return _outcome(_run(["java", "-Xmx512m", "-cp", tmp, "Main"], tmp, timeout), ["AssertionError"])


# ----------------------------------------------------------------------------- c++

def _cpp_flags():
    flags = []
    for pkg in ("boost", "openssl@3"):  # Homebrew on macOS; system paths on Linux
        prefix = Path(f"/opt/homebrew/opt/{pkg}")
        if prefix.exists():
            flags += [f"-I{prefix / 'include'}", f"-L{prefix / 'lib'}"]
    return flags


def run_cpp(code, record, timeout):
    code = _with_header(code, record, f"{record['function_name']}(")
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "prog.cpp").write_text(f"{code}\n\n{record['test']}\n")
        cmd = ["c++", "-std=c++17", "-O1", "-w", "prog.cpp", "-o", "prog", *_cpp_flags()]
        if "openssl" in code:
            cmd.append("-lcrypto")
        c = _run(cmd, tmp, COMPILE_TIMEOUT_S)
        if c is None or c.returncode != 0:
            return "compile_error", (c.stderr[-500:] if c else "compile timeout")
        return _outcome(_run(["./prog"], tmp, timeout), ["Assertion failed", "assertion"])


# ----------------------------------------------------------------------------- go

def _go_project():
    """Module dir with testify downloaded once; reused for every sample."""
    proj = CACHE_DIR / "go"
    if not (proj / "go.sum").exists():
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "go.mod").write_text(GO_MOD)
        (proj / "main.go").write_text("package main\n")
        (proj / "main_test.go").write_text(
            'package main\n\nimport (\n    "testing"\n    "github.com/stretchr/testify/assert"\n)\n\n'
            "func TestX(t *testing.T) { assert.True(t, true) }\n")
        r = _run(["go", "mod", "tidy"], proj, 300)
        if r is None or r.returncode != 0:
            raise RuntimeError(f"go mod tidy failed: {r.stderr if r else 'timeout'}")
    return proj


GO_STD_PACKAGES = ("math", "strings", "strconv", "sort", "fmt", "math/rand", "time", "crypto/md5",
                   "bytes", "unicode", "regexp", "reflect", "errors", "container/heap")
_GO_IMPORT_BLOCK = re.compile(r"^import\s*(\([^)]*\)|\"[^\"]+\")", re.M)
_GO_UNUSED = re.compile(r'"([\w/]+)" imported and not used|imported and not used: "([\w/]+)"')  # new / old Go wording
_GO_UNDEFINED = re.compile(r"undefined: (\w+)")


def _go_split(src):
    """Strip `package` and `import` lines from a Go fragment; return (imports, body)."""
    imports = set()
    for block in _GO_IMPORT_BLOCK.findall(src):
        imports.update(re.findall(r'"([^"]+)"', block))
    body = _GO_IMPORT_BLOCK.sub("", re.sub(r"^package \w+\s*$", "", src, flags=re.M))
    return imports, body


def _go_file(imports, code_body, test_body):
    imp = "\n".join(f'    "{i}"' for i in sorted(imports))
    return f"package main\n\nimport (\n{imp}\n)\n\n{code_body}\n\n{test_body}\n"


def run_go(code, record, timeout):
    """Code and tests go into ONE file (main_test.go). Go rejects unused or missing imports,
    so the import list is repaired deterministically: unused imports are dropped and
    missing standard-library packages added (at most 3 rounds). The same repair applies
    to every candidate, reference and model alike; logic errors are never touched."""
    proj = _go_project()
    code = _with_header(code, record, f"func {record['function_name']}(")
    code_imports, code_body = _go_split(code)
    test_imports, test_body = _go_split(f"{record['test_setup']}\n{record['test']}")
    imports = code_imports | test_imports
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("go.mod", "go.sum"):
            shutil.copy(proj / f, tmp)
        Path(tmp, "main.go").write_text("package main\n")
        for _ in range(4):
            Path(tmp, "main_test.go").write_text(_go_file(imports, code_body, test_body))
            c = _run(["go", "test", "-c", "-o", "hef.test", "."], tmp, COMPILE_TIMEOUT_S)
            if c is None:
                return "compile_error", "go build timeout"
            if c.returncode == 0:
                break
            out = c.stdout + c.stderr
            unused = {a or b for a, b in _GO_UNUSED.findall(out)}
            missing = {p for p in GO_STD_PACKAGES if p.split("/")[-1] in _GO_UNDEFINED.findall(out)}
            if not (unused or missing - imports):
                return "compile_error", out[-500:]
            imports = (imports - unused) | missing
        else:
            return "compile_error", out[-500:]
        return _outcome(_run(["./hef.test", f"-test.timeout={timeout}s"], tmp, timeout + 5), ["Error Trace", "--- FAIL"])


# ----------------------------------------------------------------------------- rust

def _rust_project():
    """Shared cargo project; dependencies compile once, samples only rebuild lib.rs. Not thread-safe."""
    proj = CACHE_DIR / "rust"
    if not (proj / "Cargo.lock").exists():
        (proj / "src").mkdir(parents=True, exist_ok=True)
        (proj / "Cargo.toml").write_text(CARGO_TOML)
        (proj / "src" / "lib.rs").write_text("pub fn f() {}\n")
        r = _run(["cargo", "test", "--no-run", "-q"], proj, 900)
        if r is None or r.returncode != 0:
            raise RuntimeError(f"cargo setup failed: {r.stderr if r else 'timeout'}")
    return proj


def run_rust(code, record, timeout):
    proj = _rust_project()
    code = _with_header(code, record, f"fn {record['function_name']}")
    (proj / "src" / "lib.rs").write_text(f"#![allow(unused, deprecated)]\n{code}\n\n{record['test']}\n")
    c = _run(["cargo", "test", "--no-run", "-q"], proj, COMPILE_TIMEOUT_S)
    if c is None or c.returncode != 0:
        return "compile_error", (c.stderr[-500:] if c else "cargo build timeout")
    return _outcome(_run(["cargo", "test", "-q"], proj, timeout + 30), ["panicked", "FAILED"])


# ----------------------------------------------------------------------------- javascript

def run_js(code, record, timeout):
    code = _with_header(code, record, f"const {record['function_name']}")
    program = f"{JS_STRICT_ASSERT}{record['test_setup']}\n{code}\n\n{record['test']}\n"
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "prog.js").write_text(program)
        shim = Path(tmp, "node_modules", "js-md5")  # JavaScript/162 requires the npm package 'js-md5'
        shim.mkdir(parents=True)
        (shim / "index.js").write_text(JS_MD5_SHIM)
        c = _run(["node", "--check", "prog.js"], tmp, COMPILE_TIMEOUT_S)
        if c is None or c.returncode != 0:
            return "compile_error", (c.stderr[-500:] if c else "node --check timeout")
        return _outcome(_run(["node", "prog.js"], tmp, timeout), ["AssertionError"])


RUNNERS = {"python": run_python, "java": run_java, "cpp": run_cpp, "go": run_go, "rust": run_rust, "js": run_js}


def execute(code, record, timeout=TIMEOUT_S):
    return RUNNERS[record["language"]](code, record, timeout)


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
        clean = [s for s in rows if s["id"] not in DEFECTIVE_TASKS]
        summary[language] = {
            "n": len(rows),
            "pass_at_1_pct": round(100 * counts["passed"] / len(rows), 2),
            "n_clean": len(clean),
            "pass_at_1_clean_pct": round(100 * sum(s["status"] == "passed" for s in clean) / len(clean), 2),
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
