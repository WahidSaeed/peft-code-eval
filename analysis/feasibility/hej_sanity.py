"""HumanEval-Java sanity: canonical fixes should pass all JUnit tests, buggy versions should fail.

Uses javac + JUnitCore with the jars shipped in lib/ (no Maven). Usage: hej_sanity.py <humaneval-java dir>
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1])
SF = json.loads((ROOT / "humaneval-java-sf.json").read_text())
CP = ":".join(str(p) for p in (ROOT / "lib").glob("*.jar"))


def run(name, variant):
    # whole-file versions; the package line is normalised because GET_ROW's two files have swapped packages
    folder = "correct" if variant == "fix" else "buggy"
    text = (ROOT / "src/main/java/humaneval" / folder / f"{name}.java").read_text()
    src = [re.sub(r"^package humaneval\.\w+;", "package humaneval.buggy;", text, count=1, flags=re.M)]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "humaneval/buggy").mkdir(parents=True)
        (d / "out").mkdir()
        (d / "humaneval/buggy" / f"{name}.java").write_text("".join(src))
        shutil.copy(ROOT / "src/test/java/humaneval" / f"TEST_{name}.java", d / "humaneval")
        c = subprocess.run(["javac", "-nowarn", "-cp", CP, "-d", str(d / "out"),
                            str(d / "humaneval/buggy" / f"{name}.java"), str(d / "humaneval" / f"TEST_{name}.java")],
                           capture_output=True, text=True, timeout=120)
        if c.returncode:
            return "compile_error"
        try:
            r = subprocess.run(["java", "-Xmx512m", "-cp", f"{d / 'out'}:{CP}", "org.junit.runner.JUnitCore",
                                f"humaneval.TEST_{name}"], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "timeout"
        return "passed" if r.returncode == 0 else "failed"


names = sorted(SF)
print(f"bugs in sf.json: {len(names)}")
for variant in ("fix", "buggy"):
    with ThreadPoolExecutor(8) as ex:
        out = dict(zip(names, ex.map(lambda n: run(n, variant), names)))
    print(variant, Counter(out.values()))
    odd = [n for n, o in out.items() if (o != "passed") == (variant == "fix")]
    print("  unexpected:", odd)
