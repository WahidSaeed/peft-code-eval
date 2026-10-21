"""Feasibility statistics for Code4Bench as a Java repair benchmark (RESEARCH_LOG #57/#58).

Streams the MySQL dump (no database server needed), keeps Java submissions, and reports:
faulty->accepted Java pairs, who wrote them, how large the change is, the verdict of the faulty
version, test-case completeness per problem, statements that allow several correct outputs
(need a special checker), and submission dates (contamination vs DeepSeek-Coder's pre-training).

    python analysis/feasibility/code4bench_feasibility.py <dump_dir> <out.json> [java.sqlite]

Runs on the Modal CPU job (the dump is 6.4 GB); only reads data, never executes submissions.
"""

import difflib
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

JAVA = {7: "Java 8", 25: "Java 7", 32: "Java 6", 29: "Java 8 ZIP"}
_ESC = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "\\": "\\", "'": "'", '"': '"'}
_UNESC = re.compile(r"\\(.)", re.S)
STR = r"'((?:[^'\\]|\\.)*)'"
NUM = r"(NULL|-?\d+(?:\.\d+)?)"
SOURCE_ROW = re.compile(rf"\((\d+),(\d+),{STR},(\d+),{NUM},{NUM},(NULL|'[^']*'),{NUM},{NUM},{NUM},(\d+),{NUM}\)", re.S)
TEST_ROW = re.compile(rf"\((\d+),(NULL|{STR}),(NULL|{STR}),(\d+),(NULL|'\w+')\)", re.S)
PROBLEM_ROW = re.compile(rf"\((\d+),{STR},(\d+),{STR},{STR}\)", re.S)
FAULT_ROW = re.compile(r"\((\d+),(NULL|\d+),(NULL|\d+),(NULL|\d+),")
MULTI_ANSWER = re.compile(r"(print|output) any|any of them|multiple (correct )?(answers|solutions)|"
                          r"several (possible )?(answers|solutions)|absolute or relative error", re.I)


def unescape(s):
    return _UNESC.sub(lambda m: _ESC.get(m.group(1), m.group(1)), s)


def insert_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("INSERT INTO"):
                yield line


def num(x):
    return None if x == "NULL" else int(float(x))


def changed_lines(a, b):
    d = difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
    return sum(1 for l in d if l[:1] in "+-" and not l.startswith(("+++", "---")))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def main(dump, out_path, sqlite_path=None):
    dump = Path(dump)
    stats = {}

    # ---- problems: statements that allow several outputs; contest ids
    problems = {}
    for line in insert_lines(dump / "code4bench_problems.sql"):
        for m in PROBLEM_ROW.finditer(line):
            pid, fullname, contest, _, context = m.groups()
            text = re.sub(r"<[^>]+>", " ", unescape(context))
            problems[int(pid)] = {"contest": int(contest), "multi": bool(MULTI_ANSWER.search(text))}
    stats["problems"] = len(problems)
    stats["problems_multi_answer_or_tolerance"] = sum(p["multi"] for p in problems.values())

    # ---- test cases: validity and truncation per problem
    tests = defaultdict(Counter)
    for line in insert_lines(dump / "code4bench_testcases.sql"):
        for m in TEST_ROW.finditer(line):
            pid, valid = int(m.group(6)), (m.group(7) or "NULL").strip("'")
            inp, exp = m.group(3) or "", m.group(5) or ""
            truncated = inp.endswith("...") or exp.endswith("...") or len(inp) >= 600 or len(exp) >= 600
            tests[pid][valid] += 1
            tests[pid]["truncated"] += truncated
            tests[pid]["usable"] += valid == "pass" and not truncated
    stats["testcases_by_validity"] = dict(sum((Counter({k: v for k, v in c.items() if k not in ("truncated", "usable")})
                                               for c in tests.values()), Counter()))
    stats["testcases_truncated"] = sum(c["truncated"] for c in tests.values())
    stats["problems_with_tests"] = len(tests)
    stats["problems_all_tests_usable"] = sum(1 for c in tests.values()
                                             if c["usable"] and c["usable"] == sum(v for k, v in c.items()
                                                                                   if k not in ("truncated", "usable")))

    # ---- source: Java submissions (code kept in sqlite to bound memory)
    db = sqlite3.connect(sqlite_path or ":memory:")
    db.execute("CREATE TABLE IF NOT EXISTS java (submission INTEGER PRIMARY KEY, problem INTEGER, verdict INTEGER,"
                " author INTEGER, sent TEXT, lines INTEGER, lang INTEGER, dup INTEGER, code TEXT)")
    lang_counts, verdict_counts, years = Counter(), Counter(), Counter()
    n_rows = 0
    for line in insert_lines(dump / "code4bench_source.sql"):
        rows = []
        for m in SOURCE_ROW.finditer(line):
            n_rows += 1
            lang = int(m.group(11))
            if lang not in JAVA:
                continue
            sent = None if m.group(7) == "NULL" else m.group(7).strip("'")
            rows.append((int(m.group(2)), num(m.group(9)), num(m.group(10)), int(m.group(4)), sent,
                         num(m.group(8)), lang, num(m.group(12)), unescape(m.group(3))))
            lang_counts[JAVA[lang]] += 1
            verdict_counts[num(m.group(10))] += 1
            years[(sent or "unknown")[:4]] += 1
        db.executemany("INSERT OR REPLACE INTO java VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    stats["source_rows_parsed"] = n_rows
    stats["java_submissions"] = dict(lang_counts)
    stats["java_verdicts"] = {str(k): v for k, v in verdict_counts.most_common(12)}
    stats["java_submission_years"] = dict(sorted(years.items()))

    # ---- fault pairs restricted to Java on both sides
    java_ids = {r[0] for r in db.execute("SELECT submission FROM java")}
    pairs, n_pairs_all = [], 0
    for line in insert_lines(dump / "code4bench_realfaultslocations.sql"):
        for m in FAULT_ROW.finditer(line):
            n_pairs_all += 1
            acc, wrong = num(m.group(2)), num(m.group(3))
            if acc in java_ids and wrong in java_ids:
                pairs.append((wrong, acc))
    stats["fault_pairs_all_languages"] = n_pairs_all
    stats["fault_pairs_java"] = len(pairs)

    meta = {r[0]: r[1:] for r in db.execute("SELECT submission, problem, verdict, author, sent, lines, dup, code FROM java")}
    same_author = same_problem = 0
    wrong_verdicts, sizes, by_problem = Counter(), [], Counter()
    seen, candidates = set(), []
    for wrong, acc in pairs:
        w, a = meta[wrong], meta[acc]
        same_problem += w[0] == a[0]
        same_author += w[2] == a[2]
        wrong_verdicts[w[1]] += 1
        key = (w[6], a[6])
        if w[0] != a[0] or key in seen:
            continue
        seen.add(key)
        n_changed = changed_lines(w[6], a[6])
        sizes.append(n_changed)
        p, t = problems.get(w[0], {}), tests.get(w[0], Counter())
        if (w[2] == a[2] and w[1] == 2 and n_changed <= 6 and not p.get("multi", True)
                and t["usable"] and t["usable"] == t["pass"] and not t["truncated"]):
            candidates.append((wrong, acc))
            by_problem[w[0]] += 1
    stats["java_pairs_same_problem"] = same_problem
    stats["java_pairs_same_author"] = same_author
    stats["java_pairs_unique_same_problem"] = len(sizes)
    stats["java_wrong_verdicts"] = {str(k): v for k, v in wrong_verdicts.most_common(8)}
    stats["changed_lines"] = {"median": statistics.median(sizes) if sizes else None, "p75": pct(sizes, .75),
                              "p95": pct(sizes, .95), "le_2": sum(s <= 2 for s in sizes),
                              "le_6": sum(s <= 6 for s in sizes), "le_10": sum(s <= 10 for s in sizes)}
    stats["candidate_pairs"] = {
        "rule": "same author and problem, faulty verdict 'Wrong answer on test', <= 6 changed lines, "
                "statement without multiple-answer/tolerance wording, all tests validated 'pass' and untruncated",
        "pairs": len(candidates), "problems": len(by_problem),
        "pairs_per_problem_median": statistics.median(by_problem.values()) if by_problem else None,
    }
    contests = sorted(problems[p]["contest"] for p in by_problem if p in problems)
    stats["candidate_contest_ids"] = {"min": contests[0] if contests else None, "max": contests[-1] if contests else None}
    Path(out_path).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
