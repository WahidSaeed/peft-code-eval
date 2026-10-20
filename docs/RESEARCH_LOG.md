# Research Log — Quantized PEFT for Automated Program Repair

Chronological record of every discussion, decision, action and finding in this project.
Each entry: **what** was discussed or done, **why**, and **result / evidence**.
Plan of record: the thesis research plan (summarised in `README.md`). Account-level cloud details live in a private ops log that is not published.

Conventions: ✅ done · 🔄 in progress · ⏳ waiting on external party · ❓ open decision · ⚠️ finding that affects methodology

---

## Status board (kept current)

| Item | Status | Notes |
|---|---|---|
| Repo scaffold (`peft-code-eval/`) | ✅ | Folder layout per plan §7 |
| `requirements.txt` | ✅ | tree-sitter / codebleu versions pinned (see 2026-10-20 #7) |
| `src/data_loader.py` | ✅ | Tested on CodeXGLUE Medium |
| `src/profiler.py` | ✅ (untested on GPU) | Compiles; needs GPU for a real test |
| `src/metrics_se.py` | ✅ | Sanity-checked on ground truth |
| `src/trainer.py` | ⏳ | Next |
| `src/evaluator.py` | ⏳ | |
| `src/execution.py` (HumanEvalFix Pass@1) | ✅ | 6 languages; clean subset canonical 100 % / buggy 0 % (#22) |
| AWS GPU quota (eu-central-1) | ⏳ | Denied → appeal sent 2026-10-20 (see #14, #15) |
| AWS budget alert (credits excluded) | ✅ | (#16) |
| AWS IAM user with MFA | ✅ | (#17) |
| Phase 0: reproduce Tier-A papers | ❓ | Needs artifact availability check |
| Pass@k in RQ1 (CodeXGLUE has no tests) | ✅ | Resolved: Pass@1 on HumanEvalFix (#20) |
| Citation verification (DOIs) | ❓ | User's paper list must be verified |
| LinkedIn posts 1–3 + GIFs | ✅ drafts | kept outside this repo |
| Learning path | ✅ | `docs/LEARNING_PATH.md` |
| RunPod fallback (RTX 4090, $0.90/h) | 📋 backlog | Only if AWS denies the appeal (#23) |
| Public GitHub repo | ✅ | https://github.com/WahidSaeed/peft-code-eval (#18) |

---

## 2026-10-20

### 1. Session restart — reconstructing state
- **Context:** Assistant had no memory of the previous conversation. State reconstructed from the research plan and the initial implementation prompt.
- **Finding:** Week 1 deliverables (scaffold, data loader, 50-step M1 dry run) did not exist yet; project is ~1 week behind plan.

### 2. Brainstorm — risks and issues raised by the assistant
1. **Pass@k / test pass rate is not measurable on CodeXGLUE Refinement** ⚠️ — the data is abstracted (`VAR_1`, `METHOD_1`), single methods, no test suites. RQ1 wording promises "patch test pass rates". Options: (A) drop Pass@k, rely on EM + CodeBLEU + syntax validity; (B) add a small executable benchmark, e.g. **QuixBugs (Java, 40 bugs)**. Defects4J considered too costly to set up in the time frame. ❓ *Decision pending.*
2. **Dataset sizes in plan (~10k/1k/1k)** looked inconsistent with CodeXGLUE Medium — verified later (#6): they are a seeded subsample.
3. **tree-sitter on bare methods** gives false syntax errors → must wrap in a dummy class (implemented, #7).
4. **Single seed** is statistically weak for TOSEM/EMSE. Suggestion: 3 seeds for a subset (e.g. M2/M5/M8), paired tests on predictions (McNemar for EM/syntax, Wilcoxon + Cliff's δ for CodeBLEU). ❓ *Not yet decided.*
5. **bitsandbytes 8-bit is known to be slower than 4-bit and bf16** → RQ3 8-bit throughput reflects the implementation, not 8-bit per se. To be reported as a threat to validity.
6. **M0 zero-shot will be near 0 EM** on abstracted code → weak baseline; consider few-shot M0 or frame M0 explicitly as a floor. ❓
7. **Novelty positioning** vs. prior PEFT-for-APR work (RepairLLaMA; Weyssow et al.; Li et al. ASE 2024): our gap = bit-width × rank on APR with measured resource costs and a Pareto-based decision framework.

### 3. User decision: no scope cuts
- Assistant proposed cuts (1–2 epochs, seq len 256, pynvml-only energy, HF generate only).
- **User decision:** "its ok that we are behind the schedule but there will be no cutting...we will stick to the plan."
- **Consequence:** full M0–M9, 3 epochs, seq len 512, all metrics. Items in #2 that are *correctness* fixes (not cuts) are still applied.

### 4. Repo scaffold + `requirements.txt` + `src/data_loader.py` ✅
- Created `peft-code-eval/{configs,src,scripts,results/{logs,predictions},analysis,data}` per plan §7.
- `data_loader.py`:
  - Source: Hugging Face `google/code_x_glue_cc_code_refinement`, subset `medium`.
  - Seeded subsample (`seed=42`, `dataset.shuffle(seed).select(range(n))`) to 10,000 train / 1,000 validation / 1,000 test.
  - Instruction prompt (user turn, DeepSeek chat template):
    > The following Java method contains a bug. Identifiers have been abstracted (e.g. VAR_1, METHOD_1). Return only the corrected Java method, using the same abstracted tokens, with no explanation.
    > Buggy method: {buggy}
  - Target = developer fix + EOS. **Prompt tokens are masked (`-100`) from the loss** — model is trained only on the fix.
  - Writes `data/medium_{train,validation,test}.jsonl`; reports token-length statistics.
- **Tested locally** (Mac, Python 3.14 venv, tokenizer only — no GPU needed).

### 5. Compute discussion → AWS chosen ⏳
- **User question:** Nebius or AWS (existing credit)?
- **Estimate:** ~50–70 GPU-hours total (training 9 configs × ~1.5–3 h; 8-bit slowest; plus inference for M0–M9 and reruns). Rough, to be refined after the dry run.
- **Recommendation:** AWS **g5.2xlarge** (NVIDIA A10G 24 GB, 8 vCPU, 32 GB RAM), on-demand (not Spot, because interruptions would corrupt wall-clock/throughput measurements). ≈ $1.2/h (to be verified), so ≈ $85–100 in total. g5.xlarge rejected: 16 GB system RAM is tight for loading a 6.7B model.
- **Region:** eu-central-1 (Frankfurt) keeps compute in the EU, consistent with the GDPR motivation.
- **Rule:** all configs must run on the *same GPU model*, otherwise RQ2/RQ3 numbers are not comparable.
- Nebius kept as fallback (EU data centres, but mostly H100/L40S, which is off-profile for a 24 GB budget, and it costs real money).
- **Actions:** the "Running On-Demand G and VT instances" quota was **0 vCPU** (g5.2xlarge is offered in eu-central-1). A quota increase to 8 vCPU (one g5.2xlarge) was requested on 2026-10-20. Account-level details are in the private ops log (not published).

### 6. ⚠️ Dataset facts (verified by running the loader)
| Split | Full CodeXGLUE Medium | Our subsample |
|---|---|---|
| train | 52,364 | 10,000 |
| validation | 6,546 | 1,000 |
| test | 6,545 | 1,000 |

Token lengths (DeepSeek chat-formatted prompt + target + EOS):

| Split | mean | median | p95 | max | > 512 |
|---|---|---|---|---|---|
| train | 368.8 | 366 | 455 | 545 | 14 (0.14 %) |
| validation | 369.7 | 367 | 449 | 507 | 0 |
| test | 369.3 | 367 | 453 | 508 | 0 |

→ **Seq len 512 is empirically justified** (only 0.14 % of training samples truncated, none in eval). Thesis methodology must state that 10k/1k/1k is a seeded random subsample of 52k/6.5k/6.5k.

### 7. Paper reproduction discussion
- **User question:** weren't we going to reproduce all cited papers first? Assistant found no such phase in the plan file (earlier session not visible).
- **Assistant position:** reproducing all ~25 papers is infeasible and not expected (different tasks, closed models e.g. Codex, surveys). Reviewers expect reproduction of work you compare against directly.
- **Proposed triage (Phase 0):**
  - **Tier A — reproduce on our pipeline:** Li et al. (ASE 2024, PEFT on APR); Huang et al. (TSE 2025, fine-tuning code LLMs for APR); Wei et al. (ESEC/FSE 2023, quantization + energy for code generation).
  - **Tier B — numerical sanity check / trend comparison:** CodeXGLUE (Lu et al., 2021 — *missing from user's list*); Weyssow et al.; Liu et al. ASE 2023; Shi et al. ISSTA 2023; MFTCoder (qualitative only, 13B+).
  - **Tier C — cite only:** prompt-tuning, code review, commit messages, translation, summarization, code search, test generation, surveys, green-SE vision papers.
- **Missing foundational citations to add:** LoRA (Hu et al., ICLR 2022); QLoRA (Dettmers et al., NeurIPS 2023); LLM.int8() (Dettmers et al., 2022); DeepSeek-Coder (Guo et al., 2024); CodeBLEU (Ren et al., 2020); CodeXGLUE (Lu et al., 2021).
- ⚠️ Every venue/year/title in the candidate list must be verified by DOI before use (e.g. Weyssow et al. was arXiv 2023, TOSEM later).
- ❓ Next: check whether Tier-A papers released code/data.

### 8. `src/profiler.py` ✅ (compiles; GPU test pending)
- `GPUProfiler` context manager: resets and reads `torch.cuda.max_memory_allocated()` **and** `max_memory_reserved()` (reported in GiB), wall-clock duration, samples/s, tokens/s.
- `EnergyMeter`: uses NVML's cumulative GPU energy counter (`nvmlDeviceGetTotalEnergyConsumption`, Volta+, A10G supports it); falls back to 100 ms power sampling integrated over time. The method used is logged with each result.
- `measure_generation`: TTFT measured via a `StoppingCriteria` hook that timestamps the first generated token (prefill + first decode step); also reports total tokens/s and decode-only tokens/s.
- `trainable_parameters`: trainable vs. total params (corrects for bitsandbytes 4-bit packing, 2 weights per byte), adapter size in MB.
- `append_csv`: one-row-per-run CSV logging.

### 9. `src/metrics_se.py` ✅
- **Syntax validity:** `tree-sitter-java`, method wrapped in `class __Wrapper__ { … }`, invalid if root `has_error`.
- **Exact Match:** whitespace-normalised equality.
- **CodeBLEU:** `codebleu` package, `lang="java"`, weights (0.25, 0.25, 0.25, 0.25); reports all four components. Computed on raw (un-detokenised) text, as in standard CodeXGLUE evaluation, for comparability with literature.
- Strips markdown code fences from model output (instruct models often emit them).
- CLI: `python src/metrics_se.py predictions.jsonl [--sanity] [--per_sample_out path]`.

#### ⚠️ Finding 9a — dependency conflict
`codebleu==0.7.0` requires `tree-sitter>=0.22,<0.23`; `tree-sitter-java>=0.23` requires `tree-sitter>=0.23` (crash: `TypeError: an integer is required`). **Fix:** pin `tree-sitter-java~=0.21.0`. Tested combo: codebleu 0.7.0, tree-sitter 0.22.3, tree-sitter-java 0.21.0.

#### ⚠️ Finding 9b — CodeXGLUE tokenisation artefacts break Java parsing
Ground-truth fixes should parse ~100 %. Raw result: only **93.9 %** of test targets parsed. Root cause: CodeXGLUE's tokenisation splits multi-char operators and drops dots:

| Artefact in data | Valid Java | Rule added |
|---|---|---|
| `- >` | `->` | yes |
| `: :` | `::` | yes |
| `> >` / `> > >` | `>>` / `>>>` | yes |
| `< <` | `<<` | yes |
| `VAR_3 class` | `VAR_3.class` | yes |
| `VAR_3 this` | `VAR_3.this` | yes (keywords such as `return this` excluded — a first version wrongly produced `return.this`, caught in testing) |

`detokenize()` is applied **only inside the syntax check** and **identically to predictions and targets**.

Ground-truth validity after the fix:

| Split | Valid | Invalid |
|---|---|---|
| train | 99.96 % | 4 / 10,000 |
| validation | 100.00 % | 0 / 1,000 |
| test | 99.90 % | 1 / 1,000 |

Remaining 5 failures: lost dots before explicit generic method calls (`VAR.<T>method()`) and `Outer.super` — ambiguous to repair, left as-is. **Thesis must report the ground-truth validity ceiling (99.9 % on test)** and the normalisation rules.

#### ⚠️ Finding 9c — naive-copy baseline is very strong on CodeBLEU
Returning the *buggy input unchanged* as the "fix" (test split):

| Metric | Naive copy |
|---|---|
| Exact Match | **0.0 %** |
| CodeBLEU | **0.887** |
| n-gram / weighted n-gram | 0.908 / 0.927 |
| AST match / data-flow match | 0.847 / 0.867 |
| Syntax validity | 99.9 % |

Bugs in this dataset are small edits, so CodeBLEU is high even with no repair. **Implications:** (1) Exact Match must be the primary effectiveness metric; (2) CodeBLEU must always be reported next to the naive-copy reference (0.887); (3) add "naive copy" as a reference row in all result tables (consistent with the CodeXGLUE paper, which also reports a naive-copy baseline).

- Negative control: `public void METHOD_1 ( ) { VAR_1 = ; ` → correctly flagged invalid. Fence stripping verified.

### 10. User instruction: document everything
- **User:** "document every single dot and tees we are discussing and you are doing/performing in .md file."
- **Action:** this log created at `peft-code-eval/docs/RESEARCH_LOG.md`; will be appended after every discussion, decision, command and finding.

### 11. LinkedIn build-in-public series ✅ (drafts)
- **User request:** blog the work for LinkedIn, one folder per post, each with a generated GIF explaining it.
- **Location:** `Research_1/linkedin-posts/` (kept outside the replication package). Each folder has `post.md` (text, status, facts used) and `explainer.gif`. Index in `linkedin-posts/README.md`.
- **Posts:**
  1. `01-thesis-kickoff`: research question, RQ1–RQ4, 3×3 design. Weight sizes 13.4 / 6.7 / 3.4 GB are labelled as *theory* (params × bytes).
  2. `02-validate-your-metric`: Finding 9b (93.9% → 99.9% ground-truth syntax validity).
  3. `03-naive-copy-baseline`: Finding 9c (naive copy: CodeBLEU 0.887, EM 0%).
- **GIF method:** drawn with a Python/Pillow script (`linkedin-posts/_tools/make_gifs.py`) rather than an AI video generator, so every number is exact, it costs nothing, and the GIFs can be regenerated when results change.
- **Accuracy rule:** every number in a post must come from this log. A first GIF draft contained an unmeasured claim ("~95% of tokens identical"); it was replaced with the measured CodeBLEU components (0.908 / 0.847 / 0.867). The `>` / `>=` code example in post 3 is labelled *illustrative*.
- Posts are drafts; the user reviews and publishes them.

### 12. Brand theme for all GIFs ✅
- **User request:** use the theme of https://wahidsaeed.github.io/ for all GIFs (brand consistency).
- Extracted from the site's CSS: IBM Plex Mono; `--terminal-green #00ff41`, `--terminal-dim #00cc33`, `--terminal-bg #0c0c0c`, black body; `> ` prompt; glow `0 0 5px`; blinking cursor; dot grid; accent shapes in `#ff6b9d`, `#c44569`, `#ffa07a`, `#98d8c8`.
- Implemented: terminal-window frame, glow headings, typewriter title cards with a blinking cursor, `[ OK ]`/`[FAIL]` tags (the font lacks ✓/✗), 192-colour GIF palette. IBM Plex Mono downloaded from the google/fonts repo (OFL) into `_tools/fonts/`.

### 13. Learning path while the GPU quota is pending ✅
- **User request:** list all prerequisites so they understand the work themselves rather than relying entirely on the assistant.
- Created `docs/LEARNING_PATH.md`: 9 modules (PyTorch & GPU memory; transformers & generation; LoRA; quantization/QLoRA; APR & the Tufano/CodeXGLUE dataset; metrics; efficiency & energy; empirical-SE statistics; AWS & tooling). Each has reasons, sources, hands-on exercises and self-check questions, plus a 5-day schedule and a concept → code map.
- Includes a LoRA parameter-count exercise (expected ≈ 20M / 40M / 80M trainable parameters for r = 8/16/32 *if* the model config is 32 layers / 4096 hidden / 11008 intermediate; to be confirmed against `config.json` and on the GPU).

### 14. ⚠️ AWS GPU quota request DENIED
- AWS declined the request (new-account quotas are ramped up gradually) and invited an appeal with a detailed use case.
- Every AWS GPU route in eu-central-1 was at 0: On-Demand G/VT, Spot G/VT, and all SageMaker g5/g6/g6e quotas.
- **Fallback if not approved:** RunPod RTX 4090 (already named in plan §4) or Nebius. Methodological constraint: all *official* M0–M9 runs must use one GPU model, so A10G and 4090 results are never mixed. ❓ *Platform decision pending AWS's answer.*

### 15. AWS quota appeal sent ⏳
- Appeal sent on 2026-10-20: one g5.2xlarge, ~70–100 GPU-hours from late October to late November, ~$100–150 covered by credits, plus cost controls (budget alert, IAM user with MFA, stop when idle). It was worded so it promises those controls *before the first launch* rather than claiming they already existed.

### 16. Cost control: budget alert ✅
- Monthly cost budget with email alerts at 80% / 100% actual and 100% forecast. **Credits are excluded** from the budget calculation; otherwise net spend stays ≈ $0 and the alert would never fire while the credit drains.

### 17. Cost control: non-root IAM user with MFA ✅
- Day-to-day console and CLI access moved from the root account to an IAM user with MFA (passkey). Root is kept for billing and emergencies.
- Gotcha: switching an existing `aws login` profile from root to an IAM user asks an overwrite question, which fails in a non-interactive shell ("EOF when reading a line"). `yes | aws login` works.
- Both cost controls promised in the appeal are in place. GPU quota still 0, waiting for AWS's answer.

### 18. Public GitHub repository ✅
- **User request:** make a public GitHub repo.
- **Repo:** https://github.com/WahidSaeed/peft-code-eval (public, branch `main`, MIT license, with topics).
- **Before publishing:** account-level cloud details (account and case IDs, IAM user, email, other-project spending) and the AWS appeal were moved to `docs/private/` (git-ignored). This log keeps redacted summaries with the same research-relevant facts. A secret scan of staged files was clean.
- **Not published:** `data/` (regenerated deterministically by `data_loader.py`; the dataset stays with its original distributor), `results/`, checkpoints, the research plan file and the LinkedIn drafts.
- Added `README.md` (RQs, design matrix, findings so far, quick start, data provenance), `.gitignore`, `LICENSE`.
- **Workflow from now on:** this log is public, so private/ops details go to `docs/private/OPS_LOG.md` and research-relevant facts go here.

### 19. Discussion: add a second dataset in another language? ❓
- **User question:** should we add another coding dataset in a different language?
- **Why it matters:** a single language (Java) on a single abstracted benchmark is an external-validity threat that reviewers will raise.
- **Options considered:**
  - (a) *Train* a second set of M1–M9 on another-language dataset: doubles training (+~60 GPU-h) and analysis, and there is no CodeXGLUE-style Python refinement set with the same format. Advised against before the deadline.
  - (b) **Evaluation-only** on **HumanEvalFix** (HumanEvalPack, Muennighoff et al., "OctoPack", ICLR 2024), *recommended*. Verified on the HF Hub `bigcode/humanevalpack`: 164 hand-written bugs per language (Python, Java, JS, Go, C++, Rust), each with `buggy_solution`, `canonical_solution`, unit tests (`test`) and a `bug_type` label (Python: value misuse 44, missing logic 33, excess logic 31, operator misuse 25, variable misuse 23, function misuse 8).
- **What (b) adds:** (1) **executable tests, so real Pass@1**, which resolves open decision #1 without Defects4J; (2) a second language (Python) plus non-abstracted Java; (3) a measure of whether quantized adapters trained on abstracted Java generalise or cause forgetting (out-of-distribution, new RQ1b).
- **Cost:** no extra training; 164 × 2 languages × 10 models of greedy generation plus sandboxed test execution (Python subprocess; JDK for Java).
- **Caveats to report:** possible HumanEval contamination in pre-training (affects absolute scores, not the relative comparison between configs); domain shift is intended here, not a confound.
- ❓ *Awaiting user decision.* → **Decided in #20: add it.**

### 20. HumanEvalFix added (evaluation only) ✅
- **User decision:** "yeah add it". Languages: **Python + Java**. No extra training.
- **Resolves open decision #1:** Pass@1 with real unit tests replaces the Pass@k promise in RQ1, so the RQ1 wording should say "Pass@1 on HumanEvalFix". A new sub-question **RQ1b** asks whether quantized adapters trained on abstracted Java generalise to real-identifier Java and to Python, or cause forgetting.
- **`data_loader.py --benchmark humanevalfix`** writes `data/humanevalfix_{python,java}.jsonl` with prompt, target (`import + declaration + canonical_solution`), buggy code, bug type, entry point and hidden tests.
- **Prompt design:** buggy code plus the **example tests** (`example_test`); the full **hidden tests** (`test`) are used only for scoring, so the model never sees the tests it is graded on. The model is asked for the complete code (whole `Solution` class with imports for Java; imports + function for Python). This may differ from OctoPack's exact prompt, so absolute numbers are not directly comparable to their leaderboard; relative comparison across M0–M9 is the goal.
- **`src/execution.py`:** builds program = candidate + hidden tests; Python runs with `python -I` (isolated), Java runs with `javac` + `java -Xmx512m`; fresh temp dir per sample; 10 s timeout. Outcomes: passed / failed / compile_error / runtime_error / timeout. If the model returns only a body, the declaration is prepended; missing Java imports are added. Fenced code blocks are extracted for any language tag.
- **Sanity checks (local Mac, Python 3.14, JDK 17):**

| Check | Python | Java |
|---|---|---|
| Canonical fixes | **100 %** (164/164 passed) | **100 %** (164/164 passed) |
| Buggy code | **0 %** (155 failed, 6 runtime error, 3 timeout) | **0 %** (151 failed, 10 runtime error, 3 timeout) |

- ⚠️ **Finding 20a: longer inputs than training.** Prompt + target token lengths (DeepSeek chat template): Python mean 393, p95 621, max 903, 20/164 over 512; Java mean 582, p95 884, max 1,259, **102/164 over 512**. Inference is not truncated (the model's context is 16K), but (1) adapters were trained on ≤ 512-token sequences, which is part of the OOD shift and must be reported; (2) the evaluator needs `max_new_tokens` large enough for the longest Java target (≈ 1,024 for HumanEvalFix vs. less for CodeXGLUE).
- ⚠️ **Safety:** `execution.py` runs model-generated code, so real predictions are scored on the disposable GPU VM or in a container, not on the laptop. The local runs above executed only the benchmark's own reference and buggy code.
- **Threat to validity:** HumanEval may be in DeepSeek-Coder's pre-training data, which inflates absolute scores but affects all configs alike.

### 21. Discussion: the other HumanEvalPack languages (C++, Go, Rust, JavaScript) ❓
- **User observation:** HumanEvalFix also covers cpp, go, rust and js (164 bugs each, verified on the Hub).
- **Cost of adding them:** no training; about 4 × 164 × 10 = 6,560 extra generations (batched, roughly 2–4 GPU-hours on top), plus per-language harness work.
- **Harness requirements found while inspecting the data:**
  - **JavaScript** tests use `console.assert`, which in Node **does not throw**: it prints "Assertion failed" and exits 0. A naive runner would score every buggy program as passing, so the runner must make `console.assert` throw (or check stderr).
  - **Go** tests use `testing` + `github.com/stretchr/testify/assert`, so they need a Go module with testify and must run through `go test`.
  - **Rust** tests are `#[cfg(test)]` modules run via `cargo test`; some tasks may need crates (to be checked), so use one shared, pre-built cargo project.
  - **C++** tests are a `main()` with `assert` (`#undef NDEBUG` included), so g++/clang++ with C++17 works; the md5 task may need OpenSSL.
- **Local toolchain:** clang++ 14, rustc/cargo 1.95, Node 24 present; Go to be checked.
- **Validation rule (same as #20):** each language must pass the canonical = 100 % / buggy ≈ 0 % sanity check before use. Any canonical failure is documented and the task excluded for that language.
- ❓ Awaiting user decision. → **Decided in #22: add all four.**

### 22. HumanEvalFix extended to all 6 languages ✅
- **User decision:** "yeah add all of them". HumanEvalFix now covers Python, Java, C++, Go, Rust and JavaScript (6 × 164 = 984 tasks), still evaluation-only.
- **Harness per language (`src/execution.py`):** C++ via `c++ -std=c++17` (Boost and OpenSSL paths added; `-lcrypto` when needed); Go puts code and tests in **one file** built with `go test -c` against a cached module with testify; Rust uses one shared cargo project with `rand 0.7`, `regex 1`, `md5 0.7` (deps compiled once; `cargo test --no-run` → compile_error, then `cargo test`); JavaScript runs `node --check` and then `node` with **`console.assert` patched to throw**.
- **Problems found and fixed while validating (each would have biased Pass@1):**
  1. **JavaScript `console.assert` doesn't throw in Node.** It prints and exits 0, so failing code would score as passing. Patched to throw.
  2. **The dataset's `entry_point` disagrees with the real function name in 17 tasks** (Java 3, C++ 2, Rust 11, JS 1; e.g. CPP/121 `solution` vs `solutions(`, CPP/156 `int_to_mini_roman` vs `int_to_mini_romank`, JS/114 casing). The harness wrongly prepended the declaration again. **Fix:** `data_loader.function_name()` reads the name from the declaration itself (new field `function_name`).
  3. **Go declarations omit helper functions** that the prompt defines (11 tasks: Go/10, 32, 38, 50, 72, 107, 108, 141, 145, 158, 163). Fix: helpers are extracted from the prompt (without the target's doc comment) and included in buggy and target code.
  4. **Go rejects unused or missing imports.** Code and tests now share one file, and the import list is repaired deterministically (drop unused, add missing standard-library packages, at most 3 rounds), identically for every candidate. Logic is never changed. The Go 1.19 wording `imported and not used: "x"` differs from newer Go's `"x" imported and not used`; both are handled.
  5. **Rust tests use the rand 0.7 API** (`gen_range(low, high)`), so `rand` is pinned to 0.7.
  6. **JavaScript/162 requires the npm package `js-md5`.** A local shim on Node's `crypto` is used instead of downloading it.
- **Sanity results (local Mac):**

| Language | Canonical (all 164) | Buggy (all 164) | Canonical (clean) | Buggy (clean) |
|---|---|---|---|---|
| Python | 100 % | 0 % | 100 % | 0 % |
| Java | 100 % | 0 % | 100 % | 0 % |
| C++ | 99.39 % | 0.61 % | 100 % (n=162) | 0 % |
| Go | 100 %* | 0 % | 100 % (n=163) | 0 % |
| Rust | 99.39 %* | 0 % | 100 % (n=163) | 0 % |
| JavaScript | 100 % | 0 % | 100 % | 0 % |
\* flaky tasks pass or fail from run to run.

- ⚠️ **Defective tasks (dataset issues, `DEFECTIVE_TASKS` in `execution.py`):** CPP/38, the reference fix doesn't compile (unbalanced braces: 5 `{` vs 4 `}`); CPP/23, the buggy code passes the hidden tests; Go/95, flaky, since the reference depends on random map iteration order (failed 1 of 5 repeated runs); Rust/32, flaky, since an unseeded random test sometimes makes the reference Newton solver loop forever (passed in 1.5 s once, timed out after 90 s in another run).
- **Reporting rule:** Pass@1 on all 164 per language (comparable with the literature) **and** on the clean subset (162–164 per language). The thesis uses the clean subset as primary and lists the excluded tasks.
- **Token lengths (prompt + target):** C++ mean 597 / max 1,591 (103 over 512); Go 502 / 1,012 (70); Rust 605 / 2,008 (99); JS 464 / 1,221 (49). Same OOD-length caveat as Finding 20a; the evaluator should allow up to about 1,024 new tokens (Rust max target is longer, so check).
- **GPU VM toolchain:** `apt install -y default-jdk g++ libboost-dev libssl-dev golang nodejs cargo`. The first run downloads testify and the Rust crates into `~/.cache/peft-code-eval`.

### 23. RunPod explored as the GPU platform while AWS reviews the appeal ⏳
- **User request:** check RunPod in Chrome and pick suitable options.
- Skipped RunPod's onboarding prompts, which suggest pasting a setup line into an AI agent (page content, not a user instruction). Chose **Pods** → template **PyTorch 2.8.0** (`runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404`, CUDA 12.8.1, Ubuntu 24.04) → **1× RTX 4090**.
- **Prices seen (on-demand, 2026-10-20):** RTX 4090 $0.89/h (24 GB, 31 GB RAM, 8 vCPU, 7–8 available); RTX 3090 $0.50/h (24 GB, 62 GB RAM, 2 available); RTX A5000 $0.27/h (24 GB, 1 available); L4 $0.59/h; A40 $0.59/h (48 GB); RTX 5090 $1.19/h (32 GB).
- **Chosen config (not deployed):** 1× RTX 4090, on-demand, $0.90/h total (GPU $0.89 + 50 GB container disk $0.007). The 4090 is the consumer 24 GB GPU named in the plan, and has the best availability (needed to rerun on the *same* model for weeks). A5000/3090 are cheaper but only 1–2 available.
- **Cost estimate:** ~70–100 GPU-h → ~$65–90 plus storage. RunPod is real money, unlike the AWS credit.
- **Storage:** the container disk is erased when the pod stops. Recommend a **network volume (~100 GB, EU data centre)** for the HF model cache, data, adapters and results; it persists independently of the pod, but pins the pod to that data centre's GPUs.
- **Blockers (user actions):** RunPod balance is **$0.00**, so credits must be loaded before deploying; RunPod account MFA is not enabled (banner); an SSH public key is needed in Settings for terminal access.
- **Methodology:** official M0–M9 runs must all use one GPU model.
- **User decision:** RunPod goes to the **backlog**. It's used **only if AWS denies the appeal**; then all runs (dry run + official) move to RunPod RTX 4090 with the config above. Nothing deployed, no credits loaded.

### 24. Discussion: add MBPP? → No ✅
- **User question:** should MBPP be added to the evaluation?
- **Assessment:** MBPP is code *generation* (a function written from a short description), not repair. The RQs are about program repair, which CodeXGLUE (in-distribution) and HumanEvalFix (out-of-distribution, 6 languages, unit tests) already cover. MBPP would only serve a *capability-retention / forgetting* question (does quantized fine-tuning hurt general generation?), which is not in the plan. If that question were ever wanted, HumanEvalSynthesize (same 164 problems × 6 languages, same harness) would fit better than MBPP (Python only, new harness).
- **User decision:** not added. "as you said our research is code fix not generation." The scope stays program repair only; generation benchmarks are out of scope.

---

## Open decisions (❓) — to resolve with the user
1. ~~Pass@k~~ → resolved by HumanEvalFix Pass@1 (#20).
2. Multiple seeds for a subset of configs (statistical testing)?
3. M0: zero-shot or few-shot?
4. Phase 0 reproduction: confirm Tier A list once artifact availability is checked.
5. GPU platform: **AWS A10G** if the appeal is approved; **RunPod RTX 4090 (backlog)** only if denied (#23).
7. ~~Other HumanEvalPack languages~~ → all six added (#22).
6. ~~Second language~~ → HumanEvalFix added (#20). Still to do: update the RQ1 wording in the plan and thesis.
