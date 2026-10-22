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
| `src/data_loader.py` | ✅ | Megadiff (≤ 768, path split) + HumanEvalFix-Java + HumanEval-Java (#39) |
| `src/profiler.py` | ✅ (untested on GPU) | Compiles; needs GPU for a real test |
| `src/metrics_se.py` | ✅ | Sanity-checked on ground truth |
| `src/trainer.py` | ✅ (CPU smoke test) | GPU paths validated by the dry run (#25) |
| `src/evaluator.py` | ✅ (CPU smoke test) | Greedy (Megadiff) / n = 20 sampling (test-based) + greedy batch-1 latency probe (#26, #39) |
| `scripts/run_matrix.sh`, `scripts/evaluate_all.sh` | ✅ | Tested in a bash 5 container (#26) |
| `src/execution.py` (pass@1, pass@10) | ✅ | HumanEvalFix-Java + HumanEval-Java: clean sets canonical 100 % / buggy 0 % (#39); other HEF languages validated in #22 |
| GPU dry run | ✅ | Modal A10/A10G: all bit-widths fit 24 GB; ~170–200 GPU-h total (#46) |
| AWS GPU quota (eu-central-1) | ⏳ | Denied → appeal sent 2026-10-20 (see #14, #15) |
| AWS budget alert (credits excluded) | ✅ | (#16) |
| AWS IAM user with MFA | ✅ | (#17) |
| Phase 0: reproduce Tier-A papers | ✅ | Artifacts checked (#28): Li ✅ full, Huang ✅ code+data, Wei ❌ none; scope decision pending |
| Pass@k in RQ1 | ✅ | pass@1 primary + pass@10 secondary, n = 20, T = 0.2 (#37, #39) |
| Training data | ✅ | Megadiff replaces CodeXGLUE (#34–#39) |
| Citation verification (DOIs) | ❓ | User's paper list must be verified |
| Novelty check + positioning table | ✅ | No prior bit-width × rank study for APR found (#29) |
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

### 25. `src/trainer.py` ✅ (CPU smoke-tested; GPU paths pending)
- **CLI:** `--bits {4,8,16} --rank R [--alpha A] --output_dir DIR [--run_name M1] [--max_steps N]` plus debugging flags (`--train_limit`, `--eval_limit`, `--no_eval`). α defaults to 2r.
- **Model loading:** 4-bit = NF4 + double quantization, bf16 compute (QLoRA); 8-bit = LLM.int8(); 16-bit = bf16. For 4/8-bit, `prepare_model_for_kbit_training` (standard QLoRA recipe; casts norms to fp32 and enables gradient checkpointing); for 16-bit, gradient checkpointing + input grads. SDPA attention; `use_cache=False` during training.
- **LoRA:** all linear layers `q,k,v,o,gate,up,down`, dropout 0.05, no bias.
- **Hyperparameters fixed by the plan:** LR 2e-4, batch 4 × accumulation 4 (effective 16), 3 epochs, max length 512, seed 42.
- **Hyperparameters the plan left open, chosen here (overridable, all recorded in `run_config.json`):**
  - optimizer **paged AdamW 32-bit for every bit-width**, so the optimizer is never a confound (the plan said "Paged AdamW 8-bit / AdamW"); paging only engages under memory pressure;
  - **cosine** schedule with **3 % warmup**; **grad-norm clip 0.3** (QLoRA paper default); weight decay 0;
  - validation loss after **each epoch** (overfitting check across the 3 epochs); checkpoint per epoch (keep 1) for crash recovery; final adapter in `adapter/`.
- **Profiling:** `GPUProfiler` wraps `trainer.train()`. Recorded: peak VRAM allocated and reserved, **static VRAM after model load** (weights + adapters, for RQ2), duration with and without evaluation time, samples/s, tokens/s (real non-pad tokens), energy (NVML), GPU name, library versions.
- **Verified on CPU** with a tiny 2-layer Llama using DeepSeek's real tokenizer: full pipeline runs, adapter + `run_config.json` + CSV row written; initial loss 10.36 ≈ ln(32,022) = 10.37, as expected for an untrained model.
- **Label masking verified on real data:** trained tokens are exactly `target + <|EOT|>` (3/3 examples); training starts right after `### Response:\n`; pad token `<｜end▁of▁sentence｜>` (32014) ≠ EOS `<|EOT|>` (32021), so padding can't hide the EOS.
- ✅ **Learning-path exercise confirmed** (meta-device model, no weights loaded): DeepSeek-Coder-6.7B = LLaMA, 32 layers, hidden 4096, intermediate 11008, vocab 32,256, 6,740,512,768 params. LoRA trainable = 2,498,560 × r → **19.99M (0.297 %) / 39.98M (0.593 %) / 79.95M (1.186 %)** for r = 8/16/32; adapters 38 / 76 / 152 MB in bf16.
- **Compatibility:** transformers 5 removed `warmup_ratio` (a float ratio now goes in `warmup_steps`); `trainer.py` detects which applies. `requirements.txt` now needs `transformers>=4.56` (for `dtype=`).
- ⚠️ **Not testable without a GPU:** the 4-bit/8-bit loading (bitsandbytes is CUDA-only), the paged optimizer, VRAM and energy measurement. These are validated by the 50-step M1 dry run on the GPU.

### 26. `src/evaluator.py` + run scripts ✅ (CPU smoke-tested)
- **`evaluator.py`** loads the base at `--bits` (4/8/16) and applies the adapter **unmerged** (how QLoRA models are served; M0 = no adapter, 16-bit). Greedy decoding (`do_sample=False`, 1 beam), stopping at `<|EOT|>`, left padding for batching.
  - **Phase 1, batched generation** (default batch 8; prompts length-sorted to cut padding, output order preserved) over the full test set → predictions JSONL with every field `metrics_se.py` / `execution.py` need; profiled for generation tokens/s, samples/s, peak inference VRAM and energy.
  - **Phase 2, latency probe** at **batch size 1** on the first 50 prompts → median and p90 TTFT, median decode tokens/s, single-request peak VRAM (RQ3: what one developer experiences). A short warm-up generation runs before measuring.
  - **`max_new_tokens`** from measured reference lengths (DeepSeek tokenizer): CodeXGLUE test max 194 (p99 182) → **256**; HumanEvalFix max 935 (Rust; Java 507, C++ 728, JS 545, Go 394, Python 296) → **1,024**. Each run records how many outputs hit the limit (`hit_max_new_tokens`).
  - `--score` scores CodeXGLUE inline (EM, CodeBLEU, syntax). HumanEvalFix is scored separately with `execution.py`, because it runs model-written code.
- **`metrics_se.extract_code`** now strips code fences with any language tag (it was Java-only).
- **Profiler:** `measure_generation` and `GPUProfiler` work without CUDA (CPU smoke tests). A **p90 bug** was caught in testing: the first version computed p90 with the wrong rounding and returned a value below the median for 2 samples. It now uses the nearest-rank method.
- **`scripts/run_matrix.sh`** trains M1–M9 (or selected cells) with the fixed bits/rank map, skips cells whose adapter exists (resumable), and `DRY_RUN=N` writes to `results/runs/dry_<cell>` and `results/dry_log.csv`. **`scripts/evaluate_all.sh`** runs CodeXGLUE (scored) + HumanEvalFix generation + test execution for M0–M9, skipping untrained models. Both need bash ≥ 4 (guarded).
- **Tests:** evaluator end to end on CPU with the tiny model (M0 on CodeXGLUE with scoring; adapter on HumanEvalFix → `execution.py` scores it: 0 %, as expected for a random model). Scripts tested in a `bash:5.2` container with a stub `python` that prints commands: correct bits/rank per cell, skip logic, dry-run paths, M0 without adapter; macOS bash 3.2 is refused with a clear message.
- **Pipeline status:** data → training → inference → metrics → test harness → run scripts are all implemented. Only the analysis scripts (Pareto plots, LaTeX tables, statistical tests) remain, and they need real results first.

### 27. Decisions: 3 seeds for the r = 16 column; M0 zero-shot ✅
- **User decision:** "go with 3 seeds and keep M0 zero-shot".
- **Seeds:** **M2, M5, M8** (the r = 16 column: 4-bit / 8-bit / 16-bit at fixed rank, isolating the main factor, bit-width) are trained with **seeds 42, 43, 44**. All other cells use 42. **The data split stays seed 42 for every run**, so all seeds see identical train/validation/test data. The training seed varies LoRA initialisation, dropout and data order (`seed` + `data_seed`). Inference is greedy, so it doesn't depend on the seed.
- **Cost:** +6 training runs (~2–3 GPU-h each → ~+12–18 GPU-h) and +6 evaluations; total roughly 85–120 GPU-h (~$100–145 on AWS g5.2xlarge), still inside the appeal's estimate range and the credit.
- **Statistics this enables (analysis phase):** mean ± SD over 3 seeds for M2/M5/M8; paired per-sample tests between bit-widths on the same test items (McNemar for EM, syntax validity and Pass@1; Wilcoxon signed-rank + Cliff's δ / Vargha–Delaney A12 for continuous scores), with seed variance as a reference for "meaningful" differences.
- **M0:** stays **zero-shot** (the plan's design; already implemented): the same instruction prompt as the fine-tuned models, no examples. It's framed as the untuned floor.
- **Scripts:** `SEEDS="43 44" bash scripts/run_matrix.sh M2 M5 M8` → `results/runs/<cell>_s<seed>`; seed 42 keeps the plain names (M2, M5, M8). `evaluate_all.sh` accepts `M2_s43` etc. and maps them to the cell's bit-width. Tested in a bash 5 container.

---

## 2026-10-21

### 28. Phase 0: artifact availability of the Tier-A papers ✅ (check done; scope ❓)
- **Method:** read each paper's PDF for artifact links, then opened the repositories (GitHub API) and the Hugging Face model card. Titles/venues confirmed from the PDFs (Huang et al.: TSE 51(4), 904–928, 2025, DOI 10.1109/TSE.2025.3532759).

| Paper | Artifact | What is released | Their setup | Reproducible on our 24 GB GPU? |
|---|---|---|---|---|
| **Li et al., ASE 2024**, PEFT on APR (arXiv 2406.05639) | ✅ `github.com/zjulgc/llmpeft4apr` (no licence file) + adapters on HF `survoltli/llmpeft4apr` | APR-INSTRUCTION (30k) + other instruction sets, train/inference/validation code, per-model train scripts, **generated patches + validation results**, **trained adapters (LoRA, IA3, prefix, p-tuning) for DeepSeek-Coder-6.7B-Base**, CodeLlama-7B/13B, Llama-2-7B | DeepSeek-Coder-6.7B-**Base**, bf16 (no quantization), LoRA r=32, α=16, dropout 0.05, lr 1e-4, 5 epochs, max_seq 1000, batch 3; benchmarks HumanEval-Java (163), Defects4J v2.0 (217), QuixBugs (40); metric pass@10 (bugs fixed); peak GPU memory reported. Pinned: torch 2.0.1, transformers 4.35.2, peft 0.6.2, Java 8 | **Yes.** Inference with their released DeepSeek LoRA adapter on HumanEval-Java + QuixBugs is cheap (~1–2 GPU-h). Retraining one config (DeepSeek LoRA, 30k × 5 epochs) is ~10–15 GPU-h. Defects4J needs the Defects4J framework (Java 8, Perl, large checkout) |
| **Huang et al., TSE 2025**, comprehensive fine-tuning of code LLMs for APR | ✅ `github.com/LLMC-APR/STUDY` (no licence file); experimental code + data on **Google Drive** | Fine-tuning/inference code per model (InCoder, CodeGeeX2, CodeGen25, CodeLlama-13B/70B, StarCoder/2, plus small models), training data (Transfer dataset), results | Billion-level models trained with **8-bit QLoRA, 1 epoch, A100 80 GB**; LoRA / AdaLoRA / IA3 vs full fine-tuning; test on Defects4J v1.2 + HumanEval-Java; **exact match** + test-based correct patches (manual check of plausible patches) | **Partly.** A ≤7B model (e.g. InCoder-6B or CodeGen25-7B) with LoRA in 8-bit fits 24 GB; CodeLlama-13B in 8-bit is borderline; 70B no. Drive folder size and training-set size not yet checked |
| **Wei et al., ESEC/FSE 2023**, greener code generation via quantization (arXiv 2303.05378) | ❌ none found (paper, ACM, Amazon Science, FSE page, GitHub search) | — | CodeGen 350M–16B, InCoder, CodeT5; **post-training int8** (PyTorch dynamic and static quantization); HumanEval + MBPP pass@1/5; AWS p3dn.24xlarge + MacBook M1; latency, model size, carbon estimate | **Re-implementation only.** The method is standard PyTorch PTQ and fully described, so a small re-implementation (e.g. CodeGen-2B, int8 dynamic, HumanEval pass@1) is possible, but it would not be a reproduction of their artifact. A third-party replication exists (arXiv 2503.07103) and may be a better comparison point |

- ⚠️ **Direct relevance of Li et al.:** they fine-tuned **DeepSeek-Coder-6.7B** (Base, not Instruct) with LoRA **r = 32** in bf16, which is the nearest published neighbour of our **M9** (16-bit, r = 32). They also report "rank 16 is already enough", which our rank axis (r = 8/16/32) tests directly. Differences to state in the thesis: Base vs Instruct, α = 16 vs our α = 2r, their instruction data vs CodeXGLUE, pass@10 sampling vs our greedy Pass@1.
- ⚠️ **Licences:** neither GitHub repo has a licence file. Running their code for reproduction is fine; copying their code or data into our public repo is not. We download at run time and cite.
- **Proposed Phase 0 scope (for user decision):**
  1. **Li et al. (primary):** (a) load their released DeepSeek-Coder-6.7B LoRA adapter and re-run their HumanEval-Java + QuixBugs validation; target their reported bugs-fixed counts. (b) retrain the same config once with their script to check the training side. Defects4J optional (setup cost).
  2. **Huang et al. (secondary):** reproduce one LoRA cell on a ≤7B model on HumanEval-Java after checking the Drive data size.
  3. **Wei et al.:** no artifact, so it cannot be reproduced. Either re-implement one int8 PTQ data point, or cite it (plus the 2025 replication) as related work only.
- ❓ Waiting on the user to pick the scope.

### 29. Novelty check and positioning table ✅
- **Question (user):** is it a problem that our study is close to Li et al.? **Answer:** no. Close prior work is expected and helps credibility, as long as the difference is stated explicitly. The user asked for a targeted search for any study that already does bit-width × rank for APR, plus a positioning table.
- **Search (2026-10-21):** web searches combining QLoRA / quantization / bit-width / LoRA rank / energy with program repair, HumanEvalFix and DeepSeek-Coder; abstracts and full texts read for every close hit.
- **Result:** **no study found that varies training-time quantization bit-width together with LoRA rank for program repair, or that measures training and inference energy for quantized PEFT on repair.** The closest work is below.

| Study | Task | Model(s) | Fine-tuning | Quantization | Rank varied? | Resources measured | Gap relative to ours |
|---|---|---|---|---|---|---|---|
| **Li et al., ASE 2024** (arXiv 2406.05639) | APR (Java) | CodeLlama-7B/13B, Llama-2-7B, **DeepSeek-Coder-6.7B-Base** | LoRA, IA3, prefix, p-tuning vs full FT | none (bf16) | small sweep (rank 16 "enough") | peak training memory | No quantization; no energy or latency |
| **Huang et al., TSE 2025** (10.1109/TSE.2025.3532759) | APR (Java) | InCoder, CodeGeeX2, CodeGen25, StarCoder(2), CodeLlama-13B/70B | LoRA, AdaLoRA, IA3 vs full FT | 8-bit used **as a tool** to fit models, not compared | no | memory and time | Bit-width not a factor; no energy |
| **Vallecillos-Ruiz et al., ICSME 2026** (arXiv 2606.27205) | APR (HumanEval-Java, Defects4J) | 6 models incl. **deepseek-coder-6.7b-instruct** | **none** (off-the-shelf models) | 13 post-training configs (AQLM, AWQ, bnb4, HQQ, Quanto; weights and KV cache) | n/a | inference time, energy (NVML), memory; Pareto | **Inference only, no fine-tuning.** Finds quantization cuts memory but **increases** inference time and energy |
| **Weyssow et al., TOSEM 2025** (10.1145/3714461) | code **generation** (Python) | CodeLlama-7B–34B, CodeGen2, CodeT5+ | LoRA, IA3, prompt, prefix, QLoRA | QLoRA 4-bit and 8-bit | **no** (fixed r = 16, rank sensitivity left as future work) | peak memory only | Not repair; no rank axis; no energy |
| **Machácek et al., ICSME 2025** (arXiv 2507.19909) | APR (QuixBugs, Defects4J, HumanEval-Java) | 6 code LLMs | none vs full FT vs LoRA/IA3 | none | no | — | No quantization or resources |
| **Wei et al., ESEC/FSE 2023** (arXiv 2303.05378) | code generation | CodeGen, InCoder, CodeT5 | none | post-training int8 (dynamic/static) | n/a | latency, size, carbon estimate | Not repair; no PEFT |
| **Yang et al., 2026, PAFT** (arXiv 2604.03113) | APR | DeepSeek-Coder-6.7B | new fine-tuning objective | 4-bit NF4 QLoRA **as a tool** | no | none | Bit-width not a factor |
| **Le-Cong et al., ICSE 2026, FLAMES** (arXiv 2410.16655) | APR | 1B–7B LLMs | none | discussed as an inadequate fix for memory | n/a | memory | Decoding method, not PEFT |
| **This thesis** | **APR**: CodeXGLUE Java + HumanEvalFix (6 languages) | DeepSeek-Coder-6.7B-Instruct | LoRA, 3 seeds on the r = 16 column | **training-time** NF4 4-bit vs LLM.int8() vs bf16 | **yes** (8/16/32) | training VRAM, time and energy; inference VRAM, TTFT, decode speed and energy | — |

- **Draft novelty statement (for the thesis introduction):** "Prior work compares PEFT methods for program repair at full precision (Li et al.; Huang et al.), uses quantization only to fit larger models, or quantizes off-the-shelf models at inference time (Vallecillos-Ruiz et al.). To our knowledge, this is the first study to vary training-time quantization bit-width jointly with LoRA rank for program repair and to measure the resulting trade-off across the full lifecycle: repair accuracy, training cost, and inference cost (memory, latency, energy)."
- ⚠️ **Vallecillos-Ruiz et al. (ICSME 2026) must be cited and discussed.** It is the closest work on the resource side and uses **the same model** (deepseek-coder-6.7b-instruct). Its finding that quantized inference is *slower and uses more energy* despite smaller memory is a hypothesis our RQ3 inference measurements (bnb NF4 / int8 vs bf16, A10G) can confirm or refute for **fine-tuned** models. It also reports pass@10 with temperature 0.1 on GH200 GPUs, so absolute numbers are not directly comparable to our greedy Pass@1 on A10G. Replication package: Zenodo 10.5281/zenodo.20847818.
- ⚠️ **Scope of the claim:** "to our knowledge" is bounded by this search (web search plus the papers above). Before submission, re-run the search and check the PEFT systematic review by Afrin et al. (arXiv 2504.21569, list of 28 included studies in its artifact) for anything missed.
- **Not added to the reproduction plan:** these papers are positioning only. Phase 0 scope is still decision #4.

### 30. Instruct model and greedy Pass@1: rationale ❓ (reopened for discussion, see #31)
- **User question:** why Instruct instead of Base, and why greedy only? Both come from the research plan (§4 model; greedy inference in the inference-memory metric and the `evaluator.py` description).
- **User's initial view on pass@10:** not added. "If we add pass@10 then our research will become very complicated and big." The user then asked for a deeper discussion before deciding (#31), so this is **not yet a decision**.
- **Justification for the methods chapter (Instruct):**
  1. The zero-shot M0 baseline is only meaningful for an instruction-following model. A Base model would fail on output format, not repair ability, and would inflate every fine-tuning gain.
  2. It matches the deployment scenario (adapting a code assistant on one GPU).
  3. HumanEvalFix is an instruction-style benchmark designed for instruct models.
  - *Threat to validity:* Li et al. (ASE 2024) used DeepSeek-Coder-6.7B-**Base**, so absolute numbers are not directly comparable. The Instruct model's chat habits may interact with CodeXGLUE's abstracted, space-tokenised format; fine-tuning reduces but cannot fully isolate this.
- **Justification for the methods chapter (greedy decoding):**
  1. Deterministic: differences between M0–M9 come from bit-width and rank, not sampling noise. This is required for the paired per-item tests (McNemar, Wilcoxon).
  2. Inference cost: sampled pass@10 would multiply evaluation compute about 10 times.
  3. Resource metrics (latency, VRAM, energy) depend on output length, and greedy keeps lengths fixed for a given model and input.
  4. ~~Convention for CodeXGLUE refinement~~ **Corrected in #32:** the official CodeXGLUE refinement baseline uses **beam search (beam size 5)**, not greedy. What remains of this point is the use case: a developer looks at one suggested fix.
  - *Threats to validity:* greedy Pass@1 is a lower bound on capability. It is not directly comparable to pass@10 with sampling (Li et al.; Vallecillos-Ruiz et al.) or to HumanEvalFix's original pass@1 (estimated from 20 samples at T = 0.2). Quantization can flip near-tied tokens, so single-item outcomes are noisy; seeds and paired tests address this at the aggregate level. Batched left-padding can shift greedy outputs slightly, so **batch size is fixed at 8 for every evaluation run** (`evaluate_all.sh` uses the evaluator default).
- **Base-model sensitivity check** (one Base run, about 3 GPU-h): offered, not adopted. The plan's Instruct model stays.

### 31. Literature check: DeepSeek-Coder-6.7B with Pass@1 (input to the greedy vs sampling discussion) ✅ (decision ❓)
- **User request:** check whether any paper uses DeepSeek-Coder-6.7B-Base or -Instruct with Pass@1. Protocols verified from full texts, not abstracts.

| Paper | Variant | Task / benchmark | Metric and decoding (quoted or verified) | Reported numbers |
|---|---|---|---|---|
| **DeepSeek-Coder tech report** (Guo et al., 2024, arXiv 2401.14196) | Base and Instruct 6.7B | generation: HumanEval, MBPP | Pass@1, **greedy** ("We adopted a greedy search approach") | HumanEval-Py: Base 49.4, Instruct 78.6; multilingual avg 44.7 / 66.1; MBPP 60.6 / 65.4 |
| **StarCoder2** (Lozhkov et al., 2024, arXiv 2402.19173), Table 12 | **Instruct 6.7B** | **repair: HumanEvalFix, 6 languages (our benchmark)** | Pass@1 **sampled**: "temperature of 0.2 to estimate pass@1 with 20 samples" (OctoPack protocol). The same paper uses greedy for HumanEval/MBPP | Py 44.9, JS 55.3, Java 52.2, Go 42.9, C++ 37.9, Rust 19.5; **avg 42.1** |
| **Magicoder** (Wei et al., ICML 2024, arXiv 2312.02120) | Base 6.7B (fine-tuned), Instruct as baseline | generation: HumanEval(+), MBPP(+) | Pass@1, **greedy** ("we use greedy decoding to generate one sample") | MagicoderS-DS 76.8 HumanEval |
| **Machácek et al., ICSME 2025** (arXiv 2507.19909) | Base 6.7B | repair: QuixBugs, HumanEval-Java, Defects4J | **10 patches per bug**, plausible if any passes (pass@10-style); decoding not stated; LoRA r = 8 / IA3, 1 epoch | No FT: 33/40, 107/163, 89/219; LoRA: 31/108/108 |
| **PAFT** (Yang et al., 2026, arXiv 2604.03113) | "DS-Coder-6.7B", variant not stated | repair: Defects4J, HumanEval-Java | Pass@1 **sampled at T = 0.7**; **4-bit NF4 QLoRA, r = 32**, α = 16 | D4J pass@1 5.9 → 10.1 |
| **Li et al., ASE 2024** (#28) | Base 6.7B | repair: HumanEval-Java, D4J, QuixBugs | pass@10 | (#28) |
| **Vallecillos-Ruiz et al., ICSME 2026** (#29) | Instruct 6.7B | repair: HumanEval-Java, D4J | pass@10, T = 0.1 | (#29) |

- **Findings:**
  1. **Greedy Pass@1 is the standard for DeepSeek-Coder on code *generation* benchmarks** (the official report, Magicoder/EvalPlus, StarCoder2's HumanEval section). ⚠️ This is **not** a precedent for our task, which is repair (see #32).
  2. **For HumanEvalFix, the standard is *sampled* pass@1** (n = 20, T = 0.2), not greedy and not pass@10. The only published HumanEvalFix numbers for our exact model (StarCoder2 Table 12, avg 42.1) use that protocol.
  3. **Repair papers on Java benchmarks mostly use pass@10 / plausible-patch counts** (Li; Machácek; Vallecillos-Ruiz). Only PAFT uses pass@1, and it samples.
  4. No paper found that uses DeepSeek-Coder-6.7B with greedy Pass@1 on a repair benchmark.
- ⚠️ **Implication: an external anchor for M0.** M0 (untuned Instruct, zero-shot, HumanEvalFix) can be checked against StarCoder2's 42.1 average and per-language numbers. This validates our harness, the way Phase 0 is meant to. Comparability caveats:
  - greedy vs T = 0.2 sampled pass@1 (usually close, not identical);
  - prompt wording: our `HEF_INSTRUCTION` vs the OctoPack/StarCoder2 instruct prompt;
  - our 4 defective tasks are excluded in the clean subset;
  - max_new_tokens.
- **If sampling is adopted** (open discussion from #30), the literature points to **sampled pass@1 with n = 20, T = 0.2 on HumanEvalFix** (comparable to StarCoder2/OctoPack), not pass@10 (that is the Java-repair convention on HumanEval-Java/Defects4J, which we don't use).
- ❓ Still open: greedy only vs greedy + sampled subset, and whether to run M0 once with the OctoPack protocol as a harness check.

### 32. Correction: greedy decoding has no precedent on our repair benchmarks ⚠️ (decision ❓)
- **User challenge:** "Greedy Pass@1 is standard for DeepSeek-Coder on code generation, but we have code repairing, right?" Correct. The generation precedent does not carry over to repair. I had used it as support for greedy decoding.
- **Verified conventions on the two benchmarks we actually use:**

| Benchmark | Established protocol | Source |
|---|---|---|
| CodeXGLUE code refinement (medium) | **Beam search, beam size 5**; EM (primary), BLEU, CodeBLEU | official CodeXGLUE baseline scripts (`--beam_size 5`, microsoft/CodeXGLUE, Code-Code/code-refinement) |
| HumanEvalFix | **Sampled pass@1, n = 20, T = 0.2** | OctoPack protocol, used by StarCoder2 Table 12 (our exact model: avg 42.1) |
| (not ours) HumanEval-Java / Defects4J / QuixBugs | pass@10 / plausible-patch counts | Li et al.; Machácek et al.; Vallecillos-Ruiz et al. |

- **Consequence:** greedy decoding departs from the established protocol on **both** of our benchmarks. Its justification must rest on **methodological grounds only**:
  - determinism for paired per-item tests across M0–M9;
  - one fixed decoding configuration for the latency, VRAM and energy measurements;
  - inference cost.

  Comparability with published numbers would then be lost on both benchmarks and must be stated as a threat to validity.
- **Correction to entry 30, point 4:** the claim "greedy is the convention for CodeXGLUE refinement" was wrong; it is beam search (5).
- **Note:** beam search is deterministic too, so beam 5 on CodeXGLUE would keep the paired tests valid while matching the benchmark's convention. Its cost: higher KV-cache memory and slower decoding (both affect the RQ3 inference measurements, so those would need one fixed, declared decoding configuration).
- ❓ Options for discussion with the user (no decision yet):
  - **A.** Greedy everywhere, as a declared deviation from both benchmarks' conventions.
  - **B.** Each benchmark's own convention: beam 5 on CodeXGLUE, sampled pass@1 (n = 20, T = 0.2) on HumanEvalFix.
  - **C.** Greedy as the primary protocol for all models, plus the benchmark conventions on a subset (M0 + M2/M5/M8).

### 33. Briefing document for the decoding-protocol decision ✅
- The user found #30–#32 confusing and asked for a single self-contained summary to discuss in NotebookLM. Written as `docs/decoding_protocol_briefing.md` (renamed to `docs/methodology_briefing.md` in #35). It covers the study, plain-language definitions (greedy, beam, sampling, Pass@1 vs pass@10), the verified literature, the pros and cons of greedy, options A/B/C, the remaining uncertainties, questions to explore, and source links. The decision remains ❓.

### 34. Discussion: training on Java (CodeXGLUE) but evaluating on 6 languages (HumanEvalFix) ❓
- **User question:** if we train only on CodeXGLUE Java, how can we evaluate on other languages?
- **What HumanEvalFix actually measures in this design:** **transfer / forgetting, not in-task accuracy.** The base model already repairs code in all six languages; StarCoder2 reports untuned DeepSeek-Coder-6.7B-Instruct at 42.1 avg pass@1 (#31). The adapters change that ability. HumanEvalFix measures whether the change helps, does nothing, or harms (catastrophic forgetting), and whether **bit-width and rank change how much**. This was the intent of RQ1b in #20.
- ⚠️ **Even HumanEvalFix-Java is out-of-distribution.** Training data is abstracted (`VAR_1`, `METHOD_1`), space-tokenised, single methods, ≤ 512 tokens, with one fixed instruction. HumanEvalFix has real identifiers, whole programs with imports, longer inputs (Java mean 582 tokens, 102/164 over 512, #20), example tests in the prompt, and a different instruction. So the design separates two shifts:
  1. **format shift only:** CodeXGLUE Java → HumanEvalFix Java;
  2. **format + language shift:** → Python, C++, Go, Rust, JS.
- ⚠️ **Framing problem found:** #20 said HumanEvalFix Pass@1 "replaces the Pass@k promise in RQ1". That conflates two things: in-task correctness and generalisation. Proposed framing:
  - **RQ1** (in-task repair quality) = CodeXGLUE EM / CodeBLEU / syntax validity;
  - **RQ1b** (generalisation and forgetting) = HumanEvalFix Pass@1 relative to M0.

  Consequence: the thesis has **no test-based functional correctness on the training distribution** (CodeXGLUE has no tests). This must be stated as a limitation.
- **Plausible outcome:** fine-tuned models score **below M0** on HumanEvalFix. Prior evidence: Machácek et al. (ICSME 2025) found fine-tuning lowered repair scores, attributed to differences between training data and benchmarks (#28/#31). This would be a legitimate finding; the interesting part is whether 4-bit vs 16-bit or r = 8 vs 32 forget more.
- ⚠️ **Floor-effect risk:** if every adapter collapses on HumanEvalFix (e.g. emits abstracted `VAR_1` tokens or truncated code), all models score near 0 and bit-width/rank cannot be compared. **Mitigation:** right after the dry run, evaluate M0 and the first adapter on a HumanEvalFix sample and inspect the outcome categories (compile error vs wrong logic vs abstracted tokens) before running the full matrix.
- **Alternatives considered:**
  - **(a)** Keep the design and reframe as above.
  - **(b)** Add multilingual repair training data (e.g. CommitPackFT, used by OctoPack): a different study, much bigger.
  - **(c)** Find a Java repair test set with unit tests matching the training format: none known with abstracted identifiers, so it would still be OOD.
- ❓ Awaiting user discussion. The assistant's recommendation is (a).

### 35. Briefing extended with Problem 2 ✅
- At the user's request, the cross-language evaluation question (#34) was added to the briefing as a second problem. The file was renamed `docs/decoding_protocol_briefing.md` → **`docs/methodology_briefing.md`**.
- New content:
  - Problem 2: why the evaluation is possible, transfer vs forgetting, the format-shift vs language-shift split, the RQ1/RQ1b reframing, the two risks (below-M0 scores, floor effect), and options a/b/c.
  - A section on how the two problems connect: under the reframing, published comparability matters mainly for M0, so only a small extra protocol run is needed.
  - Questions 6–11, and Biderman et al. (TMLR 2024, "LoRA Learns Less and Forgets Less", arXiv 2405.09673, verified) added as a source.
- Both decisions remain ❓.

### 36. Is CodeXGLUE the wrong training dataset? Candidate replacements ❓
- **Context:** after #30–#35 the user proposed restarting the research. The assistant argued against a full restart: the novelty is confirmed (#29) and the pipeline is validated. The user identified the real concern as **"CodeXGLUE is the wrong dataset"** (no unit tests; abstracted `VAR_1` style unlike real assistant use; root cause of Problems 1 and 2).
- **Key point:** the *training* data does not need unit tests. The *evaluation* needs tests, in a format close to the training data.
- **Candidates checked (2026-10-21):**

| Candidate | What it is | Tests? | Fit | Risks |
|---|---|---|---|---|
| **Keep CodeXGLUE** (status quo) | 52k abstracted Java method pairs | No | Pipeline done | Problems 1 and 2 stay; limitation "no test-based in-task correctness" |
| **Megadiff single-function** (HF `ASSERT-KTH/megadiff-single-function`; RepairLLaMA's training source) | 72.4k real Java single-function changes from open-source commits; RepairLLaMA filtered to < 1,024 tokens and deduplicated → 30–50k | No (training only) | Real code, function-level, like HumanEvalFix-Java / HumanEval-Java. HumanEvalFix-Java becomes a near in-task test-based check; other 5 languages = cross-language transfer. Matches the APR literature (RepairLLaMA, Li, Machácek, Vallecillos-Ruiz eval on HumanEval-Java) | Commits include non-bug-fix changes (noise); licence not stated on the card; longer functions (≤ 1,024 tokens) would raise training memory and time vs our 512, or filter to ≤ 512 and accept a length bias |
| **RunBugRun** (Prenner & Robbes, arXiv 2304.01102) | Competitive-programming buggy/fixed pairs from CodeNet etc., 8–9 languages, ~450k (v1; Java ≈ 17.6k) | **Yes**, stdin/stdout tests per problem, train/valid/test splits | The only option with train and test from the **same distribution with tests** | README: "revised and extended version" (v1 in a legacy branch, splits changed); no licence stated; SQLite dump + Bubblewrap sandbox; competitive programs, not real-project repair |
| **xCodeEval APR** (Khan et al., ACL 2024) | Codeforces-based, multilingual, ExecEval engine | Yes | Tests in many languages | Licence ambiguous (card says both cc-by-4.0 and cc-by-nc-4.0); APR split details not on the card; Docker execution engine; competitive programming |
| **FixEval** (Haque et al., arXiv 2206.07796) | AtCoder Java + Python buggy/fixed with tests | Yes | Java + tests | Smaller and older; competitive programming |

- **Assistant's initial assessment (not a decision):** **Megadiff → HumanEvalFix (+ HumanEval-Java)** fits the thesis best:
  - real Java bug-fix commits for training;
  - unit-test evaluation on function-level Java in the same format;
  - comparability with the APR papers already cited;
  - it reuses everything except the data loader and the in-task similarity metrics.

  RunBugRun is the strongest *methodologically* (same-distribution tests) but carries version, licence and infrastructure risk, and competitive programs are less realistic.
- **Proposed next step:** a CPU-only feasibility check before deciding:
  - Megadiff: licence, token-length distribution under the DeepSeek tokenizer, duplicate/near-duplicate rate, overlap with HumanEval;
  - HumanEval-Java: format and how its tests run.
- ❓ Awaiting the user.

### 37. Discussion: user proposal "Megadiff + HumanEvalFix-Java only + pass@10" ❓
- **User proposal:** train on Megadiff, evaluate only on HumanEvalFix-Java, so the study is Java-only and pass@10 becomes affordable.
- **Assistant analysis:**
  - **Strengths:**
    - Coherent Java-only story: removes Problem 2.
    - Sampling becomes cheap (164 tasks instead of 984), so Problem 1 is largely settled.
    - pass@10 matches the Java APR literature (Li; Machácek; Vallecillos-Ruiz).
  - **Concerns:**
    1. **pass@10 is less sensitive to quantization damage** (ceiling compression: untuned Instruct is already 52.2 pass@1 on HumanEvalFix-Java per StarCoder2). Proposal: draw n samples once (e.g. n = 20, T = 0.2) and report **pass@1 (primary) and pass@10 (secondary) from the same samples**, at no extra cost (Codex unbiased estimator).
    2. **Statistical power with 164 problems:** paired differences of a few points will be hard to detect. Equivalence margin likely ±5 pp, not ±3.
    3. **Format gap remains, smaller:** Megadiff is real-project code; HumanEvalFix problems are small algorithmic functions. Defects4J would match better but is too heavy with 10+ samples × 16 models.
    4. **Option to add HumanEval-Java** (Jiang et al.): 164 more test-based Java bugs on the same problems with different bugs. Gives direct comparability with Li / Machácek / Vallecillos-Ruiz and more power (correlated, not independent). Costs a second Java harness.
    5. **Dropping the other 5 languages is a scope reduction** (relevant to the no-cuts rule); the user's call.
    6. **Decoding for RQ3:** keep the existing greedy batch-size-1 probe for latency, VRAM and energy, separate from the sampled accuracy runs.
    7. **In-task similarity:** EM / CodeBLEU / syntax on a held-out Megadiff test split replaces CodeXGLUE's.
- ❓ Awaiting the user. Feasibility check (#36) still pending.

### 38. Feasibility check: Megadiff (training) + HumanEval-Java (evaluation) ✅
- **User decisions (#37):** pass@1 primary + pass@10 secondary, from the same samples ✅; **add HumanEval-Java** ✅; run the feasibility check ✅. The Megadiff switch itself is still pending, on these results.
- **Scripts:** `analysis/feasibility/megadiff_feasibility.py`, `analysis/feasibility/hej_sanity.py` (CPU only, local Mac).

#### Megadiff single-function (`ASSERT-KTH/megadiff-single-function`, 72,393 rows, 546 MB download)
- **Licence:** **none stated**, on either the GitHub repo or the HF card. The underlying code comes from open-source projects under their own licences. → Research use with citation (Monperrus et al., arXiv 2108.04631); we don't redistribute it (downloaded at run time; `data/` is gitignored).
- **Content:**
  - all rows are single-function; 70% single-hunk;
  - changed lines: median 2, p95 14, max 40; 81% change ≤ 6 lines (small, bug-fix-sized edits).
- **Trivial changes:** 0 identical, 489 whitespace-only, 1,614 comment-only → dropped.
- **Duplicates:** 5,608 extra copies of identical (buggy, fixed) pairs → deduplicated.
- **Clean pool:** **64,834** pairs.
- **Noise spot-check** (12 random clean diffs, assistant's judgement, indicative only):
  - ~8–9 look like real fixes: overflow `700L*…`, bounds `hasRemaining()`, reversed `isAssignableFrom`, exception handling, increment errors;
  - ~3 do not: parameter rename, log-level change, message edit.
  - → Megadiff is "code changes", not curated bug fixes. Roughly a quarter may be non-fix changes; state as a limitation.
- ⚠️ **Lengths (DeepSeek tokenizer, chat-formatted prompt + target + EOS):** mean 2,533, **median 1,236**, p95 7,510 tokens. Megadiff functions (median ~590 tokens) are much longer than HumanEvalFix-Java functions (median 130, p95 281).

| Token cap | Clean pairs | Mean tokens | Note |
|---|---|---|---|
| ≤ 512 | **10,398** | 356 | Same compute as the CodeXGLUE plan (mean 369); closest to eval function sizes; too few for 10k/1k/1k |
| ≤ 768 | 19,616 | 489 | 10k/1k/1k possible; ~1.35× training tokens; longer max_length (more activation memory) |
| ≤ 1,024 | 27,283 | 602 | RepairLLaMA-like; ~1.6× tokens |

- **Contamination vs HumanEvalFix-Java:**
  - 58 Megadiff rows share a 20-token sequence with a HumanEvalFix-Java fix (21 tasks). Inspected matches are **generic idioms** (`for ( int i = 0 ; i < lst . size ( ) ; i + + ) {`, a `Comparator<String>` boilerplate), not leaked solutions.
  - 32/157 function names also occur, but they are generic (`add`, `compare`, `median`).
  - → No evidence of leakage. Conservative rule: drop the 58 rows anyway.
- **Split leakage risk:** the same project can appear in train and in the held-out Megadiff test split (no repository field; only file paths in `diff`). This would inflate in-task EM/CodeBLEU, not the test-based HumanEval metrics. Mitigation: split by file path, or treat Megadiff-test EM as secondary.

#### HumanEval-Java (`lin-tan/clm`, Jiang et al., ICSE 2023)
- **Licence:** BSD-3-Clause.
- **Format:** 163 single-function bugs (bugs injected by mutation operators). Each has a whole buggy class (`src/main/java/humaneval/buggy`), a correct class (`…/correct`), and JUnit 4 tests (`src/test/java/humaneval/TEST_*.java`, 3 s timeout per test). JUnit/Hamcrest jars are bundled → plain `javac` + `JUnitCore`, no Maven.
- **Requires JDK 8:** `DO_ALGEBRA` uses the Nashorn JavaScript engine, `STRING_TO_MD5` uses `javax.xml.bind` (both removed in newer JDKs). The GPU VM needs `openjdk-8-jdk` in addition to the default JDK used for HumanEvalFix.
- **Sanity (JDK 8):** correct 162/163 pass, buggy 162/163 fail, after handling:
  1. **GET_ROW:** the buggy and correct files have **swapped package declarations** (`buggy/GET_ROW.java` says `package humaneval.correct`). Fixed by normalising the package line.
  2. **ISCUBE:** the buggy file is **identical** to the correct one (no bug) → exclude (like CPP/23 in HumanEvalFix).
  3. **DO_ALGEBRA:** passes alone (3/3 runs), fails under 8-way parallel load (3 s JUnit timeout plus Nashorn start-up) → run serially or flag as flaky.
  4. The `fix` strings in `humaneval-java-sf.json` match the correct files; whole-file versions are used to avoid line-range errors.
  - → **Clean set: 162 bugs.** With HumanEvalFix-Java's 164: **326 test-based Java bugs** (same underlying problems, different bugs; correlated).
- **javac quirk:** JDK 8 `javac -d` doesn't create the output directory; the harness must create it.

#### Updated cost (estimates until the dry run)
- **Training:** at a ≤ 512 cap, the same as the CodeXGLUE estimate (mean 356 vs 369 tokens); at ≤ 768, ~1.35×.
- **Evaluation per model:** 326 bugs × 20 samples = 6,520 sampled generations, plus greedy EM on a Megadiff test split, plus the greedy latency probe. Roughly 2.5–3.5 GPU-h per model → ~40–55 GPU-h for 16 models (halve with n = 10).
- **CPU:** ~104k Java test runs in total, ~5 h on 8 cores.
- **Overall:** roughly 110–150 GPU-h (~$135–180), within the $700 credit.
- ❓ **For the user:** (1) switch to Megadiff, yes or no; (2) token cap ≤ 512 vs ≤ 768; (3) the split rule.

### 39. Decision: switch to Megadiff (≤ 768 tokens, split by file path); pipeline rebuilt ✅
- **User decisions (2026-10-21), on the #38 results:**
  1. **Switch the training data from CodeXGLUE to Megadiff** ✅
  2. **Token cap ≤ 768** (keeps 10k/1k/1k; ~1.35× training tokens vs ≤ 512) ✅
  3. **Split by file path** ✅
  - Together with #37: evaluation is **Java only** (HumanEvalFix-Java + HumanEval-Java); **pass@1 primary, pass@10 secondary**, both from n = 20 samples at T = 0.2.
- **Data (`src/data_loader.py`, rebuilt):**
  - Cleaning: drop identical / whitespace-only / comment-only changes (2,101), duplicate pairs (5,458), pairs sharing a 20-token sequence with any HumanEvalFix-Java or HumanEval-Java method, buggy or fixed (**84**), and pairs over 768 tokens (44,644). Methods are dedented before use.
  - Pool: **20,106** pairs in 18,780 file paths.
  - Split: paths shuffled (seed 42), then whole path groups fill test, validation, then train; a group that overshoots a target is trimmed by random sampling, and the trimmed rows are discarded (never moved to another split). The loader asserts that no path occurs in two splits.

| Split | Pairs | Mean tokens | Median | p95 | Max |
|---|---|---|---|---|---|
| train | 10,000 | 485 | 492 | 740 | 768 |
| validation | 1,000 | 500 | 520 | 741 | 768 |
| test | 1,000 | 484 | 492 | 740 | 768 |

  - ⚠️ **What the path split does and doesn't prevent:** most paths have one change (20,106 pairs / 18,780 paths). It keeps repeated edits of one file and forks of the same file in one split. It can't keep a whole *project* in one split: Megadiff has no repository field. In-task EM may still be slightly optimistic; the test-based metrics are unaffected.
  - Prompt: `The following Java method contains a bug. Return only the corrected Java method, with no explanation.` + the buggy method. (The CodeXGLUE wording about abstracted identifiers is gone.)
- **HumanEval-Java (`load_humaneval_java`):** the prompt is the buggy method alone, in the Megadiff training format; the harness splices the predicted method into lines begin..end of the buggy class (a prediction containing the whole class replaces the file). The line ranges in `humaneval-java-sf.json` match the files except for tabs in MAKE_A_PILE and WORDS_STRINGS (checked with tabs expanded).
- **Harness (`src/execution.py`):** HumanEval-Java runner (JDK 8 via `HEJ_JAVA_HOME`, JUnit jars via `HEJ_LIB`); unbiased **pass@k** (Chen et al., 2021); thread-pool scoring (`--workers`); DO_ALGEBRA runs serially after the pool; ISCUBE is in `DEFECTIVE_TASKS`.
  - **Sanity (macOS, JDK 8 for HumanEval-Java):** references pass **164/164** (HumanEvalFix-Java) and **163/163** (HumanEval-Java, spliced methods); buggy code passes **0/164** and **1/163** (= ISCUBE) → **clean sets 100% / 0%**.
  - pass@k unit checks: n = 20, c = 1 → pass@1 = 0.05, pass@10 = 0.5; c = 0 → 0; c = 20 → 1.
- **Evaluator (`src/evaluator.py`):** benchmarks `megadiff` (greedy, 1 sample, scored inline), `humanevalfix` (Java by default) and `humaneval_java` (20 samples, T = 0.2, top-p 0.95). `--batch_size` counts sequences (default 20 = one prompt × 20 samples). The RNG is re-seeded before generation, so every model sees the same sampling seed. The latency probe stays **greedy, batch size 1**. Output rows carry a `predictions` list.
  - max_new_tokens: 512 (Megadiff; longest test target 375 tokens), 512 (HumanEval-Java; 383), 768 (HumanEvalFix-Java whole class; 506).
- **Metrics (`src/metrics_se.py`):** the CodeXGLUE de-tokenisation rules (#9) are removed; real Java needs none and they could alter real code. Ground-truth syntax validity on the Megadiff test split: **99.8%** (2/1,000 fail to parse with tree-sitter-java 0.21). **Naive copy: 0.876 CodeBLEU, 0% EM** (n-gram 0.875, weighted n-gram 0.878, syntax-tree 0.887, data-flow 0.862; n = 1,000). On CodeXGLUE it was 0.887 (#9c), so the trap holds on real code too.
- **Trainer:** default data prefix `megadiff`, `max_length` 768 (0 truncated by construction).
- **Tests:** CPU smoke tests with the tiny 2-layer model: trainer on Megadiff (2 steps, initial loss 10.40 ≈ ln 32k); evaluator on Megadiff (greedy, `--score`) and HumanEval-Java (4 samples per prompt, 2 prompts per call) → `execution.py` scores it (0%, as expected for a random model). `scripts/evaluate_all.sh` updated (Megadiff + both test-based benchmarks, refuses to run without `HEJ_JAVA_HOME`); `bash -n` passes.
- **Superseded:** the CodeXGLUE-specific findings (#5–#10: split sizes, length 512, de-tokenisation, naive copy 0.887) stay in this log as history; they no longer describe the pipeline.
- ❓ **Still to discuss with the user before the GPU runs** (implemented with a default, not decided):
  1. **HumanEval-Java prompt has no specification.** Only the buggy method is shown (training format, as in Jiang et al.), not the problem description above the class. Some bugs may be unidentifiable without it. Alternative: prepend the description as a comment.
  2. **HumanEvalFix-Java keeps its OctoPack-style prompt** (whole class + example tests), which differs from the training format (method only). Pro: comparable with published numbers (StarCoder2 report: 52.2 for DeepSeek-Coder-6.7B-Instruct). Con: a format shift between training and test.
  3. **Megadiff in-task EM uses greedy decoding** (1 sample). Beam search (CodeXGLUE convention) is not used.
  4. **Batch size / memory at max_length 768** and the sampling batch: to be confirmed by the GPU dry run.

### 40. Decisions: HumanEval-Java prompt gets a problem description; HumanEvalFix-Java keeps its standard prompt ✅
- **User decisions (2026-10-21), items 1–2 at the end of #39:**
  1. **Add a problem description to the HumanEval-Java prompt** ✅ (without one, bugs like `x | y` vs `x + y` can only be guessed from the method name).
  2. **Keep the standard OctoPack-style HumanEvalFix-Java prompt** (whole class + example tests) ✅, so the untuned M0 can be checked against the published **52.2** pass@1 (StarCoder2 report, Table 12: DeepSeek-Coder-6.7B-Instruct, n = 20, T = 0.2).
- **Where the descriptions come from:**
  - **90 tasks:** the comment above the class in the Java file. The clm authors adapted some of these to the Java tests: ADD_ELEMENTS adds "the negative sign counts as a digit"; ANY_INT says the numbers are doubles that must convert to integers. Of the 90, 47 equal the HumanEval original (whitespace-normalised) and 43 differ.
  - **73 tasks:** no comment in the file, so the original HumanEval docstring is used (`HumanEval.jsonl`, shipped in the clm folder).
  - **Mapping:** line i of `humaneval_loc.txt` is HumanEval/i. Nine tasks were renamed (e.g. `f` → FACTORIAL, `maximum` → MAXIMUM_K). CORRECT_BRACKETING occurs twice (HumanEval/56 and /61); the Java task is /56 (its comment is the `<>` version), and /61 has no Java task, which is why there are 163 tasks, not 164.
  - **Format:** `/* description */` directly above the buggy method, after the unchanged Megadiff instruction. The field `description_source` records `java_file` or `humaneval`.
- **Known caveats (for the thesis):**
  - Descriptions keep Python-style examples (`>>> add(2, 3)`, `True`, `[1, 2]`). The 90 Java-file comments use the same style.
  - In the 73 HumanEval-sourced descriptions of renamed tasks, the function name can differ from the Java method (e.g. "Implement the function f" above `factorial`).
  - Megadiff training methods never carry such a comment, a small format shift.
  - Jiang et al. and Li et al. gave no description, so our HumanEval-Java numbers are **not directly comparable** with theirs.
- **Checks:**
  - Megadiff splits are **byte-identical** after the rebuild (md5); the contamination check still uses the method only.
  - HumanEval-Java prompts now average 475 tokens (max 1,122), up from 347.
  - Harness sanity re-run: reference fixes pass 100%, buggy code 0% on the clean set.
- **Still open:** item 3 of #39 (→ decided in #42) and item 4 (GPU dry run, blocked on the AWS quota, see #41).

### 41. AWS quota appeal: no answer yet ⏳
- 2026-10-21: G/VT quotas still 0 (On-Demand and Spot; eu-central-1 and us-east-1). The support case shows no AWS reply since the appeal (2026-10-20 17:37 CEST).
- User decision: **wait for AWS**; RunPod stays backlog. The GPU dry run (batch size and memory at max_length 768) waits for the approval.
- 2026-10-22 ~10:30 (after the user re-authenticated; both CLI and console sessions had expired): quota still **0**, request still open, **no AWS reply** since the appeal (~41 h). Platform decision point: 2026-10-24.

### 42. Decision: greedy decoding for Megadiff Exact Match ✅
- **User decision (2026-10-21), item 3 of #39:** the Megadiff test split is decoded **greedily (1 answer per bug)** and scored with Exact Match, CodeBLEU and syntax validity.
- **Why pass@k is not possible there:** a Megadiff pair is only the method before and after a commit. There is no project to compile against and no tests, so there's nothing to count passes with; pass@k needs an executable check.
- **Why greedy, not beam search or "EM in any of 10":** EM grades one answer per bug, as it's reported in the literature. Taking the best of several samples would inflate lucky textual matches, while correct fixes that are written differently still count as wrong. Beam 5 (the CodeXGLUE convention) was tied to the old dataset and costs ~5× the decoding.
- **RQ1 metrics, final:** pass@1 (primary) and pass@10 on HumanEvalFix-Java + HumanEval-Java (n = 20, T = 0.2, top-p 0.95); EM / CodeBLEU / syntax on the Megadiff test split (greedy) as an in-task check.
- **No code change:** `evaluator.py` already decodes Megadiff greedily (`N_SAMPLES["megadiff"] = 1`).

### 43. Methodology briefing removed ✅
- `docs/methodology_briefing.md` (#33, #35) was written to discuss the open decoding and dataset questions. All of them are now decided (#37–#42), so at the user's request it was deleted. It remains in the git history.

### 44. RQ wording updated in the research plan ✅
- The research plan's RQs (§3) now match the decisions in #37–#42; the README table is in sync.
  - **RQ1:** names both factors (bit-width **and** rank) and the metrics: pass@1 (primary) and pass@10 on HumanEvalFix-Java + HumanEval-Java, and EM / CodeBLEU / syntax on held-out Megadiff fixes, relative to the 16-bit adapters and M0. Replaces "AST compliance, CodeBLEU and patch test pass rates".
  - **RQ2:** across bit-widths and ranks; "high-fidelity repair" → "test-verified repair quality".
  - **RQ3:** "adapter target modules" → **adapter rank**: target modules are fixed (all linear layers), so they were never a factor. Energy (Wh) added, since it is measured.
  - **RQ4:** the Pareto trade-off is defined as resource cost (VRAM, time, energy) vs. pass@1, instead of "functional code validity".
- The rest of the plan (§4 benchmark: CodeXGLUE; §5 sequence length 512; §6A metrics; §7–§8 file names and schedule) is still the original text; see the README and this log for the current design.

### 45. Research plan brought in line with the current design ✅
- At the user's request, the rest of the plan (§1, §4–§8) now matches #37–#44:
  - §4: Megadiff (cleaning, ≤ 768 tokens, file-path split, 10k/1k/1k) + HumanEvalFix-Java + HumanEval-Java; hardware AWS A10G with the one-GPU-model rule.
  - §5: actual fixed hyperparameters (paged AdamW 32-bit, cosine + 3% warm-up, max length 768, clip 0.3, α = 2r) and the 3-seed design (16 evaluated models).
  - §6: pass@1 / pass@10 (n = 20, T = 0.2) added as the primary RQ1 metrics; EM / CodeBLEU / syntax on Megadiff with greedy decoding and the naive-copy reference; latency defined as greedy batch-size-1.
  - §7: the real repository layout.
  - §8: schedule re-based. Week 1 ✅; week 2 = methodology freeze ✅ + GPU dry run (blocked on AWS); training in week 3, evaluation in week 4. The target finish (late November) is unchanged. Budget: ~110–150 GPU-h.
- The CodeXGLUE / Defects4J / RunPod-Colab wording is gone from the plan. #44's note that "the rest of the plan is still the original text" no longer applies.

### 46. GPU dry run on Modal (A10 / A10G, 24 GB) ✅ with findings ⚠️
- **Why Modal:** the AWS quota appeal (#41) and a second provider's limit request are both pending; Modal runs serverless GPU jobs from the free monthly credit with no quota request. New: `scripts/modal_app.py` (repo + pinned requirements + JDK 8 + HumanEval-Java in a container; Hugging Face cache and `results/` on Modal volumes), `scripts/dry_run.sh`, `scripts/dry_run_batch.sh`. Total dry-run cost ≈ $3 of credit.
- **Training, 50 optimizer steps each** (max_length 768, batch 4 × grad-accum 4, gradient checkpointing, 0 truncated):

| Cell | GPU | Peak VRAM alloc / reserved (GiB) | s / step | samples/s | Energy (Wh, 50 steps + val) | Projected full run (1,875 steps) |
|---|---|---|---|---|---|---|
| M1 4-bit r8 | A10 | 7.32 / 9.73 | 14.6 | 0.81 | 40.9 | ~7.6 h |
| M3 4-bit r32 | A10 | 7.66 / 10.75 | 14.6 | 0.81 | 41.0 | ~7.6 h |
| M6 8-bit r32 | A10G | 10.59 / 12.84 | 17.8 | 0.71 | 72.5 | ~9.3 h |
| M9 16-bit r32 | A10G | 15.10 / 18.15 | 13.1 | 0.91 | 54.0 | ~6.8 h |
| M1, batch 8 × 2 | A10G | 9.98 / 14.45 | 16.4 | 0.72 | 64.8 | ~8.6 h |
| M9, batch 8 × 2 | A10G | 16.99 / 21.12 | 14.0 | 0.86 | 56.1 | ~7.3 h |

  - **Every bit-width fits 24 GB** at max_length 768; the worst case (16-bit r32) peaks at 15.1 GiB allocated.
  - Rank barely changes speed or memory (M1 ≈ M3). 8-bit is the slowest (LLM.int8() matmuls); 16-bit is the fastest.
  - **Batch 8 × 2 is not faster** than 4 × 4 (0.72 vs 0.81 samples/s for M1; 0.86 vs 0.91 for M9), so the planned 4 × 4 stays. Caveat: the batch-4 and batch-8 M1 runs landed on different GPUs (see below).
  - Validation pass over 1,000 pairs: ~4 min.
- **Inference:**
  - First attempt: **CUDA out of memory** for the 16-bit model drawing all 20 samples in one `generate()` on the longest HumanEvalFix-Java prompts (the KV cache of 20 × ~1,650 tokens is ~16 GB next to 12.9 GB of weights).
  - **Fix:** `generate_all` now draws each prompt's samples in chunks of `--batch_size` sequences (default 10 → 2 × 10), the same for every model, so throughput stays comparable. Re-test on the 4 longest prompts: peak 19.6 GiB allocated, 108 tokens/s, 70 s per bug, 0 outputs hitting max_new_tokens.
  - HumanEval-Java, M1 dry adapter, 4 tasks × 20 samples: 137 tokens/s. Megadiff, M9 dry adapter, greedy, 40 pairs: syntax 100%, EM 5.0%, CodeBLEU 0.904 (50 training steps only: a pipeline check, not a result).
  - Latency probe (batch 1, greedy): TTFT median 0.15–0.17 s, decode 12–16 tokens/s.
  - The test harness ran on the generated samples inside the container: pass@1 and pass@10 are computed (outcomes passed / failed / compile_error).
- ⚠️ **Modal's "A10" GPU type is not one GPU model:** the first job got an NVIDIA A10, later jobs an NVIDIA A10G (same 24 GB class and similar compute, but different board power). Speed and energy (RQ3) are not comparable across the two. If Modal is used for official runs, each job must check the GPU name and refuse a mismatch (`gpu_name` is already logged in every CSV row).
- **Bugs found and fixed:**
  1. A detached `modal run` was cancelled when the local client process exited (M6 at step 37). Long jobs now use `modal deploy` + a server-side `spawn`.
  2. `append_csv` wrote rows with extra columns (e.g. Megadiff scores) under the old header, so the values were unlabelled. It now rewrites the file with the union header.
  3. The clm repository ships HumanEval-Java as a tarball; the README, `setup_gpu_vm.sh` and the Modal image now extract it.
  4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the Modal container (the same for every run) to reduce fragmentation.
- ⚠️ **Compute estimate, revised upwards** (A10/A10G):
  - Training: 5 runs per bit-width (3 ranks + 2 extra seeds at r16): 4-bit ~39 h, 8-bit ~48 h, 16-bit ~35 h → **~120 GPU-h**.
  - Evaluation: ~3–5 h per model × 16 models → **~50–80 GPU-h**.
  - **Total ≈ 170–200 GPU-h (~$190–240)**, up from 110–150 (#38). It still fits the $700 AWS credit; it needs about 7–8 days of continuous GPU time on one card, which puts pressure on the schedule (§8 of the plan).
- ❓ **For the user:** the platform for the official runs (AWS A10G if approved; Modal with GPU-name pinning; another provider), given the higher estimate and the deadline.

### 47. Which GPUs do the related papers use? ✅
- Checked each paper's implementation-details section:

| Paper | Hardware | Note |
|---|---|---|
| Weyssow et al., TOSEM 2025 (PEFT/QLoRA for code generation) | **1× NVIDIA RTX A5000, 24 GB** | Explicitly designs for "a single 24GB GPU"; models chosen to fit it |
| Huang et al., TSE 2025 (fine-tuning LLMs for APR) | RTX 3090 24 GB (small models); **Tesla A100 80 GB** (billion-scale, 8-bit QLoRA) | |
| Li et al., ASE 2024 (PEFT for APR) | **1× A800 80 GB** | |
| Wei et al., FSE 2023 (quantized code generation) | AWS p3dn.24xlarge (V100s) + a MacBook Pro | Inference only |
| Vallecillos-Ruiz et al., ICSME 2026 (quantization for APR) | HPC cluster, **NVIDIA GH200** | Post-training quantization, inference only |

- **There is no standard GPU.** The closest PEFT study (Weyssow) uses a single 24 GB data-centre card of the same class as our A10G, so one 24 GB GPU is well precedented for the "consumer-budget" framing. The APR fine-tuning papers use 80 GB cards because they don't study memory limits.
- **Consequence:** report the exact GPU model in the thesis; a 24 GB card is the faithful choice for RQ2. A larger card capped at 24 GiB is defensible but second best.

### 48. Justification of the GPU choice (A10G, 24 GB) ✅
- **Asked by the user:** how to justify and defend the A10G (24 GB) in the thesis. Draft subsection written outside this repo: `thesis/03-methodology-hardware.md`.
- **Arguments:**
  1. **Construct validity for RQ2:** the question is what fits a ≤ 24 GB budget, so the card's *physical* memory must be the limit: OOM is observed, not inferred (80 GB cards) or emulated (a larger card capped in software, where the CUDA context and non-PyTorch allocations sit outside the cap). Evidence from the dry run (#46): 16-bit inference with 20 samples at once on the longest prompts *did* OOM, so the budget binds.
  2. **Conservative stand-in for consumer 24 GB cards:** Ampere GA102 like the RTX 3090 (bf16, int8 tensor cores used by bitsandbytes). Usable memory measured at ~22 GiB, below a consumer card's 24 GB, so "fits on the A10G" implies "fits on a 3090/4090". Peak allocated memory depends on the model, settings and software stack, not the card class, so RQ2 numbers carry over.
  3. **Precedent (#47):** Weyssow et al. (TOSEM 2025), a single 24 GB RTX A5000; Huang et al. (TSE 2025), a 24 GB RTX 3090 for part of their study. The APR papers on 80 GB cards don't study memory limits.
  4. **Measurement quality and reproducibility:** data-centre card (ECC, stable clocks), NVML's cumulative energy counter (used in the dry run), widely available on demand, one GPU model for the whole campaign, ~$1.1–1.2/h.
- **Alternatives rejected:** 40–96 GB cards (no real limit for RQ2); 48 GB capped (emulation, second best); 16 GB cards (M7–M9 don't fit; T4 has no bf16); RTX 4090 (a legitimate alternative, named as such: community clouds only, no ECC, uncertain availability of one model over weeks).
- **Threats and mitigations:** absolute speed and energy are specific to the card → conclusions rest on relative comparisons on identical hardware; usable memory ~22 GiB → conservative, and the exact figure is reported; A10/A10G mixing on one platform → `gpu_name` is logged per run and must be pinned.
- **To do before submission:** take spec figures (bandwidth, board power, compute) from NVIDIA's official datasheets and cite them; consistently use the wording "24 GB-class GPU".

### 49. Pinned versions, one-board rule enforced, split IDs published; Modal hands out A10, not A10G ⚠️
- **User's plan (2026-10-21):** use the remaining Modal credit ($27.41) to keep going; move to AWS if the appeal is approved, otherwise pay for the rest. Conditions agreed first (one GPU model, identical software, spend limit, resumable long runs).
- **Done:**
  - `requirements.lock.txt`: exact versions frozen from the Modal image (Python 3.11; torch 2.14.1, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.1.0, codebleu 0.7.0). The Modal image installs from it; any other host must too.
  - `scripts/modal_app.py`: each job states the one GPU board it accepts and raises (so Modal retries in a fresh container) on any other; job timeout 24 h; every job appends board name, driver, power limit and memory to `results/platform_log.csv`; a failing command returns its exit code instead of being retried.
  - `splits/`: row indices of the Megadiff train / validation / test splits and SHA-256 checksums of all five generated data files (the data itself isn't redistributed).
  - Thesis drafts (outside this repo): a dataset-modifications subsection with a table of all 11 changes (D1–D6, E1–E4, P1), following the user's question whether altering official datasets is acceptable (answer: yes, if every change is documented, deterministic, applied identically, limited to objective defects in evaluation data, and reported alongside the unmodified version).
- ⚠️ **Finding:** the first M0 job requiring an A10G was refused 6 times in a row (first attempt + 5 retries): Modal's "A10" type returned an **NVIDIA A10** every time. An A10G can't be obtained reliably on Modal, so "start on Modal, continue on AWS" would mix GPU models.
- **User decision (option B):** wait for the AWS answer (check on 2026-10-22, 10:13). Meanwhile run M0 on Modal's A10 as a **labelled harness check** (`results/pilot/`, run name `M0_pilotA10`): correctness metrics don't depend on the board, so it validates the harness against the published 52.2 (HumanEvalFix-Java, DeepSeek-Coder-6.7B-Instruct). Its speed and energy numbers are **not** official. If AWS approves → all official runs on the A10G there; if not → all official runs on Modal's A10 (the hardware subsection then says A10).

### 50. M0 harness check on Modal's A10: results, two Java harness bugs, fixes, re-scoring ✅
- **Run:** `M0_pilotA10` (untuned DeepSeek-Coder-6.7B-Instruct, 16-bit), NVIDIA A10, 2026-10-21 ~17:20–21:50 CEST, Modal job `fc-01M5FGXG9MVCFJ8HXBBTWQBXNP`. Outputs in `results/pilot/` on the Modal volume. **Not an official result** (wrong board for the official runs; see #49): a correctness check of the harness against the published 52.2.
- The local process watching the job was killed by Claude Code when the Mac ran low on memory (~18:00); the Modal job wasn't affected (server-side spawn, #46). Progress was checked by hand afterwards.
- **Generation** (A10; speed and energy indicative only):

| Benchmark | Records × samples | Duration | Peak VRAM alloc | Hit max_new_tokens | Energy |
|---|---|---|---|---|---|
| Megadiff test (greedy) | 1,000 × 1 | 36 min | 17.3 GiB | 14 | 89.7 Wh |
| HumanEvalFix-Java | 164 × 20 | 99 min | 19.7 GiB | 0 | 245.8 Wh |
| HumanEval-Java | 163 × 20 | 84 min | 18.9 GiB | 6 | 208.3 Wh |

- **Scores as first measured (harness v1):**

| Benchmark | pass@1 | pass@10 | Other |
|---|---|---|---|
| HumanEvalFix-Java | **44.42%** (published 52.2) | 70.06% | compile_error 508 / 3,280 samples |
| HumanEval-Java (all 163 / clean 162) | 59.79% / 59.78% | 76.87% / 76.73% | compile_error 554 / 3,260 |
| Megadiff test | — | — | EM 3.1%, CodeBLEU 0.741 (naive copy 0.876), syntax validity 87.1% |

- ⚠️ **Two harness bugs found by classifying the compile errors:**
  1. **HumanEvalFix-Java, 461 of 508 compile errors: `duplicate class: Main`.** The model copies the example test's `Main` class into its answer; the harness appended the hidden test's `Main` as well. The reference harness for the published number (bigcode-evaluation-harness, `humanevalpack.py`) never runs a model-written main: it truncates Java generations before `public static void main`. **Fix:** a top-level `class Main { … }` block in the model output is removed (brace-matched) before the tests are added.
  2. **HumanEval-Java, ~330 of 554 compile errors: `illegal start of type` etc.** from `import …;` lines the model puts above its method. When the method is spliced into the class, the import lands mid-file, where Java forbids it. **Fix:** import lines in a spliced prediction are moved to the top of the file (after the package line), skipping ones the class already has.
  - Both rules apply identically to every model and to the reference fixes. The remaining compile errors (type errors, methods defined twice, missing symbols) are model mistakes and stay counted as failures.
- **Harness re-validation after the fixes** (reference code only, on the Mac): HumanEvalFix-Java reference 100% / buggy 0%. HumanEval-Java reference 99.38% on the first run (1 task timed out under 8 parallel workers while the Mac was low on memory; JUnit's 3 s limit) and **163/163 on the re-run**; buggy 0% on the clean set.
- **Also new:** a CPU-only `score` job in `scripts/modal_app.py` (8 vCPU, 16 GB, no GPU) to run unit tests on saved predictions, since model-generated code isn't run on the laptop.
- ✅ **Re-scoring:** the **same saved samples** (no new generation) re-scored with harness v2 on Modal CPU, job `fc-01M5G29DKF97QXM34GTYBYDZSZ` (~22:40–23:00 CEST) → `results/pilot/M0_pilotA10_tests_v2.json`. The v1 numbers above are kept for transparency. The result was opened only after the acceptance rule (#52) was committed.

| Benchmark | pass@1 v1 → **v2** | pass@10 v1 → v2 | compile_error v1 → v2 |
|---|---|---|---|
| HumanEvalFix-Java | 44.42 → **53.11%** | 70.06 → 72.01% | 508 → 47 |
| HumanEval-Java (all / clean) | 59.79 / 59.78 → **64.79 / 64.69%** | 76.87 / 76.73 → 81.94 / 81.82% | 554 → 291 |

  - **HumanEvalFix-Java: 53.11% vs the published 52.2 (+0.9 pp), inside the ±6 pp band of #52 → harness validated.**
  - Remaining HumanEvalFix-Java compile errors (47) are model mistakes: broken `switch` syntax, type errors, a public `Solution` class.
  - HumanEval-Java: 122 of the remaining 291 compile errors are `cannot find symbol`, and most come from **the problem descriptions, not the model or the harness**; see #53.
- **Other actions today (2026-10-21):** hourly Vultr ticket check stopped at the user's request (the user follows the ticket); Modal spend limit not yet set (asked the user, $27.41 credit before M0); commits `cfce1b8` (harness fixes + scoring job).

### 51. Terminology: what the "test harness" is ✅
- **Asked by the user:** what "harness" means in #50.
- **Definition:** the test-execution (evaluation) harness is the code that grades the model's answers: `src/execution.py`. For each candidate fix it:
  1. extracts the Java code from the model's reply (drops explanations and code-block markers);
  2. assembles a runnable program: HumanEvalFix-Java = the model's `Solution` class + the hidden test class `Main`; HumanEval-Java = the model's method spliced into the original class file in place of the buggy method;
  3. compiles it with `javac` (failure → `compile_error`);
  4. runs the unit tests with a time limit (`passed` / `failed` / `runtime_error` / `timeout`);
  5. counts passes over the 20 samples per bug → pass@1 and pass@10.
- **Why harness bugs matter:** a correct fix can be marked wrong if the program is assembled badly. This is what happened in #50 (duplicate `Main` class; imports pasted mid-file), and it lowered M0's HumanEvalFix-Java pass@1 to 44.42% against the published 52.2.
- **How the harness is validated:** reference fixes must pass 100% (a known-correct answer failing means the harness is broken) and the buggy code 0% (buggy code passing means the tests can't detect the bug). This check is re-run after every harness change.
- **Thesis wording:** "test-execution harness" or "evaluation harness".

### 52. Acceptance rule for the M0 harness check, fixed *before* reading the re-scored result ✅
- **Asked by the user:** what if M0 scores *above* 52.2? Written down at ~23:05 CEST on 2026-10-21, after the re-scoring job (#50) finished but **before its output was opened**, so the rule can't be fitted to the number.
- **52.2 is a reference point, not a target or a ceiling.** It was measured with a different protocol (bigcode-evaluation-harness prompt and post-processing, its own library versions), so an exact match isn't expected in either direction.
- **Statistical tolerance:** with 164 tasks, the per-task pass rates of the v1 scoring have SD 0.393, so the standard error of pass@1 is ±3.1 pp and the 95% interval about **±6 pp**. (This is spread over tasks only; it ignores the reference value's own uncertainty, so it's a lower bound.)
- **Rule:**
  1. **|M0 − 52.2| ≤ 6 pp → harness validated**, whichever side.
  2. **Outside that band, in either direction → investigate with the same checks**, not only when the score is low:
     - buggy code must still score 0% and reference fixes 100%;
     - manually inspect a random sample of ~20 *passed* and ~20 *failed* samples (false passes would show as wrong code that passed);
     - confirm the tests really run (assertions executed, not skipped);
     - list the protocol differences (prompt wording, example tests in the prompt, chat template, max_new_tokens, post-processing).
  3. **Any further harness change must be justified on its own terms** (it fixes a demonstrable defect, matches the reference harness, and passes the 100% / 0% check). It is never justified by moving the score towards 52.2.
- **Why a high score would be suspicious:** a harness that's too lenient (false passes) inflates every model equally, hiding real differences between configurations. A score above the reference therefore needs the same scrutiny as one below.
- **Why it matters less than it seems:** M0 is a baseline. The thesis's conclusions rest on differences between configurations, all scored by the same harness, so a constant protocol offset affects every model alike.

### 53. HumanEval-Java descriptions name functions differently from the Java methods ⚠️ ✅ decided
- **Found while checking the remaining compile errors of #50 (v2).** 122 HumanEval-Java samples fail with `cannot find symbol`, concentrated on a few tasks: GET_CLOSET_VOWEL 20, MIN_SUBARRAY_SUM 18, MATCH_PARENS 13, FIND_ZERO 11, SORTED_LIST_SUM 9, DECODE_CYCLIC 8.
- **Cause 1, renamed methods:** the description uses another name for the function than the Java method the tests call, and the model "corrects" the method name to match, so the tests no longer find it. Examples: `get_closest_vowel` vs `get_closet_vowel`; `minSubArraySum` vs `min_subarray_sum`; `digitSum` vs `digit_sum`; `words_string` vs `words_strings`; `sort_array` vs `sort_array_binary`; `triangle_area` vs `triangle_area_2`; `list_sort` vs `sorted_list_sum`; `specialFilter`, `Strongest_Extension`, `solve` vs the Java names.
  - **Most of these come from the benchmark's own Java-file comments** (e.g. GET_CLOSET_VOWEL, MIN_SUBARRAY_SUM, WORDS_STRINGS), not only from the HumanEval fallback descriptions (P1, #40). An automatic scan flagged 22 tasks; about 12 are real name mismatches (the rest are words like `len(`, `range(` in examples).
- **Cause 2, helper functions:** some descriptions mention Python helpers that don't exist in the Java class (likely for FIND_ZERO, DECODE_CYCLIC, MATCH_PARENS), and the model calls them. To be confirmed task by task.
- **Why it matters for the comparison:** fine-tuned models trained on Megadiff (method in → same method out) are likely to keep the signature, while M0 renames. That would inflate the measured gain from fine-tuning for a reason that has nothing to do with repair ability.
- ❓ **Decision for the user** (prompt design P1; no official run is affected yet):
  - (a) Keep the descriptions as they are: renames count as failures, identically for all models.
  - (b) **Recommended:** in every description, replace the name of the function being fixed with the Java method's name (and its examples), and remove or translate references to helpers that don't exist in Java. Documented as a P1 refinement in the modifications table.
  - (c) Harness-side renaming of the prediction back to the expected name: rejected (too lenient; it would also accept genuinely wrong outputs).
- **Worked example shown to the user (GET_CLOSET_VOWEL):** the benchmark file's own comment says `get_closest_vowel("yogurt") ==> "u"`; the method is `public static String get_closet_vowel(String word)`; the JUnit test calls `humaneval.buggy.GET_CLOSET_VOWEL.get_closet_vowel("yogurt")`. M0's sample 0 answered "The bug … is in the method name" and renamed the method, leaving the real bug (the loop bound `i >= 0` with `charAt(i - 1)`) unfixed, so that sample would fail even without the rename.
- **Arguments for keeping the descriptions unchanged (option a):** it's the benchmark's own text (for most of these tasks); no intervention by us; identical for every model; renaming a public method breaks its callers, so it's a genuine repair mistake; the instruction asks for the *corrected method*, not a renamed one.
- **Arguments against:** in these tasks the benchmark's text itself is inconsistent, which pushes models towards renaming; and the effect may differ between M0 and fine-tuned models (signature copying), which could bias the gain.
- **Proposed before deciding:** measure the size of the effect with a diagnostic-only re-score of the saved M0 samples in which renamed methods are renamed back (not a harness rule, never used for reported results). If few renamed samples would pass anyway, option (a) costs almost nothing.
- ✅ **Diagnostic run (user approved, 2026-10-21 ~23:30 CEST):** `analysis/diagnostics/rename_diagnostic.py` on Modal CPU, job `fc-01M5G6H16QJYAE05BV6RT85SAX` → `results/pilot/M0_pilotA10_rename_diagnostic.json`. Diagnostic only; never part of scoring.
  - Of 3,260 M0 samples, **48 renamed the method** (same name spelled differently: `get_closest_vowel` 20, `minSubArraySum` 18, `sortedListSum`/`sortedList` 6, `words_string` 4). **27 returned a different method altogether** (MATCH_PARENS: `match_parens` instead of the buggy helper `check`, 13; FIND_ZERO: the helper `poly` instead of `find_zero`, 11; SORTED_LIST_SUM: `list_sort`, 3). Renaming those back would be meaningless, so they weren't re-run.
  - Renamed back, **13 of the 48 pass**: MIN_SUBARRAY_SUM 8/18, SORTED_LIST_SUM 5/6, GET_CLOSET_VOWEL 0/20, WORDS_STRINGS 0/4. The other 35 fail anyway (20 runtime errors, 14 wrong answers, 1 compile error), i.e. the model renamed *instead of* fixing the bug.
  - **Effect on the score:** pass@1 64.79% → 65.18% (**+0.39 pp**); pass@10 81.94% → 82.55% (+0.61 pp), if every rename were forgiven.
  - **Interpretation:** even in the worst case, where a fine-tuned model never renames and M0 always does, the descriptions shift the comparison by at most ~0.4 pp of pass@1, far below the ±6 pp sampling interval of a 163-task benchmark. The "different method" cases are a model error regardless of the descriptions.
  - **Recommendation:** option (a), keep the benchmark's descriptions unchanged; report this diagnostic in the threats-to-validity chapter.
- ✅ **User decision (2026-10-22): option (a): keep the HumanEval-Java descriptions unchanged.** No change to the data or prompts. Renamed methods and "different method returned" count as failures for every model. The thesis reports the diagnostic (48 renames in 4 tasks; forgiving them would change M0's pass@1 by +0.39 pp) as a threat to validity.


### 54. Why a widely used benchmark can contain defects ✅
- **Asked by the user:** how can a dataset used by many published papers have mistakes?
- **Defects in popular benchmarks are well documented:** EvalPlus (Liu et al., NeurIPS 2023) showed that HumanEval's tests are often too weak, so wrong solutions pass; Defects4J keeps an official list of deprecated bugs; Northcutt et al. (NeurIPS 2021) found label errors in the test sets of ten widely used datasets (incl. ImageNet). In this project: CodeXGLUE's broken reference syntax (#9), HumanEvalFix's defective tasks (#22). Wide use means convenient and comparable, not audited. **[TODO: verify these citations before using them in the thesis.]**
- **Why prior work didn't hit the name mismatch (#53):** HumanEval-Java was made by translating Python HumanEval to Java. Class comments were copied from the Python docstrings (`get_closest_vowel`), the Java methods were written by hand (`get_closet_vowel`, a typo), and the tests call the Java names. Jiang et al. and Li et al. prompted with the buggy method only, without the comment (#40), so the inconsistency never reached their models. It affects us because we add the description to the prompt. It does **not** invalidate their results.
- **Defects that did affect everyone:** `ISCUBE` (buggy file identical to the fix: a free pass for any model returning the code unchanged, inflating all scores equally); `GET_ROW` (swapped package lines; effect depends on the build setup).
- **For the thesis:** report the defects found (Table 3.d) as rigour; optionally report them upstream (GitHub issue on lin-tan/clm, only with the user's approval since it's a public post).

### 55. Professor's requirement: "use a dataset published in a journal" ⚠️ ❓
- **Raised by the user (2026-10-21):** their professor asks for datasets that have been published in a journal. Checked the venue of each dataset:

| Dataset (role) | Venue of the dataset paper | Use in journal-published research |
|---|---|---|
| Megadiff (training) | **arXiv technical report** (Monperrus et al., 2021, arXiv 2108.04631), not peer-reviewed | **RepairLLaMA** (Silva, Fang, Monperrus), **IEEE TSE 2025**, DOI 10.1109/TSE.2025.3581062: fine-tunes on Megadiff, single-function pairs, deduplicated, total length < 1,024 tokens (30k–50k pairs) |
| HumanEval-Java (test) | **ICSE 2023** (Jiang et al., DOI 10.1109/ICSE48619.2023.00125), peer-reviewed conference | The same **RepairLLaMA TSE 2025** evaluates on HumanEval-Java (**162** single-function bugs, the same clean set as ours, #38) |
| HumanEvalFix-Java (test) | **ICLR 2024** (OctoPack), peer-reviewed conference | No journal use found in this check |
| CodeXGLUE refinement (dropped, #39) | NeurIPS 2021 Datasets & Benchmarks; built from **Tufano et al., ACM TOSEM 2019** (journal) | — |

- **Two readings of the requirement:** (strict) the dataset itself must come from a journal paper, which only CodeXGLUE/Tufano meets; (practical) the dataset must be established, peer-reviewed and used in journal research, which our design meets via the RepairLLaMA TSE 2025 precedent (Megadiff training + HumanEval-Java evaluation, same preprocessing). In SE, ICSE is a CORE A* venue.
- **Suggested to the user:** ask the professor which reading applies, presenting the table and the TSE precedent; offered to draft a factual email for the user to send. Fallbacks if strict: (a) keep Megadiff + HumanEval-Java citing TSE 2025, HumanEvalFix-Java as secondary; (b) return to Tufano's journal dataset (reopens #32–#36). No change until the professor answers.
- #53 decided: descriptions unchanged (option a).

### 56. Email to the professor drafted (dataset requirement) ✅ draft
- 2026-10-22: drafted at the user's request; the user sends it. Saved outside the repo (`thesis/emails/2026-10-22_professor_dataset_question.md`).
- Content: why CodeXGLUE was dropped (abstracted code, no tests, naive copy 0.887 CodeBLEU, Java-only vs six languages); the proposed datasets with their venues; the RepairLLaMA IEEE TSE 2025 precedent; the question whether "published in a journal" means (a) introduced in a journal paper or (b) peer-reviewed/established and used in journal research.
- To check before sending: the RepairLLaMA DOI on IEEE Xplore.

### 57. Candidate dataset: Code4Bench (journal-published) ❓
- **Raised by the user (2026-10-22)**, after emailing the professor about #55 (the user will report the reply).
- **Facts checked:** Code4Bench (Majd et al., *Journal of Computer Languages*, vol. 53, pp. 38–52, 2019 **[TODO: verify year and DOI on ScienceDirect]**); Zenodo 10.5281/zenodo.2582968, CC BY 4.0, 636 MB database dump. Codeforces submissions: ~3.4 M programs, 28 languages, **~205k Java** (Java 6/7/8). Tables include `testcases` (input / expected output per problem, with an `isValid` flag for incomplete tests) and `Realfaultslocations` (faulty submission → accepted submission, with localised faults).
- **Strengths:** journal-published (meets the strict reading of #55); executable tests (pass@k possible); real human bugs; open licence; pairs and tests in one dataset.
- **Risks:** (1) **contamination**: Codeforces solutions are widely published online, so DeepSeek-Coder likely saw many in pre-training; (2) competitive-programming stdin/stdout programs, not methods in software projects (Defects4C calls such bugs "simplified"); (3) a wrong→accepted pair may be a rewrite, not a small fix; (4) problems with multiple valid outputs (special checkers) and truncated tests break exact output comparison; (5) no use in LLM APR research found, so no published reference number; (6) ~2–3 days of loader + harness work.
- **Assessment:** don't switch blindly. If the professor insists on the strict reading, the most useful role is an **additional journal-published, test-based Java benchmark** next to HumanEval-Java. Using it for training would turn the study into competitive programming.
- **Proposed:** a CPU-only feasibility check like #38 (Java pair count and diff sizes, test completeness, special-checker share, problem dates vs DeepSeek-Coder's training cut-off), ~half a day. Awaiting the user's go-ahead.

### 58. Code4Bench feasibility check ✅
- **User approved** the check proposed in #57 (2026-10-22).
- **Where:** on a Modal CPU container, not the laptop (only ~9.6 GB free disk; memory pressure earlier). `scripts/modal_app.py`: the CPU `score` job gets `unar` and a new volume `thesis-raw` for downloaded third-party archives. (A 200 GB scratch-disk request was rejected by Modal, whose minimum is 512 GiB; the default disk is used.)
- **Download:** `code4bench.rar` (636 MB, Zenodo record 2582968), verified against the published MD5 `3ae77dfabec6e7a97ca7608c1aa41c04`, extracted on the volume.
- **To measure:** Java faulty→accepted pairs and their diff sizes; test-case completeness (`isValid`); problems needing special checkers (multiple valid outputs); problem dates vs DeepSeek-Coder's training cut-off (contamination).
- ✅ **Download verified:** MD5 `3ae77dfabec6e7a97ca7608c1aa41c04` matches Zenodo. Extracted: 6.4 GB of MySQL dump files (`source` 5.9 GB, `realfaultslocations` 775 MB, `testcases` 72 MB, `problems` 16 MB, plus users/locations/languages/verdicts).
- **Schema findings:**
  - `testcases.inputdata` / `expectedresult` are `varchar(600)`: long Codeforces tests are stored truncated; `isValid` is `notValidated` / `fail` / `pass`.
  - `realfaultslocations`: ~9.1 M faulty→accepted pairs (all languages; C/C++ also in a separate table), with change/insert/delete counts and line locations.
  - `problems`: 2,990 problems with contest number and the HTML statement; `source`: submission, code, author, verdict, language, `sent` timestamp, duplicate flag.
  - 41 languages (Java 8, Java 7, Java 6, Java 8 ZIP among them); verdict 2 = "Wrong answer on test".
- 🔄 **Analysis running** (`analysis/feasibility/code4bench_feasibility.py`). First attempt (job `fc-01M5G9XQ8SV5TMJGW86ZR2QNYD`) failed at once, "file not found": it ran on a warm container from the previous deployment, before the new script was deployed. Re-started as job `fc-01M5G9Z0HJMKFN6YV7772NZ19C` (~00:45 CEST) after confirming the file is in the container.
- ✅ **Results** (`results/code4bench_stats.json`; Java code also stored in `java.sqlite` on the `thesis-raw` volume for later use):
  - **Problems:** 2,990; **610 (20%)** have statements with multiple-answer or error-tolerance wording (regex heuristic), so they need a special checker rather than exact output comparison.
  - **Test cases:** 175,373 in total: pass 91,826 / fail 80,546 / notValidated 3,001. **81,151 (46%) are truncated** (600-character cap). Only **623 problems** have *all* their tests validated and untruncated.
  - **Java submissions:** 205,113 (Java 8 154,087; Java 7 27,931; Java 6 22,988; Java 8 ZIP 107), submitted **2010–2017** (2017: 97,333; 2016: 51,636).
  - **Java fault pairs:** 106,690 of 9,123,245 overall; all pair a user's own faulty and accepted submissions to the same problem; 105,132 unique. Faulty verdicts: wrong answer 71,156, time limit 15,162, runtime error 13,046, compilation error 5,139.
  - ⚠️ **Change size: median 28 changed lines** (p75 68, p95 284). Only 22,568 (21%) change ≤ 6 lines (Megadiff: median 2, 81% ≤ 6). Most pairs are rewrites, not small fixes.
  - **Strict candidate set** (same author and problem, faulty verdict "wrong answer", ≤ 6 changed lines, no multiple-answer/tolerance wording, all tests validated and untruncated): **7,052 pairs over 460 problems** (median 8 pairs per problem; contest ids 1–839).
- **Assessment:**
  - **As training data: not suitable.** Pairs are mostly rewrites; restricting to small fixes leaves 7k pairs over 460 problems, and the test sets would have to come from the same problems.
  - **As an additional, journal-published, test-based evaluation set: feasible.** E.g. one wrong-answer bug per problem → up to ~460 Java bugs with complete stored tests (comparable to HumanEvalFix-Java + HumanEval-Java combined).
  - ⚠️ **Contamination: high risk.** All submissions date from 2010–2017, years before DeepSeek-Coder's training data was collected, and Codeforces solutions are widely mirrored on GitHub. Results would need a contamination probe and a clear caveat.
  - ⚠️ **Weaker tests:** only the tests shown on Codeforces are stored (the full hidden test sets aren't), so a fix that passes them can still be wrong.
  - Not yet measured: token length of whole programs (with the problem statement, prompts may be long), and the effort of a stdin/stdout harness (~1–2 days).
- ❓ **Decision pending** (user, possibly after the professor's reply): add Code4Bench as a third evaluation benchmark, or not.
 It streams the dump without a database server; parsers unit-tested on sample rows first. Candidate rule: same author and problem, faulty verdict "Wrong answer on test", ≤ 6 changed lines, statement without multiple-answer/tolerance wording, all tests validated and untruncated.

### 59. Could the experiments run on the user's Mac? No ✅
- **Asked by the user** (2026-10-22): how long would everything take on the local Mac instead of a cloud GPU?
- **Machine:** Apple M1, 8 GB unified memory, 8 CPU cores, 8-core GPU.
- **Answer: not slow but impossible**, for three reasons:
  1. The 16-bit model's weights alone are 13.4 GB, more than the Mac's 8 GB of memory: M0 and M7–M9 can't even be loaded.
  2. NF4 QLoRA and LLM.int8() (M1–M6) come from `bitsandbytes`, which needs CUDA (NVIDIA); Apple Silicon can't run them.
  3. RQ2/RQ3 measure peak VRAM, speed and NVML energy on a 24 GB NVIDIA GPU; a Mac has unified memory and no CUDA memory or NVML energy counters, so its numbers wouldn't answer the research questions.
- Running a GGUF 4-bit model through llama.cpp would be a different quantization method (not the one studied), inference only.
- **The Mac's role:** data preparation, harness checks on reference code, analysis, writing. A rented NVIDIA GPU is required for training and evaluation.

### 60. LinkedIn posts updated: posts 5 and 6 drafted ✅
- 2026-10-22, at the user's request. Posts 1–4 re-checked against the log: still accurate, no changes needed.
- **Post 5, "My model scored 44%. The bug was in my grader."** (#50–#53): M0 on HumanEvalFix-Java 44.4% → 53.1% vs the published 52.2 after fixing two grader bugs (461 duplicate `Main` classes; ~330 misplaced imports); the acceptance rule committed before re-scoring; the 100% / 0% self-check. Labelled as a pilot harness check, not a thesis result.
- **Post 6, "The dry run"** (#38, #46): peak memory at rank 32 of 7.7 / 10.6 / 15.1 GB (4/8/16-bit); ~7.6 / 9.3 / 6.8 h per full training run; the out-of-memory error with 20 samples at once and the 2 × 10 fix; the cancelled job; batch 8 × 2 not faster; estimate 110–150 → 170–200 GPU-hours (~40% more); dry run ≈ $3.
- Two errors caught in my own drafts before saving: post 6 first said "~100 GPU-hours planned" and "2×" (the log says 110–150 and ~40%), and its memory chart used 4-bit's rank-8 value (7.3) under a "rank 32" label (correct: 7.66). Both fixed.
- GIFs generated with the same brand theme (`_tools/make_gifs.py`, `post5()` / `post6()`); posts live outside this repo (`linkedin-posts/`).

### 61. Supervisor approves the dataset design ✅
- **2026-10-22:** the supervisor replied to the email of #56 that IEEE TSE is a very good journal in this domain and, since the RepairLLaMA TSE 2025 study uses this setup, to go ahead. The user confirmed: "we are good with Megadiff".
- **Decided:** training on **Megadiff** (filtered, #39); evaluation on **HumanEval-Java** and **HumanEvalFix-Java** (#37–#40, #53). The journal requirement (#55) is satisfied under the "used in journal-published research" reading, with RepairLLaMA (IEEE TSE 2025) as the precedent to cite.
- **Code4Bench (#57–#58): not pursued.** It's no longer needed for the journal requirement, and its contamination risk and mostly rewrite-sized pairs argued against it anyway. The feasibility results stay in the log; the downloaded archive, the 6.4 GB extracted dump and `java.sqlite` were **deleted** from the `thesis-raw` Modal volume on 2026-10-22 (the user ran `modal volume rm thesis-raw code4bench -r`). The summary files (`code4bench_stats.json`, feasibility log, schema, inspection listing) are kept on `thesis-results`; the script can rebuild everything from Zenodo.
- **For the thesis:** cite RepairLLaMA (Silva, Fang, Monperrus, IEEE TSE 2025, DOI 10.1109/TSE.2025.3581062 **[TODO: verify on IEEE Xplore]**) as the precedent for both datasets.

### 62. Platform: leaning to RunPod RTX 4090, final after the AWS check ❓
- **Context (2026-10-22):** the university provides no compute funding (the user asked). Options ranked by cost to the user: AWS A10G $0 (appeal pending); Vultr A40 capped at 24 GiB ~$0–20 from its credit (limit ticket pending); **RunPod RTX 4090 ~$80–100** (estimated ~90–110 h at ~$0.90/h, assuming ~2× A10 speed; unverified); Modal A10 ~$170–190.
- **User's leaning: RunPod RTX 4090 (24 GB)**, decided after tomorrow's AWS check (2026-10-23, 10:13). If AWS approves, all official runs go on the A10G instead.
- **Consequences if RunPod is chosen:**
  - All official runs on one card model (RTX 4090); the first ~30 min on the card is a short dry run to confirm speed, memory and whether NVML's energy counter works on a GeForce card (the profiler falls back to power sampling otherwise).
  - The hardware subsection (#48) is rewritten around a **consumer** 24 GB card, which fits RQ2's "developer workstation" framing directly; threats: no ECC memory, community-cloud availability of one card model for about a week.
  - The user creates the RunPod account and adds payment; the pod is set up with `scripts/setup_gpu_vm.sh` and `requirements.lock.txt`; long jobs run inside `tmux` so a closed terminal can't stop them.

### 63. Using the remaining Modal credit while the platform is open: checkpoint/resume, Phase 0 (Li et al.), end-to-end check ✅
- **User (2026-10-22):** use the Modal credit (~$22) in the meantime; approved three card-independent tasks. Speed, memory and energy from these runs are **not official**.
- **#1 Checkpoint and resume (`src/trainer.py`) ✅ (CPU-tested):**
  - Full runs save a checkpoint every `--save_steps` (default 200) optimizer steps; `save_total_limit=1`. On restart, the latest *complete* checkpoint (one with `trainer_state.json`) is resumed: adapter, optimizer, scheduler, RNG and data position. `--no_resume` ignores checkpoints.
  - A resumed run is flagged (`resumed_from` column): its timing and energy cover only the resumed part, so it isn't used for official RQ3 values.
  - **Bug found by the test:** killing the run while checkpoint-4 was being written left an incomplete folder, and the first version tried to resume from it ("Can't find a valid checkpoint"). Fixed by requiring `trainer_state.json` (written last). Re-test, both cases: killed mid-save → resumed from checkpoint-2; killed right after a complete save → resumed from checkpoint-4; both finished all 16 steps with `resumed_from` recorded.
- **#2 Phase 0: reproduce Li et al. (ASE 2024) on HumanEval-Java** (open decision 4, scope (a) of #28): validates our HumanEval-Java harness against a published number, as #50 did for HumanEvalFix-Java.
  - **Their protocol** (read from `zjulgc/llmpeft4apr`): DeepSeek-Coder-6.7B-**Base**, fp16, their LoRA adapter (HF `survoltli/llmpeft4apr`, `deepseek_coder_6.7b/LoRA`); prompt template "@@ Instruction … @@ Response" with "Write a solution to the following coding problem: …" and buggy lines marked `// buggy lines start/end` (their `benchmarks/humaneval.json`, 163 tasks); **beam search, 10 beams, 10 returned**, `max_new_tokens` = prompt length + 256, prompt truncated at 1,024 tokens; each output cut at its last `}`. Validation: the patch replaces the task's "function range" lines in the buggy file, then the JUnit tests run (Maven, 10 s). **pass@k = a plausible patch among the top k beams**, over all 163 tasks (ISCUBE included).
  - **Published target (their released result file):** **pass@1 = 70/163 = 42.94%**, **pass@10 = 109/163 = 66.87%**.
  - **Our reproduction:** their generation protocol exactly (prompt, fp16, beams, truncation, post-processing); scoring with **our** harness (same splice semantics; JUnitCore with JDK 8 instead of Maven; our import hoisting, #50), on Modal's A10.
  - **Acceptance rule, fixed before running:** both pass@1 and pass@10 within **±6 pp** of the published values → our HumanEval-Java harness is validated. Beam search is deterministic in principle, but fp16 numerics and library versions (theirs: torch 2.0.1, transformers 4.35.2, peft 0.6.2; ours: #49 lock file) can change beams, so exact equality isn't expected. Outside the band, in either direction → investigate (same checks as #52); no change to fit the number.
  - **Scoring-path check (reference fixes as fake outputs, their function ranges, Modal CPU):** 160/163 pass; the 3 that fail (DECODE_CYCLIC, ISCUBE, SORT_ARRAY_BINARY) are exactly the tasks whose "function range" in their `humaneval.json` differs from the benchmark's own `humaneval-java-sf.json` (e.g. DECODE_CYCLIC lines 8–26 vs 6–31), so our reference method doesn't fit their range. Model outputs are spliced into *their* ranges, as in their scoring. (A first attempt to run this check on the Mac was cut off by a 15-minute time limit: 1,630 Java runs are too slow locally.)
  - ✅ Generation done (Modal job `fc-01M5HV02DZJEEFYPA6YZ6228EX`; landed on an A10 before the "any" option existed): 163 tasks × 10 beams, no empty outputs. **Our top beam is identical to their released top beam in 146 of 163 tasks**, so the generation protocol is reproduced faithfully (7 tasks have all 10 beams identical; fp16 and library versions shift lower-ranked beams).
  - **Result vs the pre-registered rule:** pass@1 **79/163 = 48.47%** vs 42.94% (**+5.5 pp, inside ±6**); pass@10 **121/163 = 74.23%** vs 66.87% (**+7.4 pp, outside, too high**). By #52's rule, too high gets the same investigation as too low.
  - **Investigation, using their released per-task, per-beam validation results:**
    - Tasks we pass at pass@10 and they don't: 13. In **10 of them all 10 of their candidates were "uncompilable"** (DECODE_SHIFT, EVEN_ODD_PALINDROME, GET_ROW, INTERSECTION, IS_MULTIPLY_PRIME, LARGEST_PRIME_FACTOR, MAKE_PALINDROME, MATCH_PARENS, SKJKASDKD, SORT_ARRAY_BINARY); 3 were "wrong" ×10 (CAR_RACE_COLLISION, CHOOSE_NUM, X_OR_Y). They pass one task we don't: ISCUBE (buggy file equals the fix).
    - **Their own released patches, scored by our grader** (job on Modal CPU): pass@1 75/163 (46.0%), pass@10 118/163 (72.4%). On the 10 "all uncompilable" tasks our grader compiles their patches and passes **some but not all** (e.g. GET_ROW 5 passed / 3 failed / 2 compile errors; LARGEST_PRIME_FACTOR 8/2; SKJKASDKD 7/3), and on the 3 "wrong" tasks it fails their patches too. So the difference sits in **their build of those tasks** (every candidate failing to compile points to a setup defect, e.g. GET_ROW's swapped package lines, #38), not in our grader being lenient: our grader still rejects wrong patches on the same tasks, and the buggy programs still score 0% (#50).
    - **On the tasks both graders could compile** (149 tasks: excluding the 13 tasks where all 10 of their candidates were uncompilable, plus ISCUBE): pass@1 69 vs 73 (**+2.7 pp**), pass@10 108 vs 111 (**+2.0 pp**), both well inside ±6.
  - **Conclusion:** the overall pass@10 criterion was missed (+7.4 pp), and that is recorded as such. The investigation attributes the excess to a demonstrable defect in the reference measurement (13 tasks that could not compile at all in their setup), not to our harness; on comparable tasks our results agree within 3 pp. No change to our harness (rule 3 of #52). **Our HumanEval-Java harness is considered validated**, with this caveat reported. Phase 0 scope (a) of #28 is done; retraining their configuration (scope (b)) is not planned.
  - Our numbers come from our harness (JUnitCore, 3 s per test, import hoisting) versus their Maven run with a 10 s limit per task, a further source of small differences.
  - **Modal wrapper bugs found on the way:** (1) a job couldn't see a file uploaded with `modal volume put` because warm containers don't refresh a volume; jobs now call `results.reload()` at start; (2) after a redeploy, calls can still land on a warm container running the previous code version (as in #58), so a new script or fix may need a retry.
- **#3 End-to-end check** (`scripts/e2e_check.sh`): 4-bit r=8 on 256 training pairs × 2 epochs with checkpoints every 5 steps; killed once a complete checkpoint exists, then resumed on the GPU (4-bit + paged optimizer); adapter saved; generation on all three benchmarks (6 tasks each) with that adapter; Megadiff scoring and unit-test scoring. First attempt (job `fc-01M5HV6GJR2K5X4PCNVV9N12FR`) never started: it required an "NVIDIA A10" and Modal handed out an **A10G six times in a row** (the reverse of #49). For pilot and diagnostic jobs the board doesn't matter, so `run()` now accepts `MODAL_REQUIRE_GPU=any`; official runs must still name one board. Re-started as job `fc-01M5HVW0EDTZM2JAY49ZMFPG7J` (NVIDIA A10).
  - ✅ **Passed every stage** (13:20–13:35 UTC, ~15 min):
    1. Trained 4-bit r=8 on 256 pairs; killed once `checkpoint-5` was complete.
    2. Resumed on the GPU from `checkpoint-5` (4-bit model + paged AdamW state restored) and finished: **32 global steps** (= 256 / 16 × 2 epochs, so no steps lost or repeated), validation loss per epoch [0.0697, 0.0679], adapter saved, `resumed_from = checkpoint-5` recorded. Its timing (1.23 samples/s) covers only the resumed part, as designed, so it isn't comparable with the dry run's 0.81.
    3. Generated with the saved adapter on all three benchmarks (6 tasks each): no outputs hit max_new_tokens; Megadiff scored inline.
    4. Unit tests ran on all 240 samples: HumanEvalFix-Java pass@1 80.8% / pass@10 83.3%, HumanEval-Java 55.8% / 66.7% (6 easy tasks after a tiny training run: a pipeline check, not a result).
  - **The official pipeline now works end to end**, including recovery from an interrupted training run.


### 64. Phase 0 closed; RepairLLaMA check deferred ✅
- **User decision (2026-10-22):** close Phase 0 with the Li et al. reproduction (#63); do an inference-only RepairLLaMA check later, once the official training runs are under way.
- **Rationale:** Phase 0's purpose is met: both harnesses agree with published numbers (HumanEvalFix-Java 53.1 vs 52.2, #50; HumanEval-Java within 3 pp of Li et al. on comparable tasks, #63), and a released adapter was run faithfully (146/163 identical top beams). Further reproductions would compete with ~7–8 days of official GPU time.
- **Not reproduced, with reasons for the thesis:**
  - **Huang et al. (IEEE TSE 2025):** would need retraining a different model family (InCoder/CodeGen, 8-bit QLoRA, ~10–20 GPU-h plus 1–2 days of setup) and adds no new check of our pipeline. Related work.
  - **Wei et al. (FSE 2023):** no released artifact, so only a re-implementation would be possible; inference-only quantization for code generation. Related work.
- **Deferred: RepairLLaMA (IEEE TSE 2025)** inference check: released LoRA adapters on Hugging Face (`ASSERT-KTH/RepairLLaMA-*`, CodeLlama-7B trained on Megadiff) and processed datasets (`ASSERT-KTH/repairllama-datasets`). It gives a same-data, same-benchmark anchor for the discussion chapter. ~1–2 GPU-h plus ~half a day to match their input representation; run on leftover Modal credit in parallel with the official runs, never blocking them.

---

## Open decisions (❓) — to resolve with the user
1. ~~Pass@k~~ → pass@1 primary + pass@10 secondary from n = 20 samples (#37, #39).
2. ~~Multiple seeds~~ → 3 seeds (42/43/44) for M2, M5, M8 (#27).
3. ~~M0~~ → zero-shot (#27).
4. ~~Phase 0 reproduction~~ → Li et al. adapter re-run on HumanEval-Java: generation reproduced (146/163 identical top beams); harness agrees within 3 pp on comparable tasks (#63). Retraining not planned.
5. GPU platform: **AWS A10G** if the appeal is approved; **RunPod RTX 4090 (backlog)** only if denied (#23).
7. ~~Other HumanEvalPack languages~~ → added (#22), then dropped from the study: evaluation is Java only (#37, #39).
6. ~~Second language~~ → HumanEvalFix added (#20). RQ wording updated in the plan (#44); the thesis text follows when written.
8. ~~Training dataset~~ → Megadiff, ≤ 768 tokens, split by file path (#39).
9. ~~Prompt formats~~ → HumanEval-Java with description, HumanEvalFix-Java standard (#40). Greedy for Megadiff EM (#42). ~~GPU dry run~~ → done on Modal (#46).
10. ~~HumanEval-Java description name mismatches~~ → keep the descriptions unchanged, report the diagnostic (#53).
11. GPU platform for the official runs: AWS A10G if the appeal is approved on 2026-10-23, otherwise RunPod RTX 4090 (user's leaning, #62).
12. ~~Professor's "journal-published dataset" requirement~~ → approved by the supervisor: Megadiff + HumanEval-Java + HumanEvalFix-Java (#61). Code4Bench not pursued.
