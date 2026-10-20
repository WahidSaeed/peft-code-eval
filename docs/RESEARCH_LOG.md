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
| AWS GPU quota (eu-central-1) | ⏳ | Denied → appeal sent 2026-10-20 (see #14, #15) |
| AWS budget alert (credits excluded) | ✅ | (#16) |
| AWS IAM user with MFA | ✅ | (#17) |
| Phase 0: reproduce Tier-A papers | ❓ | Needs artifact availability check |
| Pass@k in RQ1 (CodeXGLUE has no tests) | ❓ | Option: add QuixBugs-Java |
| Citation verification (DOIs) | ❓ | User's paper list must be verified |
| LinkedIn posts 1–3 + GIFs | ✅ drafts | kept outside this repo |
| Learning path | ✅ | `docs/LEARNING_PATH.md` |

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

---

## Open decisions (❓) — to resolve with the user
1. Pass@k: drop, or add QuixBugs-Java as an executable-test benchmark?
2. Multiple seeds for a subset of configs (statistical testing)?
3. M0: zero-shot or few-shot?
4. Phase 0 reproduction: confirm Tier A list once artifact availability is checked.
5. GPU platform: AWS (if the appeal is approved) or RunPod RTX 4090.
