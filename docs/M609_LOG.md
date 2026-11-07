# M609 Module Log — Peak GPU Memory of Quantized LoRA (branch `m609-memory-study`)

Log for the M609 *Scientific Work for Computer Science* paper, a reduced, preliminary part of the thesis study.
It lives on the module branch only; the thesis plan, code and log on `main` are unchanged (`docs/RESEARCH_LOG.md` stops at #96 here too).
Account-level details (job IDs, billing) are in the private ops log.

Conventions as in `RESEARCH_LOG.md`: ✅ done · 🔄 in progress · ⏳ waiting · ❓ open decision · ⚠️ finding that affects methodology

---

## Status board

| Item | Status | Notes |
|---|---|---|
| Reduced scope (RQ2 only) | ✅ | M1 |
| Branch `m609-memory-study` | ✅ local | Not pushed (M2) |
| Worst-case options in trainer/evaluator | ✅ | Commit bd19cf2 (M3) |
| Training memory grid (typical + worst case) | ✅ | A10G (M4) |
| Inference memory grid, chunks of 10 | ✅ ⚠️ | 16-bit adapters OOM on HumanEval-Java (M5) |
| Inference memory grid, chunks of 5 | ✅ | All 20 cells fit (M6, M7) |
| Email to the module leader about the scope | ✅ draft | Not sent; kept outside this repo (M8) |
| Literature review outline + paper draft | 🔄 | Kept outside this repo (M8) |
| Thesis follow-up: chunk size for the official evaluation | ❓ | `main` still uses chunks of 10 (M9) |

---

### M1. Reduced scope for the M609 paper (2026-11-04) ✅
- **Why:** no GPU budget for the full study yet (AWS GPU quota still 0, monthly cloud credit used on pilots). The user will run the full study when funding arrives (expected next month). The thesis plan is unchanged; this is a module-only reduction.
- **User decision:** the paper answers RQ2 only. Approved wording:
  > *How much peak GPU memory do LoRA fine-tuning and inference of a 6.7B code model (DeepSeek-Coder-6.7B-Instruct) need at 4-bit (NF4), 8-bit (LLM.int8()) and 16-bit (bf16) precision and LoRA ranks 8/16/32, and which set-ups fit consumer GPUs with 16 or 24 GB?*
- Out of scope for the paper: correctness (pass@k, EM), speed/energy trade-off analysis, seeds. These stay in the thesis.

### M2. Separate branch (2026-11-04) ✅
- #96 committed to `main` (0699920) and pushed first, at the user's request. Module work then moved to the local branch `m609-memory-study`.
- **User rule (2026-11-05):** `main` is for the full research only; never change it for module work. All M609 code and logs live on this branch.
- The old dry run (#46, 2026-10-21) wasn't reusable: it used an older training set and trainer. All 9 cells were re-measured.

### M3. Worst-case measurement options (commit bd19cf2) ✅
- `src/trainer.py --longest_n N`: train only on the N longest examples (tokens); logs `longest_n`, `min_seq_tokens`, `max_seq_tokens`.
- `src/evaluator.py --longest_n N`: keep the N longest prompts; `--force_max_new_tokens` sets `min_new_tokens = max_new_tokens`, so EOS can't end a sample early (largest possible KV cache). Logs `longest_n`, `forced_max_new_tokens`, `max_prompt_tokens`.
- `scripts/m609_memory_grid.sh`: (1) typical training, 50 steps, seed 42, full training set, identical batches per cell, `--no_eval`; (2) worst-case training, the 80 longest sequences, 5 steps; (3) worst-case inference for M0 + the 9 typical adapters on HumanEvalFix-Java and HumanEval-Java (longest prompt, 20 samples, full output length, no latency probe).
- Tested on a CPU container with a tiny Qwen2 model. One job, one GPU model (A10G pinned), one commit.

### M4. Training grid results (2026-11-04, NVIDIA A10G, commit bd19cf2) ✅
Peak GPU memory, GiB (allocated / reserved). LoRA on all linear layers, alpha = 2 × rank, gradient checkpointing, sequences ≤ 768 tokens.

| Precision | Static weights | r = 8 | r = 16 | r = 32 |
|---|---|---|---|---|
| 4-bit NF4 | 4.17–4.40 | 7.314 / 7.883 | 7.486 / 8.098 | 7.624 / 8.498 |
| 8-bit LLM.int8() | 7.10–7.32 | 10.239 / 10.936 | 10.411 / 11.148 | 10.548 / 11.469 |
| 16-bit bf16 | 12.63–12.85 | 14.776 / 15.168 | 14.948 / 15.383 | 15.074 / 15.742 |

- Trainable parameters: 20.0 M / 40.0 M / 80.0 M (0.30 / 0.59 / 1.17 %).
- **Worst case (80 longest sequences, 766–768 tokens) gives the same allocated peak in every cell** (reserved within ±0.1 GiB, max 15.80 GiB). The 50 typical steps already contained a 768-token batch, so the typical peak is already the upper bound for this data.
- Rank adds only 0.3 GiB from r = 8 to r = 32; precision dominates (4 → 16 bit: +7.5 GiB).
- Step time ~16.0 s (4-bit), ~18.0 s (8-bit), ~13.2 s (16-bit).

### M5. Inference grid, chunks of 10 (2026-11-04, A10G) ✅ ⚠️
Worst case: longest prompt (HumanEvalFix-Java 923 tokens + 768 output; HumanEval-Java 771 + 1,024), 20 samples in chunks of 10, every sample forced to full length.

| Model | HumanEvalFix-Java | HumanEval-Java |
|---|---|---|
| M0 (16-bit, no adapter) | 20.95 / 21.23 | 21.46 / 21.72 |
| 4-bit (M1–M3) | 12.07–12.30 / 14.58–14.84 | 12.59–12.81 / 15.25–15.48 |
| 8-bit (M4–M6) | 15.00–15.22 / 17.52–18.18 | 15.51–15.74 / 18.42–19.65 |
| 16-bit + adapter (M7–M9) | 21.02–21.25 / 21.38–21.69 | **out of memory** |

- ⚠️ M7–M9 on HumanEval-Java: `torch.OutOfMemoryError` at 21.4 GiB allocated, 22.06 GiB usable on the A10G. The grid script printed a misleading "no adapter, skipping" after the failure (fixed in M7).
- 8-bit generation ~2.2× slower than 4-bit.

### M6. 16-bit HumanEval-Java at chunks of 5 on an A10 (2026-11-05) ✅
- Modal served only plain A10s for this job (the GPU check refused 9 launches). **User decision:** run these 4 cells on an A10 and footnote the card.
- Results (allocated / reserved): M0 17.01 / 17.18, M7 17.09 / 17.32, M8 17.16 / 17.47, M9 17.31 / 17.65. All fit.
- Later identical to the A10G numbers in M7 (to the MB): peak allocation doesn't depend on the card. The footnote is no longer needed; these rows (`*_bs5_A10`) stay in `infer_worstcase.csv` as the cross-card check.

### M7. Decision: chunks of 5 for all models; full inference grid re-run (2026-11-05/06) ✅
- **User decision:** generation in chunks of 5 for all models, in the module branch only (`main` untouched, see M2).
- Commits: 78e3853 (`src/evaluator.py` default `--batch_size` 10 → 5); ac01ccb (grid script writes `infer_worstcase_bs5.csv` and `predictions_bs5/`, keeping the chunks-of-10 results; failed cells print FAILED; the misleading message fixed).
- Re-run of all 20 inference cells on an A10G (commit ac01ccb), training skipped (adapters present). No failures; every sample hit the full output length.

| Precision | Model | HumanEvalFix-Java | HumanEval-Java |
|---|---|---|---|
| 16-bit | M0 | 16.76 / 16.92 | 17.01 / 17.18 |
| 4-bit | M1–M3 | 7.88–8.10 / 9.02–9.58 | 8.14–8.36 / 9.66–9.84 |
| 8-bit | M4–M6 | 10.80–11.03 / 11.39–11.58 | 11.06–11.29 / 11.64–11.83 |
| 16-bit | M7–M9 | 16.83–17.05 / 17.06–17.38 | 17.09–17.31 / 17.32–17.65 |

- Chunks of 5 cut the peak by 4.2–4.5 GiB at every precision. Rank adds 0.15–0.30 GiB.
- Worst-case generation time: 4-bit 260–350 s, 8-bit 610–820 s, 16-bit 175–280 s.
- **Fit (answer to the RQ):** 24 GB: every set-up, training and inference. 16 GB (~15.5 GiB usable): 4-bit and 8-bit training and inference fit; 16-bit inference doesn't (17.2–17.7 GiB); 16-bit training is borderline (15.1 allocated / 15.8 reserved).
- Total module GPU cost ≈ USD 9–10: grid ≈ 6, A10 check ≈ 0.35, chunk-of-5 re-run ≈ 3 (estimates, not yet checked against the bill; details in the private ops log).

### M8. Paper material (2026-11-04 onwards) 🔄
- Email to the module leader explaining the reduced scope: drafted, not sent (private).
- Literature review outline: taxonomy T1 memory components, T2 PEFT, T3 quantization (inference / quantized fine-tuning), T4 systems techniques, T5 empirical memory/APR studies. The comparison table was filled from the PDFs. Closest prior work: Weyssow et al. 2025 (TOSEM; 16/8/4-bit training peak, one rank, attention only), Lee et al. 2026 (IST; 4-bit only), Le Cong et al. 2026 (ICSE; inference only), Vallecillos-Ruiz et al. 2026 (ICSME; post-training quantization, inference). Gap: rank × precision crossed, all-linear LoRA, sampled-generation inference peak per precision, worst-case bounds, training + inference on one model and GPU.
- Paper draft 1 written 2026-11-06 (private): all sections, 2,931 words of prose, 34 references (29 in the literature review), 5 tables.

### M9. Open item for the thesis (not applied to `main`) ❓
- ⚠️ The thesis evaluation (#93/#96: HumanEval-Java at 1,024 tokens, n = 20) uses chunks of 10 on `main`. M5 shows the 16-bit fine-tuned models can run out of memory there on a 24 GB A10G. M7 shows chunks of 5 fix it (≤ 17.7 GiB) at ~1.5× the generation time.
- To decide for the thesis separately (the user's call), then log in `RESEARCH_LOG.md` on `main`.
