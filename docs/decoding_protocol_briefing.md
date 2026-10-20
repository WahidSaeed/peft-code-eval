# Briefing: Which decoding protocol should my thesis use to evaluate code repair?

*Self-contained summary of an open methodology decision, written 2026-10-21 for discussion (e.g. in NotebookLM). Nothing has been decided yet.*

---

## 1. My study in one paragraph

MEng thesis: **"Resource-Aware Large Code Models: Quantized PEFT for Automated Program Repair."**
I fine-tune **DeepSeek-Coder-6.7B-Instruct** with LoRA adapters while the base model is held at three precisions: **4-bit (NF4 / QLoRA), 8-bit (LLM.int8()), and 16-bit (bf16)**. The LoRA rank is **r = 8, 16 or 32** (α = 2r). That gives 9 fine-tuned models (**M1–M9**) plus **M0**, the untuned model used zero-shot. The r = 16 models (M2, M5, M8) are trained with 3 seeds, so 16 models get evaluated in total.

- **Training data:** CodeXGLUE Java code refinement (medium), a 10k / 1k / 1k subsample.
- **Evaluation:**
  1. **CodeXGLUE refinement test set** (1,000 Java methods): Exact Match (primary), CodeBLEU, syntax validity.
  2. **HumanEvalFix** (984 buggy functions in Python, Java, C++, Go, Rust, JavaScript): **Pass@1**, i.e. does the fixed function pass the hidden unit tests?
- **Resource measurements:** GPU memory, training time and energy, and inference latency (time to first token), decoding speed and energy.
- **The research questions** compare the models *with each other*: how do bit-width and rank change accuracy versus resource cost?

## 2. The problem in plain words

When a language model writes an answer, it produces one token at a time. **Decoding** is the rule for choosing each token. There are three common rules:

| Decoding rule | How it works | Same input → same output? | Cost |
|---|---|---|---|
| **Greedy** | Always pick the single most likely next token. One answer. | Yes (deterministic) | Cheapest |
| **Beam search (e.g. 5 beams)** | Keep the 5 best partial answers at every step, return the best complete one. | Yes (deterministic) | ~2–5× slower, more GPU memory |
| **Sampling (temperature T)** | Pick tokens randomly, weighted by probability. Low T (0.2) = mostly the likely tokens. Run it n times to get n different answers. | No (random) | n× the generations |

**Pass@1** = the chance that one answer passes the tests. It can be measured two ways:
- **Greedy Pass@1:** one greedy answer per bug, scored pass/fail.
- **Sampled Pass@1:** e.g. 20 sampled answers per bug. The per-bug pass rate (say 14/20 = 0.7) is averaged over all bugs. This is a statistical estimate.

**pass@10** = the chance that *at least one of 10* sampled answers passes. That measures something different: whether the right fix is among the model's likely answers.

**My pipeline currently uses greedy decoding for everything.** The question is whether that is acceptable for a code-repair thesis.

## 3. What the literature does (verified from the papers' full texts)

### 3a. The conventions of the two benchmarks I use

| Benchmark | Established protocol | Source |
|---|---|---|
| **CodeXGLUE code refinement** | **Beam search, beam size 5**; Exact Match is the main metric | Official CodeXGLUE baseline scripts (`--beam_size 5`): github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement |
| **HumanEvalFix** | **Sampled Pass@1: 20 samples, temperature 0.2** | OctoPack (Muennighoff et al., 2024), which introduced HumanEvalFix; also used by StarCoder2 (Lozhkov et al., 2024, arXiv 2402.19173, Section 7.2.1) |

→ **Greedy decoding is not the established protocol on either of my repair benchmarks.**

### 3b. Papers using DeepSeek-Coder-6.7B (my model family)

| Paper | Variant | Task | Protocol | Result |
|---|---|---|---|---|
| DeepSeek-Coder technical report (Guo et al., 2024, arXiv 2401.14196) | Base & Instruct | Code **generation** (HumanEval, MBPP) | **Greedy** Pass@1 | HumanEval: Base 49.4, Instruct 78.6 |
| Magicoder (Wei et al., ICML 2024, arXiv 2312.02120) | Base (fine-tuned) | Code **generation** | **Greedy** Pass@1 | — |
| **StarCoder2 paper, Table 12** (arXiv 2402.19173) | **Instruct** (my exact model) | **Repair: HumanEvalFix** (my benchmark) | **Sampled** Pass@1, n = 20, T = 0.2 | Py 44.9, JS 55.3, Java 52.2, Go 42.9, C++ 37.9, Rust 19.5, **avg 42.1** |
| Li et al., ASE 2024 (arXiv 2406.05639) | Base | Repair: HumanEval-Java, Defects4J, QuixBugs | **pass@10** | LoRA r = 32 in bf16 |
| Machácek et al., ICSME 2025 (arXiv 2507.19909) | Base | Repair: same Java benchmarks | 10 patches per bug (pass@10-style) | LoRA r = 8 |
| Vallecillos-Ruiz et al., ICSME 2026 (arXiv 2606.27205) | Instruct | Repair: HumanEval-Java, Defects4J | **pass@10**, T = 0.1 | Post-training quantization, no fine-tuning |
| PAFT (Yang et al., 2026, arXiv 2604.03113) | not stated | Repair: Defects4J, HumanEval-Java | **Sampled** Pass@1, T = 0.7 | 4-bit NF4 QLoRA, r = 32 |

**Summary of the evidence:**
1. Greedy Pass@1 is standard for DeepSeek-Coder on code **generation**, but that is **not my task**.
2. On **repair**, nobody I found uses greedy Pass@1 with DeepSeek-Coder-6.7B. Repair papers use beam search (CodeXGLUE), sampled Pass@1 (HumanEvalFix, PAFT) or pass@10 (Java repair benchmarks I don't use).
3. There *is* a published number for my exact model on my exact benchmark (StarCoder2: HumanEvalFix avg 42.1, sampled protocol). It can serve as a sanity check for my untuned baseline M0.

## 4. Why greedy is still attractive (the methodological case)

1. **Clean comparisons.** My research questions compare 16 models with each other. Greedy is deterministic, so any difference between models comes from bit-width or rank, not random sampling luck. That supports **paired tests** on the same bugs: McNemar for pass/fail, Wilcoxon for continuous scores.
2. **Clean resource measurements.** Beam search and sampling change GPU memory (KV cache), latency and energy. The resource RQ needs one fixed, declared decoding setup.
3. **Cost and time.** One answer per bug is the cheapest option. Thesis deadline is ~2026-11-25, and the GPU quota is still pending.
4. **Realistic use case.** A developer usually looks at one suggested fix.

## 5. Why greedy alone is risky

1. **No precedent on repair benchmarks:** an examiner may ask why I didn't follow the benchmarks' own protocols.
2. **Comparability with published numbers is lost on both benchmarks.**
3. **Robustness:** an examiner may ask whether my conclusions (e.g. "4-bit ≈ 16-bit") only hold under greedy decoding.
4. **Precision:** per language, HumanEvalFix has only 164 bugs. One greedy pass/fail per bug gives a 95% confidence interval of about **±7–8 percentage points per language** and **±3 points pooled** over all 984. Small differences between bit-widths may not be detectable. *(This is a statistics issue whatever decoding is chosen. The planned fix is **equivalence testing** (e.g. TOST with a pre-defined margin such as ±3 points) so that "no difference" can be stated properly.)*

## 6. The options

| Option | What it means | Extra GPU time (rough estimate, to be measured in the dry run) | Pros | Cons |
|---|---|---|---|---|
| **A. Greedy everywhere** | Keep the current pipeline; state it as a deliberate deviation in the threats to validity | 0 | Simplest, cheapest, deterministic | No comparability; robustness question unanswered |
| **B. Benchmark conventions for all 16 models** | Beam 5 on CodeXGLUE + sampled Pass@1 (n = 20, T = 0.2) on HumanEvalFix | Large: likely more than doubles the evaluation budget, plus 20× the test executions on CPU | Fully comparable | "Very big"; threatens the deadline |
| **C. Hybrid (assistant's recommendation)** | Greedy for all 16 models (main results), **plus** beam 5 / sampled Pass@1 on **4 models only**: M0, M2, M5, M8 (the bit-width comparison at r = 16, seed 42) | ~30–45 GPU-h (~$35–55); ~half that with n = 10 | Main results stay clean and deterministic; one extra table answers the robustness question; M0 can be checked against the published 42.1 | Moderate extra cost and one more table; needs a pre-set cut-off date in case time runs short |

Rejected earlier: adding **pass@10**. It is the convention for Java repair benchmarks I don't use, not for HumanEvalFix or CodeXGLUE, and I considered it too big.

## 7. Facts I'm still unsure about

- Exact GPU-hours per model: estimates only until the 50-step dry run on the GPU.
- Whether my supervisor or proposal expects comparability with published numbers. The original plan said "Pass@k", which I later changed to Pass@1 on HumanEvalFix.
- How close greedy Pass@1 and sampled Pass@1 (T = 0.2) usually are. They are typically close at low temperature, but not identical.
- Prompt differences: my HumanEvalFix prompt wording differs from OctoPack/StarCoder2's. I also exclude 4 defective tasks (CPP/38, CPP/23, Go/95, Rust/32) in a "clean" subset.

## 8. Questions I want to explore

1. For a master's thesis whose research questions are *internal comparisons* (bit-width × rank), is a declared deviation to greedy decoding (Option A) acceptable, or will examiners expect benchmark-standard protocols?
2. Is Option C's "protocol robustness check on a subset" a recognised and defensible practice in empirical SE / ML papers?
3. Should CodeXGLUE use beam 5, given that beam search also inflates the inference memory and latency I'm measuring?
4. Is n = 10 samples an acceptable substitute for n = 20 if GPU time is tight?
5. How should I phrase this in the "Threats to Validity" section under each option?
6. How do equivalence testing (TOST) and McNemar's test work for comparing my models, and what margin is reasonable?

## 9. Sources to load alongside this briefing

- CodeXGLUE refinement baseline: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement
- StarCoder2 paper (HumanEvalFix protocol + DeepSeek-Coder-6.7B-Instruct numbers): https://arxiv.org/abs/2402.19173
- OctoPack (introduced HumanEvalFix): https://arxiv.org/abs/2308.07124
- DeepSeek-Coder technical report: https://arxiv.org/abs/2401.14196
- Magicoder: https://arxiv.org/abs/2312.02120
- Li et al., ASE 2024: https://arxiv.org/abs/2406.05639
- Machácek et al., ICSME 2025: https://arxiv.org/abs/2507.19909
- Vallecillos-Ruiz et al., ICSME 2026: https://arxiv.org/abs/2606.27205
- PAFT: https://arxiv.org/abs/2604.03113
- Codex paper (defines the unbiased pass@k estimator): https://arxiv.org/abs/2107.03374
