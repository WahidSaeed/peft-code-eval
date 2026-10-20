# Briefing: Two open methodology problems in my thesis

*Self-contained summary written 2026-10-21 for discussion (e.g. in NotebookLM). **Nothing below has been decided yet.***

- **Problem 1:** Which decoding protocol (greedy, beam search, sampling) should I use to evaluate code repair?
- **Problem 2:** I train only on Java (CodeXGLUE). What does evaluating on 6 languages (HumanEvalFix) actually measure?

---

## 0. My study in one paragraph

MEng thesis: **"Resource-Aware Large Code Models: Quantized PEFT for Automated Program Repair."**
I fine-tune **DeepSeek-Coder-6.7B-Instruct** with LoRA adapters while the base model is held at three precisions: **4-bit (NF4 / QLoRA), 8-bit (LLM.int8()), and 16-bit (bf16)**. The LoRA rank is **r = 8, 16 or 32** (α = 2r). That gives 9 fine-tuned models (**M1–M9**) plus **M0**, the untuned model used zero-shot. The r = 16 models (M2, M5, M8) are trained with 3 seeds, so 16 models get evaluated in total.

- **Training data:** CodeXGLUE Java code refinement (medium), a 10k / 1k / 1k subsample. Each example is one buggy Java method and its fix, with identifiers abstracted (`VAR_1`, `METHOD_1`), space-tokenised, ≤ 512 tokens.
- **Evaluation:**
  1. **CodeXGLUE refinement test set** (1,000 Java methods): Exact Match (primary), CodeBLEU, syntax validity. **No unit tests exist for this dataset.**
  2. **HumanEvalFix** (984 buggy functions, 164 each in Python, Java, C++, Go, Rust, JavaScript): **Pass@1**, i.e. does the fixed function pass the hidden unit tests? Used for evaluation only, never for training.
- **Resource measurements:** GPU memory, training time and energy, and inference latency (time to first token), decoding speed and energy.
- **The research questions** compare the models *with each other*: how do bit-width and rank change accuracy versus resource cost?

---

# PROBLEM 1: Which decoding protocol?

## 1.1 The problem in plain words

When a language model writes an answer, it produces one token at a time. **Decoding** is the rule for choosing each token:

| Decoding rule | How it works | Same input → same output? | Cost |
|---|---|---|---|
| **Greedy** | Always pick the single most likely next token. One answer. | Yes (deterministic) | Cheapest |
| **Beam search (e.g. 5 beams)** | Keep the 5 best partial answers at every step, return the best complete one. | Yes (deterministic) | ~2–5× slower, more GPU memory |
| **Sampling (temperature T)** | Pick tokens randomly, weighted by probability. Low T (0.2) = mostly the likely tokens. Run it n times to get n different answers. | No (random) | n× the generations |

**Pass@1** = the chance that one answer passes the tests. It can be measured two ways:
- **Greedy Pass@1:** one greedy answer per bug, scored pass/fail.
- **Sampled Pass@1:** e.g. 20 sampled answers per bug. The per-bug pass rate (say 14/20 = 0.7) is averaged over all bugs. This is a statistical estimate.

**pass@10** = the chance that *at least one of 10* sampled answers passes. That measures whether the right fix is among the model's likely answers.

**My pipeline currently uses greedy decoding for everything.** Is that acceptable for a code-repair thesis?

## 1.2 What the literature does (verified from the papers' full texts)

### The conventions of the two benchmarks I use

| Benchmark | Established protocol | Source |
|---|---|---|
| **CodeXGLUE code refinement** | **Beam search, beam size 5**; Exact Match is the main metric | Official CodeXGLUE baseline scripts (`--beam_size 5`) |
| **HumanEvalFix** | **Sampled Pass@1: 20 samples, temperature 0.2** | OctoPack (Muennighoff et al., 2024), which introduced HumanEvalFix; also used by StarCoder2 (Lozhkov et al., 2024, Section 7.2.1) |

→ **Greedy decoding is not the established protocol on either of my repair benchmarks.**

### Papers using DeepSeek-Coder-6.7B (my model family)

| Paper | Variant | Task | Protocol | Result |
|---|---|---|---|---|
| DeepSeek-Coder technical report (Guo et al., 2024) | Base & Instruct | Code **generation** (HumanEval, MBPP) | **Greedy** Pass@1 | HumanEval: Base 49.4, Instruct 78.6 |
| Magicoder (Wei et al., ICML 2024) | Base (fine-tuned) | Code **generation** | **Greedy** Pass@1 | — |
| **StarCoder2 paper, Table 12** | **Instruct** (my exact model) | **Repair: HumanEvalFix** (my benchmark) | **Sampled** Pass@1, n = 20, T = 0.2 | Py 44.9, JS 55.3, Java 52.2, Go 42.9, C++ 37.9, Rust 19.5, **avg 42.1** |
| Li et al., ASE 2024 | Base | Repair: HumanEval-Java, Defects4J, QuixBugs | **pass@10** | LoRA r = 32 in bf16 |
| Machácek et al., ICSME 2025 | Base | Repair: same Java benchmarks | 10 patches per bug (pass@10-style) | LoRA r = 8 |
| Vallecillos-Ruiz et al., ICSME 2026 | Instruct | Repair: HumanEval-Java, Defects4J | **pass@10**, T = 0.1 | Post-training quantization, no fine-tuning |
| PAFT (Yang et al., 2026) | not stated | Repair: Defects4J, HumanEval-Java | **Sampled** Pass@1, T = 0.7 | 4-bit NF4 QLoRA, r = 32 |

**Summary of the evidence:**
1. Greedy Pass@1 is standard for DeepSeek-Coder on code **generation**, but that is **not my task**.
2. On **repair**, nobody I found uses greedy Pass@1 with DeepSeek-Coder-6.7B. Repair papers use beam search (CodeXGLUE), sampled Pass@1 (HumanEvalFix, PAFT) or pass@10 (Java repair benchmarks I don't use).
3. There *is* a published number for my exact model on my exact benchmark (StarCoder2: HumanEvalFix avg 42.1, sampled protocol). It can serve as a sanity check for my untuned baseline M0.

## 1.3 Why greedy is still attractive (the methodological case)

1. **Clean comparisons.** My research questions compare 16 models with each other. Greedy is deterministic, so any difference between models comes from bit-width or rank, not random sampling luck. That supports **paired tests** on the same bugs: McNemar for pass/fail, Wilcoxon for continuous scores.
2. **Clean resource measurements.** Beam search and sampling change GPU memory (KV cache), latency and energy. The resource RQ needs one fixed, declared decoding setup.
3. **Cost and time.** One answer per bug is the cheapest option. Thesis deadline is ~2026-11-25, and the GPU quota is still pending.
4. **Realistic use case.** A developer usually looks at one suggested fix.

## 1.4 Why greedy alone is risky

1. **No precedent on repair benchmarks:** an examiner may ask why I didn't follow the benchmarks' own protocols.
2. **Comparability with published numbers is lost on both benchmarks.**
3. **Robustness:** an examiner may ask whether my conclusions (e.g. "4-bit ≈ 16-bit") only hold under greedy decoding.
4. **Precision:** per language, HumanEvalFix has only 164 bugs. One greedy pass/fail per bug gives a 95% confidence interval of about **±7–8 percentage points per language** and **±3 points pooled** over all 984. Small differences between bit-widths may not be detectable. *(This is a statistics issue whatever decoding is chosen. The planned fix is **equivalence testing** (e.g. TOST with a pre-defined margin such as ±3 points) so that "no difference" can be stated properly.)*

## 1.5 The options

| Option | What it means | Extra GPU time (rough estimate, to be measured in the dry run) | Pros | Cons |
|---|---|---|---|---|
| **A. Greedy everywhere** | Keep the current pipeline; state it as a deliberate deviation in the threats to validity | 0 | Simplest, cheapest, deterministic | No comparability; robustness question unanswered |
| **B. Benchmark conventions for all 16 models** | Beam 5 on CodeXGLUE + sampled Pass@1 (n = 20, T = 0.2) on HumanEvalFix | Large: likely more than doubles the evaluation budget, plus 20× the test executions on CPU | Fully comparable | "Very big"; threatens the deadline |
| **C. Hybrid (assistant's recommendation)** | Greedy for all 16 models (main results), **plus** beam 5 / sampled Pass@1 on **4 models only**: M0, M2, M5, M8 (the bit-width comparison at r = 16, seed 42) | ~30–45 GPU-h (~$35–55); ~half that with n = 10 | Main results stay clean and deterministic; one extra table answers the robustness question; M0 can be checked against the published 42.1 | Moderate extra cost and one more table; needs a pre-set cut-off date in case time runs short |

Rejected earlier: adding **pass@10**. It is the convention for Java repair benchmarks I don't use, not for HumanEvalFix or CodeXGLUE, and I considered it too big.

---

# PROBLEM 2: Trained on Java only. What does a 6-language evaluation measure?

## 2.1 The problem in plain words

The adapters are trained only on **CodeXGLUE Java**. Yet I evaluate on **HumanEvalFix in 6 languages**. How can a model trained on Java be tested on Python, Go or Rust?

**Why it is technically possible:** DeepSeek-Coder was pre-trained on a large multi-language code corpus and can already fix bugs in all six languages **before** any fine-tuning. The StarCoder2 paper reports the untuned Instruct model at a **42.1% average Pass@1** on HumanEvalFix. My LoRA training on Java does **not teach** Python repair. It **changes a model that already has that skill.**

**So HumanEvalFix measures what happens to that existing skill after fine-tuning:**
- **Transfer:** learning to repair Java also helps other languages → scores go **up** vs M0.
- **No effect:** scores stay about the same as M0.
- **Forgetting** ("catastrophic forgetting"): narrow Java training damages the general ability → scores go **down** vs M0.

The thesis-relevant question is **whether bit-width and rank change how much transfer or forgetting happens**. For example, does 4-bit training forget more than 16-bit? Does rank 32 forget more than rank 8?

## 2.2 Even HumanEvalFix's Java tasks are not "the same task" as training

| | CodeXGLUE (training) | HumanEvalFix Java |
|---|---|---|
| Identifiers | Abstracted: `VAR_1`, `METHOD_1` | Real names |
| Input | One method, space-tokenised | Whole program with imports |
| Length | ≤ 512 tokens | Mean 582 tokens; 102 of 164 tasks over 512 |
| Prompt | One fixed training instruction | A different instruction, with example tests included |

So the six languages split into two kinds of shift:
1. **Java column:** only the *format* changes (same language).
2. **Python, C++, Go, Rust, JS:** both the *format* and the *language* change.

This lets the thesis separate "harm from format change" from "harm from language change".

## 2.3 The framing problem I found

An earlier note said HumanEvalFix Pass@1 "replaces Pass@k in RQ1". That mixes up two different things:

| Research question | Measures | Metric |
|---|---|---|
| **RQ1:** in-task repair quality | What the adapters were trained for | CodeXGLUE Exact Match, CodeBLEU, syntax validity |
| **RQ1b:** generalisation and forgetting | What happens to the model's general repair skill | HumanEvalFix Pass@1, **compared against M0** |

**Honest consequence:** the thesis has **no unit-test-based correctness check on the training task itself**, because CodeXGLUE has no tests. This must be stated as a limitation.

## 2.4 Risks

1. **Fine-tuned models may score *below* M0 on HumanEvalFix.** There is precedent: Machácek et al. (ICSME 2025) found fine-tuning **lowered** repair scores, which they attributed to differences between the training data and the benchmarks. Framed as RQ1b, that would be a legitimate finding, not a failure. Related literature: Biderman et al. (TMLR 2024), "LoRA Learns Less and Forgets Less": LoRA keeps more of the base model's out-of-domain ability than full fine-tuning.
2. **Floor effect: every model could score near zero.** If the adapters learn to always output `VAR_1`-style code, or truncated or abstracted answers, all fine-tuned models would score about 0%. Bit-width and rank then couldn't be compared at all. **Planned mitigation:** right after the GPU dry run, test M0 and the first adapter on a sample of HumanEvalFix and inspect *why* they fail (compile error vs wrong logic vs abstracted tokens) before running the full matrix.

## 2.5 The options

| Option | What it means | Cost | Pros | Cons |
|---|---|---|---|---|
| **(a) Keep the design, reframe (assistant's recommendation)** | RQ1 = CodeXGLUE (in-task); RQ1b = HumanEvalFix (generalisation / forgetting vs M0); early floor-effect check | ~0 extra | No new data or training; forgetting × quantization is an interesting question in itself | No test-based correctness on the training task (stated as a limitation) |
| **(b) Add multilingual repair training data** | e.g. CommitPackFT (used by OctoPack to train for HumanEvalFix) | Large: new data pipeline, retrain all models | HumanEvalFix becomes partly in-distribution | Changes the study's question; much bigger; deadline risk |
| **(c) Find a Java repair test set with unit tests in CodeXGLUE's abstracted style** | Replace or complement HumanEvalFix | Unknown | Would give in-task functional correctness | No such dataset known; a real-code test set would still be a format mismatch |

---

# How the two problems connect

- If HumanEvalFix is reframed as a **generalisation / forgetting** measure (Problem 2, option a), its job is to compare each fine-tuned model with **M0 under the same protocol**. Comparability with published numbers then matters mainly for **M0** (to validate the harness against the StarCoder2 42.1), not for every model. That supports a **small** extra protocol run (Problem 1, option C) rather than a big one.
- The **floor-effect check** (Problem 2) and the **M0 harness check** (Problem 1) can run together in the first GPU session, before the full matrix.

---

## Facts I'm still unsure about

- Exact GPU-hours per model: estimates only until the 50-step dry run on the GPU.
- Whether my supervisor or proposal expects comparability with published numbers. The original plan said "Pass@k", which I later changed to Pass@1 on HumanEvalFix.
- How close greedy Pass@1 and sampled Pass@1 (T = 0.2) usually are. They are typically close at low temperature, but not identical.
- Prompt differences: my HumanEvalFix prompt wording differs from OctoPack/StarCoder2's. I also exclude 4 defective tasks (CPP/38, CPP/23, Go/95, Rust/32) in a "clean" subset.
- How strongly the fine-tuned models will forget on HumanEvalFix (unknown until the first adapter is evaluated).

## Questions I want to explore

**Problem 1 (decoding):**
1. For a master's thesis whose research questions are *internal comparisons* (bit-width × rank), is a declared deviation to greedy decoding (Option A) acceptable, or will examiners expect benchmark-standard protocols?
2. Is Option C's "protocol robustness check on a subset" a recognised and defensible practice in empirical SE / ML papers?
3. Should CodeXGLUE use beam 5, given that beam search also inflates the inference memory and latency I'm measuring?
4. Is n = 10 samples an acceptable substitute for n = 20 if GPU time is tight?
5. How do equivalence testing (TOST) and McNemar's test work for comparing my models, and what margin is reasonable?

**Problem 2 (cross-language evaluation):**
6. Is it methodologically sound to train on one language and evaluate on others, if it is framed as a generalisation / forgetting study?
7. How should RQ1 and RQ1b be worded so the thesis doesn't overclaim functional correctness?
8. How do papers usually measure catastrophic forgetting after fine-tuning (relative to the base model)? Is "score difference vs M0" enough?
9. Is there any known Java repair dataset with unit tests that matches CodeXGLUE's abstracted format?
10. What would a reviewer say about the floor-effect risk, and is the early check enough?

**Both:**
11. How should each problem be described in the "Threats to Validity" section?

## Sources to load alongside this briefing

- CodeXGLUE refinement baseline: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement
- StarCoder2 paper (HumanEvalFix protocol + DeepSeek-Coder-6.7B-Instruct numbers): https://arxiv.org/abs/2402.19173
- OctoPack (introduced HumanEvalFix and CommitPackFT): https://arxiv.org/abs/2308.07124
- DeepSeek-Coder technical report: https://arxiv.org/abs/2401.14196
- Magicoder: https://arxiv.org/abs/2312.02120
- Li et al., ASE 2024: https://arxiv.org/abs/2406.05639
- Machácek et al., ICSME 2025: https://arxiv.org/abs/2507.19909
- Vallecillos-Ruiz et al., ICSME 2026: https://arxiv.org/abs/2606.27205
- PAFT: https://arxiv.org/abs/2604.03113
- Biderman et al., "LoRA Learns Less and Forgets Less" (TMLR 2024): https://arxiv.org/abs/2405.09673
- Codex paper (defines the unbiased pass@k estimator): https://arxiv.org/abs/2107.03374
