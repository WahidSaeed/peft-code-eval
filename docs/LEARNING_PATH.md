# Learning Path — what to understand before the GPU runs start

**Goal:** you can explain and defend every design choice in this thesis yourself, in front of your supervisor, without needing the assistant.
**Time budget:** ~5 focused days while the AWS GPU quota is pending (created 2026-10-20).
Each module has: *why it matters here* → *learn from* → *do this* → *self-check* (you're ready when you can answer without notes).

Priority: 🔴 essential before the first run · 🟠 essential before writing results · 🟢 deepen when time allows

---

## Module 1 — PyTorch & GPU memory basics 🔴 (½ day)

**Why here:** RQ2 is about GPU memory. You must know *what* fills VRAM and what `max_memory_allocated` vs `max_memory_reserved` mean (`src/profiler.py`).

**Learn from**
- PyTorch "Learn the Basics" tutorial (pytorch.org → Tutorials → Learn the Basics): tensors, autograd, training loop
- PyTorch docs: "CUDA semantics → Memory management" (caching allocator, allocated vs reserved)
- Hugging Face Transformers docs: "Efficient training on a single GPU" (gradient checkpointing, gradient accumulation, mixed precision)

**Key concepts:** dtype (fp32 / fp16 / **bf16**), what's in VRAM during training (weights + gradients + optimizer states + activations), gradient checkpointing, gradient accumulation (our effective batch 16 = 4 × 4).

**Do this:** by hand, compute weight memory for 6.7B parameters in fp32, bf16, int8, 4-bit. *(Answer: 26.8 / 13.4 / 6.7 / 3.35 GB, i.e. 4, 2, 1, 0.5 bytes per weight.)* Then explain why full fine-tuning with AdamW needs far more (Adam keeps 2 extra fp32 states per trainable parameter).

**Self-check**
1. Why can't we fully fine-tune 6.7B on a 24 GB GPU, but can train LoRA on it?
2. What's the difference between allocated and reserved memory, and which one do we report as "peak VRAM"? Why report both?
3. What does gradient checkpointing trade away to save memory?

---

## Module 2 — Transformers & LLM generation 🔴 (1 day)

**Why here:** DeepSeek-Coder is a decoder-only transformer. TTFT, tokens/s and the KV-cache (RQ3) only make sense if you know how generation works.

**Learn from**
- Jay Alammar, "The Illustrated Transformer" (blog)
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (YouTube, ~2h). The best investment on this list.
- Hugging Face LLM Course (huggingface.co/learn), chapters 1–3: tokenizers, models, fine-tuning

**Key concepts:** tokens & tokenizer, attention, decoder-only / causal LM, prefill vs decode phase, **KV-cache**, greedy decoding, chat templates (instruct models), EOS token.

**Do this:** in `src/data_loader.py`, find where the chat template is applied and where prompt tokens get label `-100`. Explain to yourself why we mask the prompt.

**Self-check**
1. What is Time-to-First-Token, and which phase (prefill or decode) dominates it?
2. Why does the KV-cache grow during generation, and why does it matter for inference memory?
3. Why do we use greedy decoding (not sampling) for evaluation?
4. What would go wrong if we did *not* mask the prompt tokens in the loss?

---

## Module 3 — Fine-tuning & LoRA 🔴 (1 day)

**Why here:** the "PEFT" in the title. Our variables are rank r ∈ {8, 16, 32}, α = 2r, target modules = all linear layers.

**Learn from**
- **LoRA paper:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022 (arXiv 2106.09685). Read Sections 1, 4, 7.
- Sebastian Raschka, "Practical Tips for Finetuning LLMs Using LoRA" (Ahead of AI magazine)
- Hugging Face PEFT documentation: LoraConfig (`r`, `lora_alpha`, `target_modules`, `lora_dropout`)

**Key concepts:** W' = W + (α/r)·BA, frozen base weights, rank as adapter capacity, α scaling, which modules to adapt (q, k, v, o, gate, up, down), trainable-parameter ratio.

**Do this, the LoRA parameter count exercise:**
Open `deepseek-ai/deepseek-coder-6.7b-instruct` → *Files* → `config.json` on Hugging Face. Note `num_hidden_layers`, `hidden_size`, `intermediate_size`.
A LoRA adapter on a linear layer (in → out) adds r·(in + out) parameters. Compute the total for all 7 linear modules × all layers, for r = 8, 16, 32.
*(Expected if the config is 32 layers / 4096 / 11008: ≈ 20.0M, 40.0M, 80.0M, i.e. ≈ 0.3 / 0.6 / 1.2 % of 6.7B. We'll confirm with `trainable_parameters()` in `src/profiler.py` on the GPU.)*

**Self-check**
1. Why does LoRA save memory during training, even though the full model is still loaded?
2. What does increasing r change? And α?
3. Why do we keep α = 2r across all configurations?

---

## Module 4 — Quantization: 8-bit, 4-bit NF4, QLoRA 🔴 (1 day)

**Why here:** the core independent variable (bit-width). You must be able to explain M1–M3 vs M4–M6 vs M7–M9.

**Learn from**
- **QLoRA paper:** Dettmers et al., NeurIPS 2023 (arXiv 2305.14314). Read Sections 1–3: NF4, double quantization, paged optimizers.
- **LLM.int8() paper:** Dettmers et al., 2022 (arXiv 2208.07339). Skim: outlier features, mixed-precision decomposition.
- Hugging Face blog: "A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes"
- Hugging Face blog: "Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA"

**Key concepts:** quantize / dequantize, block-wise quantization, NF4 vs int8, compute dtype (bf16) vs storage dtype, why 4-bit weights must be dequantized for each matmul (→ speed cost), paged AdamW.

**Self-check**
1. In QLoRA, which parts are 4-bit and which are bf16? Are the LoRA adapters quantized?
2. Why might 8-bit training be *slower* than 16-bit (our threat-to-validity note in the research log)?
3. Why is NF4 "information-theoretically optimal" for normally distributed weights (in one sentence)?

---

## Module 5 — Automated Program Repair & our dataset 🔴 (½ day)

**Why here:** the domain. Your supervisor will ask where the data comes from and what it can and can't measure.

**Learn from**
- Tufano et al., "An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation", ACM TOSEM 2019. **The origin of the CodeXGLUE refinement data** and of its abstraction (`VAR_1`, `METHOD_1`) and small/medium split.
- Lu et al., "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation", 2021 (arXiv 2102.04664). The Code Refinement section and its baseline table, including **naive copy**.
- Monperrus, "Automatic Software Repair: a Bibliography", ACM Computing Surveys 2018. APR landscape: test-based repair, patch overfitting, plausible vs correct patches.

**Do this:** open `data/medium_test.jsonl`, read 10 examples, and classify each bug (operator change, wrong method call, missing null check…). Compare with the categories in Tufano et al.

**Self-check**
1. Why is Exact Match a *lower bound* on true repair ability?
2. Why can't we measure Pass@k on this dataset? (Ties to open decision #1 in the research log.)
3. What is the "medium" split, and why 50–100 tokens?

---

## Module 6 — Evaluation metrics: EM, BLEU, CodeBLEU, ASTs 🟠 (½ day)

**Why here:** RQ1. Two of our week-1 findings (tokenisation artefacts, naive-copy 0.887) are metric issues.

**Learn from**
- Ren et al., "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis", 2020 (arXiv 2009.10297)
- Papineni et al., "BLEU", ACL 2002. Just the idea of n-gram precision and the brevity penalty.
- Tree-sitter documentation + **online playground** (tree-sitter.github.io). Paste a Java method and look at the tree and ERROR nodes.

**Do this (runs on your Mac, no GPU):**
```bash
cd peft-code-eval
python src/metrics_se.py data/medium_test.jsonl --sanity
```
Then paste `foo ( VAR_3 class )` and `foo ( VAR_3.class )` into the tree-sitter playground and see the difference yourself.

**Self-check**
1. Name CodeBLEU's 4 components and what each captures.
2. Why does naive copy score 0.887 CodeBLEU but 0% EM?
3. Why must `detokenize()` be applied to predictions *and* targets identically?

---

## Module 7 — Efficiency & energy measurement 🟠 (2 hours)

**Why here:** RQ3 and the Green SE angle.

**Learn from**
- Wei et al., "Towards Greener Yet Powerful Code Generation via Quantization: An Empirical Study", ESEC/FSE 2023. Their measurement method is our comparison point (Tier A).
- NVIDIA NVML docs: `nvmlDeviceGetTotalEnergyConsumption`, `nvmlDeviceGetPowerUsage`
- `nvidia-smi` basics (you'll use it constantly on the GPU machine)

**Self-check**
1. Why do we measure GPU energy only, and what does that leave out (CPU, RAM, cooling)? → threat to validity
2. Why must all runs use the same GPU model?

---

## Module 8 — Empirical SE methodology & statistics 🟠 (1 day, can overlap with training week)

**Why here:** this is what makes it a TOSEM/EMSE-quality study rather than a benchmark report.

**Learn from**
- Arcuri & Briand, "A practical guide for using statistical tests to assess randomized algorithms in software engineering", ICSE 2011
- ACM SIGSOFT **Empirical Standards** (github.com/acmsigsoft/EmpiricalStandards), especially "Benchmarking" and "Experiments"
- Wohlin et al., *Experimentation in Software Engineering* (Springer). The threats-to-validity chapter (construct, internal, external, conclusion).

**Key concepts:** paired tests on the same test set (McNemar for EM / syntax yes-no outcomes; Wilcoxon signed-rank for per-sample scores), effect size (Cliff's δ, Vargha-Delaney A12), multiple-comparison correction, seeds and variance, Pareto front.

**Self-check**
1. Why McNemar rather than a chi-square test for comparing two models' EM on the same 1,000 bugs?
2. What's the difference between statistically significant and practically meaningful?
3. List one construct, one internal and one external validity threat in *this* study.

---

## Module 9 — Cloud & tooling for the GPU runs 🔴 (½ day, do before the quota arrives)

**Why here:** you'll operate a paid machine. Mistakes cost credit.

**Learn from**
- AWS docs: EC2 "Get started", instance lifecycle (**stop vs terminate**), EBS volumes, security groups, key pairs
- AWS **Deep Learning AMI** (GPU PyTorch). Comes with CUDA drivers preinstalled.
- AWS Budgets ("Create a cost budget") and IAM best practices (don't use root day-to-day)
- MIT "The Missing Semester of Your CS Education": shell, **tmux**, ssh, git lectures

**Do this now (no quota needed):**
1. Create an IAM admin user with MFA and stop using root.
2. Create a monthly AWS Budget of $150 with an email alert at 80%.
3. Generate an SSH key pair in eu-central-1.
4. Practise `tmux` locally: start a session, detach, reattach. Training runs for hours and must survive an SSH disconnect.

**Self-check**
1. What keeps costing money after you *stop* an instance? (EBS storage.) And after *terminate*?
2. How will you copy `results/` back to your Mac? (`scp` / `rsync`)

---

## Suggested schedule (while the quota is pending)

| Day | Morning | Afternoon |
|---|---|---|
| 1 | Module 1 + Module 9 "do this now" | Module 2 (Karpathy video) |
| 2 | Module 2 self-check, Module 3 (LoRA paper) | LoRA parameter-count exercise |
| 3 | Module 4 (QLoRA + blogs) | Module 5 (Tufano, CodeXGLUE, read 10 examples) |
| 4 | Module 6 (CodeBLEU, run metrics locally) | Module 7 |
| 5 | Module 8 | Re-read the plan; answer every self-check aloud; note gaps |

## Map: concept → where it lives in the code

| Concept | File / function |
|---|---|
| Chat template, prompt masking, seq len 512 | `src/data_loader.py` → `chat_prompt`, `tokenize_for_training` |
| Peak VRAM, throughput, energy | `src/profiler.py` → `GPUProfiler`, `EnergyMeter` |
| TTFT, decode tokens/s | `src/profiler.py` → `measure_generation` |
| Trainable-parameter ratio | `src/profiler.py` → `trainable_parameters` |
| Syntax validity, detokenisation, EM, CodeBLEU | `src/metrics_se.py` |
| Bits / rank / alpha | `src/trainer.py` (next) |
| Every decision and why | `docs/RESEARCH_LOG.md` |

**Tip:** when you finish a module, ask the assistant to quiz you on its self-check questions, or to explain the matching code line by line.
