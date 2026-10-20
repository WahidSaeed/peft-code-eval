# Resource-Aware Large Code Models: Quantized PEFT for Automated Program Repair

Replication package for an MEng thesis (GISMA University of Applied Sciences, Berlin): an empirical study of how **bit-width** (4-bit NF4 QLoRA, 8-bit, 16-bit bf16 LoRA) and **LoRA rank** (r ∈ {8, 16, 32}) trade off **correctness, GPU memory, speed and energy** when fine-tuning a large code model for Java bug fixing.

> 🚧 **Work in progress.** Data pipeline and metrics are implemented and validated. Training and evaluation scripts are in progress. Results will be added as runs complete.

## Research questions

| RQ | Question |
|---|---|
| **RQ1** Correctness | How much does bit-width compression degrade syntax validity, Exact Match and CodeBLEU of generated patches? |
| **RQ2** Memory | What is the peak GPU VRAM during training and inference across ranks, and does high-fidelity repair fit a ≤ 24 GB consumer GPU? |
| **RQ3** Efficiency | How do precision and rank affect training throughput, time-to-first-token, generation throughput and energy? |
| **RQ4** Decision framework | Which configuration is Pareto-optimal for developer workstations and CI pipelines? |

## Experimental design

- **Model:** `deepseek-ai/deepseek-coder-6.7b-instruct`
- **Training + in-distribution test:** CodeXGLUE Code Refinement (Java), *medium* split. Seeded random subsample (seed 42) of 10,000 train / 1,000 validation / 1,000 test pairs from 52,364 / 6,546 / 6,545.
- **Out-of-distribution test (evaluation only):** HumanEvalFix from HumanEvalPack (Muennighoff et al., ICLR 2024), 164 hand-written bugs in each of **6 languages** (Python, Java, C++, Go, Rust, JavaScript) with unit tests, giving **Pass@1** and a check of whether quantized adapters generalise beyond abstracted Java.
- **Hardware:** one 24 GB NVIDIA GPU, with the same GPU model for every run

| | r = 8 | r = 16 | r = 32 |
|---|---|---|---|
| **4-bit NF4 (QLoRA)** | M1 | M2 | M3 |
| **8-bit int8** | M4 | M5 | M6 |
| **16-bit bf16 (LoRA)** | M7 | M8 | M9 |

Plus **M0** (untuned base model) and a **naive-copy** reference (output = buggy input).
Fixed: α = 2r, all linear layers adapted, LR 2e-4, batch 4 × grad-accum 4, 3 epochs, max length 512.

## Findings so far (metric validation)

1. **Sequence length 512 is justified:** chat-formatted samples average ~369 tokens (p95 ≈ 455). Only 0.14% of training samples are truncated, and none in validation/test.
2. **CodeXGLUE tokenisation breaks Java syntax.** Operators are split (`- >`, `: :`, `> >`, `< <`) and dots dropped (`X class`, `Outer this`). Only 93.9% of *ground-truth* fixes parsed with tree-sitter. After targeted de-tokenisation (applied identically to predictions and references), **99.9%** parse, which is the metric's ceiling.
3. **The naive-copy baseline scores 0.887 CodeBLEU and 0% Exact Match.** Bugs are small edits, so similarity metrics reward copying. Exact Match is therefore the primary effectiveness metric, and CodeBLEU is always reported next to the naive-copy reference.

4. **The HumanEvalFix test harness is validated in all 6 languages:** reference fixes pass 100% and buggy originals 0% on the clean subset (980 of 984 tasks). Four dataset defects are documented and reported separately: CPP/38 (reference doesn't compile), CPP/23 (tests miss the bug), Go/95 and Rust/32 (flaky tests). The dataset's `entry_point` field disagrees with the actual function name in 17 tasks, so the harness reads names from the code.

Full details, including every decision and its rationale: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## Repository layout

```text
src/
  data_loader.py   CodeXGLUE + HumanEvalFix loading, seeded subsampling, chat-template prompts, loss masking, length report
  metrics_se.py    tree-sitter syntax validity (with de-tokenisation), Exact Match, CodeBLEU
  execution.py     sandboxed test runner for HumanEvalFix Pass@1 (Python, Java, C++, Go, Rust, JS)
  profiler.py      peak VRAM, throughput, NVML energy, TTFT, trainable-parameter ratio, CSV logging
  trainer.py       (in progress) PEFT training with --bits / --rank / --alpha
  evaluator.py     (planned) greedy batch inference
configs/  scripts/  analysis/   (planned) run configs, matrix runner, Pareto plots and LaTeX tables
docs/
  RESEARCH_LOG.md      chronological log of decisions, actions and findings
  LEARNING_PATH.md     background reading and exercises behind every design choice
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install the CUDA build of torch on GPU hosts

# 1. Build the dataset splits (no GPU needed)
python src/data_loader.py --output_dir data

# 2. Sanity-check the metrics on the ground truth (expects EM 100, CodeBLEU 1.0, validity ≈ 99.9%)
python src/metrics_se.py data/medium_test.jsonl --sanity

# 3. HumanEvalFix: build files, then check the harness
#    toolchains (Ubuntu): apt install -y default-jdk g++ libboost-dev libssl-dev golang nodejs cargo
python src/data_loader.py --benchmark humanevalfix --output_dir data
python src/execution.py data/humanevalfix_*.jsonl --sanity canonical   # expect pass_at_1_clean_pct = 100
python src/execution.py data/humanevalfix_*.jsonl --sanity buggy       # expect pass_at_1_clean_pct = 0
```

The first Go/Rust run downloads `testify` (Go proxy) and `rand 0.7`, `regex`, `md5` (crates.io) into `~/.cache/peft-code-eval` (override with `HEF_CACHE`). JavaScript/162's `js-md5` package is replaced by a local shim built on Node's `crypto`.

⚠️ `execution.py` runs model-generated code. Score real predictions on a disposable machine or container.

Version note: `codebleu==0.7.0` requires `tree-sitter` 0.22.x, which in turn requires `tree-sitter-java` 0.21.x. The pins in `requirements.txt` are tested together.

## Data

Datasets are downloaded from the Hugging Face Hub (`google/code_x_glue_cc_code_refinement`, `bigcode/humanevalpack`) and are not redistributed here. `src/data_loader.py` regenerates the exact splits deterministically. The original data comes from Tufano et al., *An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation* (ACM TOSEM 2019), packaged by Lu et al., *CodeXGLUE* (2021).

## License

Code: MIT (see [`LICENSE`](LICENSE)). The dataset and model are subject to their own licenses.
