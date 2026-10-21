# Resource-Aware Large Code Models: Quantized PEFT for Automated Program Repair

Replication package for an MEng thesis (GISMA University of Applied Sciences, Berlin): an empirical study of how **bit-width** (4-bit NF4 QLoRA, 8-bit, 16-bit bf16 LoRA) and **LoRA rank** (r ∈ {8, 16, 32}) trade off **correctness, GPU memory, speed and energy** when fine-tuning a large code model for Java bug fixing.

> 🚧 **Work in progress.** The full pipeline (data, training, inference, metrics, test harness, run scripts) is implemented and smoke-tested on CPU. GPU runs are pending. Results will be added as runs complete.

## Research questions

| RQ | Question |
|---|---|
| **RQ1** Correctness | How do bit-width and LoRA rank change the test-verified repair rate (pass@1 primary, pass@10) on HumanEvalFix-Java and HumanEval-Java, and Exact Match, CodeBLEU and syntax validity on held-out Megadiff fixes, relative to the 16-bit adapters and the zero-shot model (M0)? |
| **RQ2** Memory | What is the peak GPU VRAM during training and inference across bit-widths and ranks, and can test-verified repair quality be reached within a ≤ 24 GB consumer GPU? |
| **RQ3** Efficiency | How do precision and rank affect training throughput, time-to-first-token, generation throughput and energy? |
| **RQ4** Decision framework | Which configurations are Pareto-optimal between resource cost (VRAM, time, energy) and pass@1, for developer workstations and CI pipelines? |

## Experimental design

- **Model:** `deepseek-ai/deepseek-coder-6.7b-instruct`
- **Training + in-task test:** Megadiff single-function (Monperrus et al., 2021): real Java commit changes, one method each. After dropping trivial (whitespace/comment-only), duplicate and benchmark-overlapping pairs, and pairs over **768 tokens** (never truncated), **20,106** pairs remain. They are split **by file path** (all changes to one file land in one split) into 10,000 train / 1,000 validation / 1,000 test (seed 42). The test split gives Exact Match, CodeBLEU and syntax validity (greedy decoding).
- **Test-based evaluation (Java only):** **HumanEvalFix-Java** (Muennighoff et al., ICLR 2024; 164 bugs) and **HumanEval-Java** (Jiang et al., ICSE 2023; 163 bugs, 162 clean), both with unit tests. **20 samples per bug** (temperature 0.2, top-p 0.95); **pass@1** (primary) and **pass@10** (secondary) from the same samples, using the unbiased estimator of Chen et al. (2021).
- **Hardware:** one 24 GB NVIDIA GPU, with the same GPU model for every run

| | r = 8 | r = 16 | r = 32 |
|---|---|---|---|
| **4-bit NF4 (QLoRA)** | M1 | M2 | M3 |
| **8-bit int8** | M4 | M5 | M6 |
| **16-bit bf16 (LoRA)** | M7 | M8 | M9 |

Plus **M0** (untuned base model, **zero-shot**) and a **naive-copy** reference (output = buggy input).

**Seed variance:** the r = 16 column (**M2, M5, M8**) is trained with **3 seeds (42, 43, 44)**, so the bit-width comparison comes with mean ± SD and paired statistical tests. Other cells use seed 42. The data split is fixed (seed 42) for every run.
Fixed: α = 2r, all linear layers adapted (q, k, v, o, gate, up, down), LoRA dropout 0.05, LR 2e-4 (cosine, 3% warmup), batch 4 × grad-accum 4, 3 epochs, max length 768, paged AdamW (32-bit) for every bit-width, grad-norm clip 0.3, seed 42.

| r | Trainable params | % of 6.74B | Adapter size (bf16) |
|---|---|---|---|
| 8 | 19,988,480 | 0.297 % | 38 MB |
| 16 | 39,976,960 | 0.593 % | 76 MB |
| 32 | 79,953,920 | 1.186 % | 152 MB |

## Findings so far (data and metric validation)

1. **The training data changed from CodeXGLUE to Megadiff** ([log #34–#39](docs/RESEARCH_LOG.md)). CodeXGLUE's abstracted identifiers (`VAR_1`, `METHOD_2`) don't match real code, and it has no tests. Megadiff is real Java, and the evaluation is Java-only, so training and test-based evaluation share one language.
2. **Length cap 768, not truncation.** Megadiff methods are long (median 1,236 tokens chat-formatted). Pairs above 768 tokens are dropped, so no target is cut off; the splits average ~490 tokens (p95 740).
3. **Contamination guard:** 84 Megadiff pairs share a 20-token sequence with a HumanEvalFix-Java or HumanEval-Java method. Inspected matches are generic idioms, but all 84 are dropped.
4. **Naive-copy baseline on the Megadiff test split: 0.876 CodeBLEU, 0% Exact Match.** Fixes are small edits, so similarity metrics reward copying. CodeBLEU is always reported next to this reference. The syntax-validity ceiling (ground-truth fixes that tree-sitter-java parses) is 99.8%.
5. **Both test harnesses are validated:** reference fixes pass 100% and buggy code 0% on the clean sets of HumanEvalFix-Java (164) and HumanEval-Java (162). HumanEval-Java needs **JDK 8** (Nashorn, `javax.xml.bind`); its `ISCUBE` task has no bug (excluded), `GET_ROW` has swapped package lines (normalised), and `DO_ALGEBRA` runs serially (times out under parallel load).
6. **Earlier HumanEvalFix work in 6 languages** ([log #22](docs/RESEARCH_LOG.md)) stays in the harness, but only Java is evaluated: a Java-only adapter evaluated on five other languages would measure language transfer, not repair (log #34).

Full details, including every decision and its rationale: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## Repository layout

```text
src/
  data_loader.py   Megadiff cleaning + file-path split, HumanEvalFix + HumanEval-Java loading, chat-template prompts, loss masking, length report
  metrics_se.py    tree-sitter syntax validity, Exact Match, CodeBLEU
  execution.py     sandboxed test runner + unbiased pass@k (HumanEval-Java; HumanEvalFix in Python, Java, C++, Go, Rust, JS)
  profiler.py      peak VRAM, throughput, NVML energy, TTFT, trainable-parameter ratio, CSV logging
  trainer.py       PEFT training for one matrix cell (--bits 4|8|16, --rank, --alpha) with profiling + CSV log
  evaluator.py     inference for M0–M9: greedy (Megadiff) or n = 20 samples (test-based) + greedy batch-size-1 latency probe
scripts/
  run_matrix.sh    train M1–M9 (or selected cells; DRY_RUN=50 for dry runs); resumable
  evaluate_all.sh  predictions + scores for M0–M9 on Megadiff, HumanEvalFix-Java and HumanEval-Java
analysis/
  feasibility/     Megadiff statistics and HumanEval-Java sanity check behind the data switch (log #38)
                   (planned) Pareto plots and LaTeX tables
docs/
  RESEARCH_LOG.md      chronological log of decisions, actions and findings
  LEARNING_PATH.md     background reading and exercises behind every design choice
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install the CUDA build of torch on GPU hosts

# 1. HumanEval-Java source (BSD-3), pinned commit
git clone https://github.com/lin-tan/clm data/clm && git -C data/clm checkout 662769e515cfd2a4980d7d4e98f4739ef6265b45
tar xzf data/clm/humaneval-java/humaneval-java.tar.gz -C data/clm   # the repo ships HumanEval-Java as a tarball

# 2. Build all data files (no GPU; ~7 min, mostly tokenising Megadiff for the length filter)
python src/data_loader.py --output_dir data
#    -> megadiff_{train,validation,test}.jsonl, humanevalfix_java.jsonl, humaneval_java.jsonl

# 3. Sanity-check the metrics on the ground truth (expects EM 100, CodeBLEU 1.0, validity 99.8%)
python src/metrics_se.py data/megadiff_test.jsonl --sanity

# 4. Check the test harnesses (toolchain: apt install -y default-jdk openjdk-8-jdk)
export HEJ_JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64      # HumanEval-Java needs JDK 8
python src/execution.py data/humanevalfix_java.jsonl data/humaneval_java.jsonl --sanity canonical  # expect pass_at_1_clean_pct = 100
python src/execution.py data/humanevalfix_java.jsonl data/humaneval_java.jsonl --sanity buggy      # expect pass_at_1_clean_pct = 0
```

The harness still supports HumanEvalFix in Python, C++, Go, Rust and JavaScript (`python src/data_loader.py --hef_languages python java cpp go rust js`; toolchains: `g++ libboost-dev libssl-dev golang nodejs cargo`), but the study evaluates Java only.
The first Go/Rust run downloads `testify` (Go proxy) and `rand 0.7`, `regex`, `md5` (crates.io) into `~/.cache/peft-code-eval` (override with `HEF_CACHE`). JavaScript/162's `js-md5` package is replaced by a local shim built on Node's `crypto`.

⚠️ `execution.py` runs model-generated code. Score real predictions on a disposable machine or container.

Training one matrix cell (GPU):

```bash
python src/trainer.py --bits 4 --rank 8 --run_name M1 --output_dir results/runs/M1                   # full run
python src/trainer.py --bits 4 --rank 8 --run_name dry --output_dir results/runs/dry --max_steps 50  # dry run
```

Each run writes `adapter/`, `run_config.json` (all arguments, fixed hyperparameters, library versions, results) and appends one row to `results/train_log.csv` (peak VRAM allocated/reserved, static VRAM after load, duration with and without evaluation, samples/s, tokens/s, energy, losses, GPU name).

Full experiment on the GPU machine (bash ≥ 4):

```bash
DRY_RUN=50 bash scripts/run_matrix.sh M1   # 1. dry run: checks 4-bit loading, VRAM and energy logging
bash scripts/run_matrix.sh                 # 2. train M1–M9 (resumable: finished cells are skipped)
SEEDS="43 44" bash scripts/run_matrix.sh M2 M5 M8      # 3. extra seeds for the r = 16 column
bash scripts/evaluate_all.sh               # 4. predictions + scores for M0–M9
bash scripts/evaluate_all.sh M2_s43 M2_s44 M5_s43 M5_s44 M8_s43 M8_s44
```

Outputs: `results/train_log.csv`, `results/eval_log.csv` (throughput, TTFT, inference VRAM, energy, Megadiff EM/CodeBLEU/syntax), `results/predictions/*.jsonl` (a `predictions` list per bug), `results/scores/*_tests.json` (pass@1 and pass@10 per benchmark, all tasks and clean set).

Version note: `codebleu==0.7.0` requires `tree-sitter` 0.22.x, which in turn requires `tree-sitter-java` 0.21.x. The pins in `requirements.txt` are tested together.

## Data

Nothing is redistributed here; `src/data_loader.py` regenerates every file deterministically.

| Dataset | Source | Licence | Use |
|---|---|---|---|
| Megadiff single-function | Hugging Face `ASSERT-KTH/megadiff-single-function`; Monperrus et al., *Megadiff: A Dataset of 600k Java Source Code Changes Categorized by Diff Size* (arXiv 2108.04631, 2021) | none stated (code from open-source projects under their own licences); used for research with citation | training, validation, in-task test |
| HumanEvalFix-Java | Hugging Face `bigcode/humanevalpack`; Muennighoff et al., *OctoPack* (ICLR 2024) | MIT | test-based evaluation |
| HumanEval-Java | GitHub `lin-tan/clm`; Jiang et al., *Impact of Code Language Models on Automated Program Repair* (ICSE 2023) | BSD-3-Clause | test-based evaluation |

## License

Code: MIT (see [`LICENSE`](LICENSE)). The dataset and model are subject to their own licenses.
