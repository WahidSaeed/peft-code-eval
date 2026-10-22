"""Phase 0: reproduce Li et al. (ASE 2024) DeepSeek-Coder-6.7B-Base + LoRA on HumanEval-Java (RESEARCH_LOG #63).

Generation follows their released code (zjulgc/llmpeft4apr, inference_and_validation_src/peft_generate_patch.py):
their prompt template, fp16, beam search with 10 beams / 10 returned, max_new_tokens = prompt length + 256,
prompt truncated at 1,024 tokens, each output cut at its last '}'. Scoring uses OUR harness
(src/execution.py): the patch replaces the task's "function range" lines in the buggy class, then the JUnit
tests run. pass@k = a plausible patch among the top k beams, over all 163 tasks (their definition).

    python analysis/phase0/li_reproduce.py generate <out_dir>     # GPU
    python analysis/phase0/li_reproduce.py score <out_dir>        # CPU; runs model-generated code (container only)

Their benchmark file and adapter are downloaded at run time (their repository has no licence file).
"""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BASE = "deepseek-ai/deepseek-coder-6.7b-base"
ADAPTER_REPO, ADAPTER_DIR = "survoltli/llmpeft4apr", "deepseek_coder_6.7b/LoRA"
BENCH_URL = "https://raw.githubusercontent.com/zjulgc/llmpeft4apr/HEAD/benchmarks/humaneval.json"
PUBLISHED = {"pass_at_1": 70 / 163, "pass_at_10": 109 / 163}  # their released result_look file

PROMPT_TEMPLATE = ("You are an exceptionally intelligent coding assistant that consistently delivers accurate and "
                   "reliable responses to user instructions.\n\n@@ Instruction\n{instruction}\n\n@@ Response\n")
BENCHMARK_PROMPT = ("Write a solution to the following coding problem:\nThe input is buggy code, which bug lines start "
                    "from '// buggy lines start' and end at '// buggy lines end'. Please fix the follwing code. \n{problem}")


def benchmark():
    with urllib.request.urlopen(BENCH_URL) as r:
        return json.load(r)["data"]


def generate(out_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = benchmark()
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token_id = 32018  # as in their code ("<pad>")
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, ADAPTER_REPO, subfolder=ADAPTER_DIR)
    model.eval()
    eos = tok.convert_tokens_to_ids(tok.eos_token)
    out = {}
    for i, (name, task) in enumerate(data.items(), 1):
        prompt = PROMPT_TEMPLATE.format(instruction=BENCHMARK_PROMPT.format(problem=task["input"]))
        ids = tok(prompt, truncation=True, max_length=1024, padding=False, return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            gen = model.generate(input_ids=ids, max_new_tokens=len(ids[0]) + 256, num_beams=10,
                                 num_return_sequences=10, pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                 do_sample=False)
        cands = [tok.decode(g[len(ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in gen]
        out[name] = {"function_range": task["function range"], "outputs": [c[:c.rfind("}") + 1] for c in cands]}
        print(f"{i}/{len(data)} {name}", flush=True)
        (out_dir / "li_lora_humaneval_java_outputs.json").write_text(json.dumps(out, indent=1))


def score(out_dir):
    from execution import execute

    out_dir = Path(out_dir)
    gens = json.loads((out_dir / "li_lora_humaneval_java_outputs.json").read_text())
    recs = {r["name"]: r for r in map(json.loads, open(ROOT / "data" / "humaneval_java.jsonl"))}
    jobs = []
    for name, g in gens.items():
        start, end = (int(part.split(",")[0]) for part in g["function_range"].split("-"))
        rec = {**recs[name], "begin": start, "end": end}
        jobs += [(name, rank, patch, rec) for rank, patch in enumerate(g["outputs"])]
    with ThreadPoolExecutor(8) as ex:
        statuses = list(ex.map(lambda j: execute(j[2], j[3])[0], jobs))
    per_task = {}
    for (name, rank, _, _), st in zip(jobs, statuses):
        per_task.setdefault(name, [None] * 10)[rank] = st
    n = len(per_task)
    p1 = sum(v[0] == "passed" for v in per_task.values())
    p10 = sum("passed" in v for v in per_task.values())
    summary = {
        "tasks": n,
        "pass_at_1": {"ours": p1, "ours_pct": round(100 * p1 / n, 2), "published": 70,
                      "published_pct": round(100 * PUBLISHED["pass_at_1"], 2)},
        "pass_at_10": {"ours": p10, "ours_pct": round(100 * p10 / n, 2), "published": 109,
                       "published_pct": round(100 * PUBLISHED["pass_at_10"], 2)},
        "outcomes": {s: statuses.count(s) for s in sorted(set(statuses))},
        "per_task": per_task,
    }
    (out_dir / "li_lora_humaneval_java_scores.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))


if __name__ == "__main__":
    {"generate": generate, "score": score}[sys.argv[1]](sys.argv[2])
