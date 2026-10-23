"""Phase 0 (deferred part): reproduce RepairLLaMA's DeepSeek-Coder LoRA on HumanEval-Java (RESEARCH_LOG #64, #73).

RepairLLaMA (Silva, Fang, Monperrus, IEEE TSE 2025) fine-tuned deepseek-coder-6.7b-base with LoRA (r = 8, alpha 16,
q/v projections) on Megadiff in their IR4 x OR2 representation: the buggy function with the buggy lines commented
out and a fill-in-the-middle hole; the model generates the fixed chunk. Their released adapter
(repairllama-deepseek-lora/) and their released HumanEval-Java results (results/0_original/
evaluation_humanevaljava_deepseek_lora.jsonl) are downloaded at run time (their repository has no licence file).

Generation follows their framework (ASSERT-KTH/elle-elle-aime, deepseek_fim.py): the exact prompts of their
released result file, adapter merged into the base model, beam search with 10 beams / 10 returned, max_length
4,096, early stopping; the function is rebuilt by putting each filling into the hole. Deviation: their code loads
the model in the default dtype (fp32, 27 GB, doesn't fit 24 GB); we use bf16 (the model's native dtype).
Scoring uses OUR harness (src/execution.py): the function replaces the buggy method (benchmark line range), then
the JUnit tests run. pass@k = a plausible patch among the top k beams, over all 163 tasks (as in Phase 0, #63).

    python analysis/phase0/repairllama_reproduce.py generate <out_dir>   # GPU
    python analysis/phase0/repairllama_reproduce.py score <out_dir>      # CPU; runs model-generated code (container only)
"""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BASE = "deepseek-ai/deepseek-coder-6.7b-base"
RAW = "https://raw.githubusercontent.com/ASSERT-KTH/repairllama/main"
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
RESULTS_URL = f"{RAW}/results/0_original/evaluation_humanevaljava_deepseek_lora.jsonl"
FIM_BEGIN, FIM_HOLE, FIM_END = "<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"


def fetch(url, dest):
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def released(out_dir):
    """Their released result file: one record per task with prompt, 10 generations and their evaluation."""
    path = fetch(RESULTS_URL, Path(out_dir) / "theirs_evaluation_humanevaljava_deepseek_lora.jsonl")
    return [json.loads(line) for line in open(path)]


def generate(out_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    adapter = out_dir / "adapter"
    for f in ADAPTER_FILES:
        fetch(f"{RAW}/repairllama-deepseek-lora/{f}", adapter / f)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()
    out_path = out_dir / "ours_outputs.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    tasks = released(out_dir)
    for i, t in enumerate(tasks, 1):
        name, prompt = t["identifier"], t["prompt"]
        if name in out:
            continue
        assert prompt.startswith(FIM_BEGIN) and prompt.endswith(FIM_END) and prompt.count(FIM_HOLE) == 1, name
        # transformers 5 no longer adds DeepSeek-Coder's BOS for plain text (add_bos_token is ignored); their
        # environment did, and without it the FIM prompts degenerate (RESEARCH_LOG #73), so prepend it explicitly
        ids = [tok.bos_token_id] + tok(prompt, add_special_tokens=False)["input_ids"]
        assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1, name
        inputs = {"input_ids": torch.tensor([ids], device="cuda")}
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        n = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=4096, num_beams=10, num_return_sequences=10,
                                 early_stopping=True, do_sample=False, use_cache=True)
        fillings = tok.batch_decode(gen[:, n:], skip_special_tokens=True)
        body = prompt.replace(FIM_BEGIN, "").replace(FIM_END, "")
        out[name] = [body.replace(FIM_HOLE, f) for f in fillings]
        print(f"{i}/{len(tasks)} {name}", flush=True)
        out_path.write_text(json.dumps(out, indent=1))


def score_set(patches, recs):
    """patches: {task: [function, ...]} -> {task: [status, ...]} with our harness."""
    from execution import execute

    jobs = [(name, rank, p) for name, ps in patches.items() for rank, p in enumerate(ps)]
    with ThreadPoolExecutor(8) as ex:
        statuses = list(ex.map(lambda j: execute(j[2], recs[j[0]])[0], jobs))
    per_task = {name: [None] * len(ps) for name, ps in patches.items()}
    for (name, rank, _), st in zip(jobs, statuses):
        per_task[name][rank] = st
    return per_task


def summarise(per_task):
    n = len(per_task)
    p1 = sum(v[0] == "passed" for v in per_task.values())
    p10 = sum("passed" in v for v in per_task.values())
    flat = [s for v in per_task.values() for s in v]
    return {"tasks": n, "pass_at_1": p1, "pass_at_1_pct": round(100 * p1 / n, 2), "pass_at_10": p10,
            "pass_at_10_pct": round(100 * p10 / n, 2), "outcomes": {s: flat.count(s) for s in sorted(set(flat))}}


def score(out_dir):
    out_dir = Path(out_dir)
    recs = {r["name"]: r for r in map(json.loads, open(ROOT / "data" / "humaneval_java.jsonl"))}
    tasks = released(out_dir)
    theirs_eval = {t["identifier"]: [bool(e["test"]) for e in t["evaluation"]] for t in tasks}
    published = {"pass_at_1": sum(v[0] for v in theirs_eval.values()),
                 "pass_at_10": sum(any(v) for v in theirs_eval.values()), "tasks": len(theirs_eval)}
    summary = {"published_by_their_harness": published}
    # 1. scoring-path check: their ground-truth fixed function through our harness (should pass)
    ref = score_set({t["identifier"]: [t["fixed_code"]] for t in tasks}, recs)
    summary["refcheck_fixed_code"] = {"passed": sum(v[0] == "passed" for v in ref.values()), "tasks": len(ref),
                                      "not_passed": {k: v[0] for k, v in ref.items() if v[0] != "passed"}}
    # 2. their released generations, scored by our harness (separates grader effects from generation effects)
    theirs = score_set({t["identifier"]: t["generation"] for t in tasks}, recs)
    summary["theirs_scored_by_us"] = summarise(theirs)
    # 3. our re-generation, scored by our harness
    ours_path = out_dir / "ours_outputs.json"
    if ours_path.exists():
        ours_gen = json.loads(ours_path.read_text())
        ours = score_set(ours_gen, recs)
        summary["ours_scored_by_us"] = summarise(ours)
        top = {t["identifier"]: t["generation"][0] for t in tasks}
        summary["identical_top_beam"] = sum(ours_gen[k][0].strip() == top[k].strip() for k in ours_gen)
        summary["identical_all_10"] = sum([g.strip() for g in ours_gen[k]] == [g.strip() for g in t["generation"]]
                                          for t in tasks for k in [t["identifier"]] if k in ours_gen)
        summary["per_task"] = {"ours": ours, "theirs_by_us": theirs, "theirs_by_them": theirs_eval}
    (out_dir / "repairllama_humaneval_java_scores.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))


if __name__ == "__main__":
    {"generate": generate, "score": score}[sys.argv[1]](sys.argv[2])
