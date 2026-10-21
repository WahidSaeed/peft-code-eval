"""Run any repository command on a Modal GPU container.

    pip install modal && modal token new            # once, on the local machine
    modal run scripts/modal_app.py --cmd "bash scripts/dry_run.sh"     # attached: stops if the laptop disconnects
    MODAL_GPU=L4 modal run scripts/modal_app.py --cmd "bash scripts/run_matrix.sh M1"

    # Long jobs: deploy once, then start the job server-side; it no longer depends on the local client
    modal deploy scripts/modal_app.py
    python scripts/modal_app.py "bash scripts/run_matrix.sh"           # prints the call id; returns at once
    modal app logs peft-code-eval                                      # follow the output
    modal volume get thesis-results / results_modal                    # download results

The container gets the local repository (including the gitignored data/*.jsonl built by
src/data_loader.py), the pinned requirements, a default JDK (HumanEvalFix-Java), JDK 8 and the
HumanEval-Java checkout (HumanEval-Java). The Hugging Face cache and results/ live on Modal
volumes, so the 13 GB model is downloaded once and results survive the container.
All official runs must use one GPU model (RESEARCH_LOG #23): each job refuses any other board (MODAL_REQUIRE_GPU, default "NVIDIA A10G").
"""

import os
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = os.environ.get("MODAL_GPU", "A10")
# Modal's "A10" type returns NVIDIA A10 or A10G boards (#46). Each job states the one board it accepts;
# official runs must all use the same one. MODAL_REQUIRE_GPU is read locally and passed to the job.
REQUIRED_GPU = os.environ.get("MODAL_REQUIRE_GPU", "NVIDIA A10G")
CLM_COMMIT = "662769e515cfd2a4980d7d4e98f4739ef6265b45"
JDK8_URL = "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jdk/hotspot/normal/eclipse"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "default-jdk-headless")
    .run_commands(
        f"mkdir -p /opt/jdk8 && wget -qO- {JDK8_URL} | tar xz -C /opt/jdk8 --strip-components=1",
        f"git clone -q https://github.com/lin-tan/clm /opt/clm && git -C /opt/clm checkout -q {CLM_COMMIT}"
        " && tar xzf /opt/clm/humaneval-java/humaneval-java.tar.gz -C /opt/clm",  # repo ships it as a tarball
    )
    .pip_install_from_requirements(str(REPO / "requirements.lock.txt"))  # exact versions, same as on any other host
    .add_local_dir(REPO, "/root/repo", ignore=[".git", ".venv", "venv", "results", "data/clm", "**/__pycache__"])
)
hf_cache = modal.Volume.from_name("thesis-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("thesis-results", create_if_missing=True)
app = modal.App("peft-code-eval")


@app.function(gpu=GPU, image=image, volumes={"/cache/hf": hf_cache, "/results": results}, timeout=24 * 3600,
              retries=modal.Retries(max_retries=5, initial_delay=5.0, backoff_coefficient=1.0))
def run(cmd: str, require_gpu: str = REQUIRED_GPU):
    """Runs `cmd` in the repo. Raises (so Modal retries in a fresh container) only for a wrong GPU or a lost
    container; a failing command returns its exit code instead, so it is never re-run automatically."""
    info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    name = info.split(",")[0].strip()
    if name != require_gpu:
        raise RuntimeError(f"wrong GPU: got {name}, need {require_gpu}")  # Modal retries in a new container
    with open("/results/platform_log.csv", "a") as f:  # which board / driver / power limit ran which job
        f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")},modal,{info},"{cmd}"\n')
    results.commit()
    env = {**os.environ, "HF_HOME": "/cache/hf", "HEJ_JAVA_HOME": "/opt/jdk8",
           "HEJ_LIB": "/opt/clm/humaneval-java/lib",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # less fragmentation; same for every run
    setup = "cd /root/repo && ln -sfn /results results && ln -sfn /opt/clm data/clm"
    try:
        code = subprocess.run(["bash", "-c", f"{setup} && {cmd}"], env=env).returncode
    finally:
        hf_cache.commit()
        results.commit()
    print(f"exit code {code}")
    return code


@app.local_entrypoint()
def main(cmd: str = "bash scripts/dry_run.sh"):
    print(f"GPU: {GPU} (require {REQUIRED_GPU}) | command: {cmd}")
    print(f"exit code {run.remote(cmd, REQUIRED_GPU)}")


if __name__ == "__main__":  # start a job on the DEPLOYED app (see docstring)
    import sys

    cmd = " ".join(sys.argv[1:]) or "bash scripts/dry_run.sh"
    call = modal.Function.from_name("peft-code-eval", "run").spawn(cmd, REQUIRED_GPU)
    print(f"spawned {call.object_id} on {GPU}, accepting only {REQUIRED_GPU}: follow with `modal app logs peft-code-eval`")
