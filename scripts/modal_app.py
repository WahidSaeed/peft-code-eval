"""Run any repository command on a Modal GPU container.

    pip install modal && modal token new            # once, on the local machine
    modal run scripts/modal_app.py --cmd "bash scripts/dry_run.sh"
    MODAL_GPU=L4 modal run scripts/modal_app.py --cmd "bash scripts/run_matrix.sh M1"
    modal volume get thesis-results / results_modal  # download results

The container gets the local repository (including the gitignored data/*.jsonl built by
src/data_loader.py), the pinned requirements, a default JDK (HumanEvalFix-Java), JDK 8 and the
HumanEval-Java checkout (HumanEval-Java). The Hugging Face cache and results/ live on Modal
volumes, so the 13 GB model is downloaded once and results survive the container.
All official runs must use one GPU type (RESEARCH_LOG #23); MODAL_GPU defaults to A10 (24 GB).
"""

import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = os.environ.get("MODAL_GPU", "A10")
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
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .add_local_dir(REPO, "/root/repo", ignore=[".git", ".venv", "venv", "results", "data/clm", "**/__pycache__"])
)
hf_cache = modal.Volume.from_name("thesis-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("thesis-results", create_if_missing=True)
app = modal.App("peft-code-eval")


@app.function(gpu=GPU, image=image, volumes={"/cache/hf": hf_cache, "/results": results}, timeout=6 * 3600)
def run(cmd: str):
    env = {**os.environ, "HF_HOME": "/cache/hf", "HEJ_JAVA_HOME": "/opt/jdk8",
           "HEJ_LIB": "/opt/clm/humaneval-java/lib"}
    setup = "cd /root/repo && ln -sfn /results results && ln -sfn /opt/clm data/clm"
    try:
        subprocess.run(["bash", "-c", f"{setup} && {cmd}"], check=True, env=env)
    finally:
        hf_cache.commit()
        results.commit()


@app.local_entrypoint()
def main(cmd: str = "bash scripts/dry_run.sh"):
    print(f"GPU: {GPU} | command: {cmd}")
    run.remote(cmd)
