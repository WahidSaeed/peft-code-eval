#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu GPU VM (22.04/24.04, NVIDIA driver preinstalled by the provider).
#   bash scripts/setup_gpu_vm.sh            # run from the repository root on the VM
# Installs Python tooling, JDK 8 + default JDK (test harnesses), a venv with requirements.txt,
# and the HumanEval-Java checkout. Data files are built separately (src/data_loader.py) or copied in.
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi || { echo "no NVIDIA driver: install one first (provider GPU image recommended)"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q
sudo apt-get install -y -q python3-venv python3-pip git default-jdk openjdk-8-jdk-headless tmux

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

if [[ ! -d data/clm ]]; then
  git clone -q https://github.com/lin-tan/clm data/clm
  git -C data/clm checkout -q 662769e515cfd2a4980d7d4e98f4739ef6265b45
fi

.venv/bin/python - <<'EOF'
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
print("transformers", transformers.__version__, "peft", peft.__version__, "bitsandbytes", bitsandbytes.__version__)
EOF
echo "JDK 8 for HumanEval-Java: export HEJ_JAVA_HOME=$(dirname "$(dirname "$(readlink -f "$(ls /usr/lib/jvm/java-8-openjdk-*/bin/javac | head -1)")")")"
