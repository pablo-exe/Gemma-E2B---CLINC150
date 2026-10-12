#!/usr/bin/env bash
# Run from the repository root: bash scripts/setup_runpod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
# Validate the actual template, not only its advertised name. Do not replace torch.
python3 -c 'import sys, torch; assert sys.version_info[:2] == (3,11), "Expected template Python 3.11"; assert torch.__version__.split("+")[0] == "2.8.0", "Expected template PyTorch 2.8.0"; assert torch.version.cuda == "12.8", "Expected CUDA 12.8 wheel"; assert torch.cuda.is_available(), "CUDA unavailable"; print("Template:", sys.version.split()[0], torch.__version__, torch.version.cuda)'
# Reuse the template's GPU stack while installing project packages in our own venv.
python3 -m venv --system-site-packages .venv-pod
source .venv-pod/bin/activate
python -c 'import sys; from pathlib import Path; assert "include-system-site-packages = true" in Path(sys.prefix, "pyvenv.cfg").read_text(), "Old isolated .venv-pod: use a fresh pod/environment"'
python -c 'import torch; assert torch.__version__.split("+")[0] == "2.8.0", "Existing .venv-pod shadows template torch; use a fresh environment"'
python -c 'import importlib.metadata as m; from pathlib import Path; names = ("torch", "torchvision", "torchaudio", "triton"); installed = {p.metadata["Name"].lower(): p.version for p in m.distributions()}; Path(".venv-pod/template-constraints.txt").write_text("".join(f"{name}=={installed[name]}\n" for name in names if name in installed))'
python -m pip install --upgrade pip
python -m pip install -r scripts/requirements-pod.lock.txt -c .venv-pod/template-constraints.txt -c scripts/constraints-pod.txt
# Install only base package metadata; the Windows/local GPU extra is not selected.
python -m pip install --no-deps -e .
python -m pip check
python -m pip freeze > .venv-pod/installed-requirements.txt
python scripts/check_pod_environment.py --minimum-vram-gib 44
df -h /workspace
echo 'Ready. Activate with: source .venv-pod/bin/activate'
echo 'Cache: export HF_HOME=/workspace/hf-cache'
echo 'Authenticate if necessary: hf auth login'
