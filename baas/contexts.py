import os
import sys
import json
import platform
import shutil
import subprocess
import importlib.util
from typing import Dict, List, Tuple

def _has_cmd(cmd: str) -> int:
    return 1 if shutil.which(cmd) else 0

def _has_module(name: str) -> int:
    return 1 if importlib.util.find_spec(name) is not None else 0

def _probe_gpu() -> Tuple[int, int]:
    """
    Returns (has_nvidia, has_cuda_toolkit) as ints {0,1}.
    We avoid heavy deps; just probe for commands/libs.
    """
    has_nvidia = 0
    try:
        out = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1.5
        )
        has_nvidia = 1 if out.returncode == 0 else 0
    except Exception:
        has_nvidia = 0

    # CUDA toolkit presence via nvcc
    has_cuda = _has_cmd("nvcc")
    return has_nvidia, has_cuda

def _os_onehot() -> List[int]:
    """Simple one-hot for OS: [linux, mac, win]."""
    sysname = platform.system().lower()
    return [
        1 if "linux" in sysname else 0,
        1 if "darwin" in sysname or "mac" in sysname else 0,
        1 if "windows" in sysname else 0,
    ]

def _python_major_minor() -> Tuple[int, int]:
    v = sys.version_info
    return v.major, v.minor

def _pkg_mgrs() -> List[int]:
    """One-hot for presence of package managers: [pip, conda, npm, uv]."""
    return [_has_cmd("pip"), _has_cmd("conda"), _has_cmd("npm"), _has_cmd("uv")]

def _cloud_envs() -> List[int]:
    """
    Heuristics for cloud/dev containers:
    - Codespaces, Colab, Kaggle, Paperspace, A100 mention (very rough)
    """
    env = os.environ
    codespaces = 1 if env.get("CODESPACES") or env.get("GITHUB_CODESPACES_PORT") else 0
    colab = 1 if "COLAB_GPU" in env or "COLAB_JUPYTER_IP" in env else 0
    kaggle = 1 if env.get("KAGGLE_KERNEL_RUN_TYPE") else 0
    return [codespaces, colab, kaggle]

def _ml_stack() -> List[int]:
    """Presence of popular ML libs: [torch, tensorflow, jax, transformers, vllm]."""
    return [
        _has_module("torch"),
        _has_module("tensorflow"),
        _has_module("jax"),
        _has_module("transformers"),
        _has_module("vllm"),
    ]

def _budget_flags() -> List[int]:
    """
    Placeholder for user preferences (could be set via env vars later):
    BAAS_LOW_COST, BAAS_LOW_LATENCY, BAAS_PRIVACY.
    """
    env = os.environ
    low_cost = 1 if env.get("BAAS_LOW_COST") == "1" else 0
    low_latency = 1 if env.get("BAAS_LOW_LATENCY") == "1" else 0
    privacy = 1 if env.get("BAAS_PRIVACY") == "1" else 0
    return [low_cost, low_latency, privacy]

def get_context() -> Dict:
    """Return a rich, structured context dict."""
    os_oh = _os_onehot()
    py_major, py_minor = _python_major_minor()
    has_nvidia, has_cuda = _probe_gpu()

    ctx = {
        "os_onehot": os_oh,                           # len 3
        "python_major": py_major,                     # int
        "python_minor": py_minor,                     # int
        "gpu_nvidia": has_nvidia,                     # 0/1
        "cuda_toolkit": has_cuda,                     # 0/1
        "pkg_mgrs": _pkg_mgrs(),                      # len 4
        "cloud_envs": _cloud_envs(),                  # len 3
        "ml_stack": _ml_stack(),                      # len 5
        "budget_flags": _budget_flags(),              # len 3
    }
    return ctx

def vectorize_context(ctx: Dict) -> List[float]:
    """
    Convert context dict to a numeric feature vector for the bandit.
    Keep ordering stable; scale small integers lightly.
    """
    vec: List[float] = []
    vec += list(map(float, ctx["os_onehot"]))                # 3
    vec += [float(ctx["python_major"]) / 4.0]                # 1
    vec += [float(ctx["python_minor"]) / 12.0]               # 1
    vec += [float(ctx["gpu_nvidia"]), float(ctx["cuda_toolkit"])]  # 2
    vec += list(map(float, ctx["pkg_mgrs"]))                 # 4
    vec += list(map(float, ctx["cloud_envs"]))               # 3
    vec += list(map(float, ctx["ml_stack"]))                 # 5
    vec += list(map(float, ctx["budget_flags"]))             # 3
    return vec

if __name__ == "__main__":
    ctx = get_context()
    print(json.dumps(ctx, indent=2))
    vec = vectorize_context(ctx)
    print("vector_length:", len(vec))
    print("vector_preview:", vec[:10])