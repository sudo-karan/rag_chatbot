"""Hardware-aware profile selection.

At import time, detects the host's RAM, CPU cores, and GPU availability and
chooses one of five tiered profiles. Each profile bundles the model + Ollama
options the rest of the app should use.

Override with PROFILE_OVERRIDE=tiny|small|medium|large|xl — required for
container deployments where cgroup limits hide the host specs, and for
cloud envs where the auto-detection would pick the wrong tier.

Tiers (RAM is host RAM, not container limits):
  tiny    : <16 GB RAM, no GPU      — single small model for everything
  small   : 16-32 GB RAM, no GPU    — 8b main + 1b helper
  medium  : 32-64 GB RAM, no GPU    — 8b main + 1b helper, verification on
  large   : 64+ GB RAM, no GPU      — 8b main + 1b helper, models pinned
  xl      : any GPU detected        — 70b main + 8b helper, models pinned
"""
import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    main_model: str
    helper_model: str
    keep_alive: str           # Ollama keep_alive: "5m", "1h", "-1" for forever
    num_thread: int           # Ollama num_thread option (0 = auto)
    num_ctx: int              # Ollama context window for the main model
    enable_verification: bool # Run the grounding verifier on RAG answers
    embed_batch_size: int     # sentence-transformer batch size


PROFILES: dict[str, Profile] = {
    "tiny": Profile(
        name="tiny",
        main_model="llama3.2:3b",
        helper_model="llama3.2:1b",
        keep_alive="5m",
        num_thread=0,
        num_ctx=2048,
        enable_verification=False,
        embed_batch_size=16,
    ),
    "small": Profile(
        name="small",
        main_model="llama3.1:8b",
        helper_model="llama3.2:1b",
        keep_alive="5m",
        num_thread=0,
        num_ctx=4096,
        enable_verification=False,
        embed_batch_size=32,
    ),
    "medium": Profile(
        name="medium",
        main_model="llama3.1:8b",
        helper_model="llama3.2:1b",
        keep_alive="30m",
        num_thread=0,
        num_ctx=4096,
        enable_verification=True,
        embed_batch_size=64,
    ),
    "large": Profile(
        name="large",
        main_model="llama3.1:8b",
        helper_model="llama3.2:1b",
        keep_alive="-1",
        num_thread=0,
        num_ctx=8192,
        enable_verification=True,
        embed_batch_size=128,
    ),
    "xl": Profile(
        name="xl",
        main_model="llama3.1:70b",
        helper_model="llama3.1:8b",
        keep_alive="-1",
        num_thread=0,
        num_ctx=16384,
        enable_verification=True,
        embed_batch_size=256,
    ),
}


def _detect_gpu() -> bool:
    """True if an NVIDIA GPU is visible via nvidia-smi. Cheap and quiet on miss."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            timeout=3,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def _detect_specs() -> tuple[float, int, bool]:
    """Return (ram_gb, cpu_cores, has_gpu)."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except Exception:
        ram_gb = 8.0
        cores = os.cpu_count() or 1
    return ram_gb, cores, _detect_gpu()


def _auto_profile(ram_gb: float, has_gpu: bool) -> str:
    if has_gpu:
        return "xl"
    if ram_gb >= 64:
        return "large"
    if ram_gb >= 32:
        return "medium"
    if ram_gb >= 16:
        return "small"
    return "tiny"


def detect_profile() -> tuple[Profile, dict]:
    """Return the active Profile plus a summary dict for logging.

    Resolution order:
      1. PROFILE_OVERRIDE env var if set to a known profile name.
      2. Auto-detect from RAM + GPU.
    """
    ram_gb, cores, has_gpu = _detect_specs()

    override = os.getenv("PROFILE_OVERRIDE", "").strip().lower()
    if override and override in PROFILES:
        profile = PROFILES[override]
        source = "override"
    else:
        auto_name = _auto_profile(ram_gb, has_gpu)
        profile = PROFILES[auto_name]
        source = "auto"

    return profile, {
        "profile": profile.name,
        "source": source,
        "ram_gb": round(ram_gb, 1),
        "cpu_cores": cores,
        "gpu": has_gpu,
    }


ACTIVE_PROFILE, PROFILE_INFO = detect_profile()
