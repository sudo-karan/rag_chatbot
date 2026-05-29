"""Hardware-aware profile selection.

Detects the host's RAM, CPU cores, and (NVIDIA) GPU + VRAM, then recommends
one of six tiered profiles. Each profile bundles the model + Ollama options
the rest of the app should use. Detection is lazy — call detect_profile();
nothing runs at import time.

The recommendation is advisory: it is logged at startup (and offered as an
interactive prompt by the terminal client). The operator confirms or changes
it with PROFILE_OVERRIDE=tiny|small|medium|large|gpu|xl — required for
container deployments where cgroup limits hide the host specs.

Tiers (RAM is host RAM; VRAM is summed across visible NVIDIA GPUs):
  tiny    : <16 GB RAM, no usable GPU    — one small model for everything
  small   : 16-32 GB RAM, no usable GPU  — 8b main + 1b helper
  medium  : 32-64 GB RAM, no usable GPU  — 8b main + 1b helper, verification on
  large   : 64+ GB RAM, no usable GPU    — 8b main + 1b helper, models pinned
  gpu     : GPU with >=8 and <40 GB VRAM — 8b main on GPU, models pinned
  xl      : GPU with >=40 GB VRAM        — 70b main + 8b helper, models pinned

VRAM thresholds are env-tunable: PROFILE_XL_MIN_VRAM_GB (default 40),
PROFILE_GPU_MIN_VRAM_GB (default 8).
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
    "gpu": Profile(
        name="gpu",
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
    except Exception:
        return False


def _detect_vram_gb() -> float:
    """Total VRAM across visible NVIDIA GPUs, in GB. 0.0 if unknown.

    Ollama can split a model across GPUs, so we sum rather than take the max."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return 0.0
        total_mib = 0.0
        for tok in result.stdout.split():
            try:
                total_mib += float(tok)
            except ValueError:
                pass
        return total_mib / 1024.0
    except Exception:
        return 0.0


def _detect_specs() -> tuple[float, int, bool, float]:
    """Return (ram_gb, cpu_cores, has_gpu, vram_gb)."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except Exception:
        ram_gb = 8.0
        cores = os.cpu_count() or 1
    has_gpu = _detect_gpu()
    vram_gb = _detect_vram_gb() if has_gpu else 0.0
    return ram_gb, cores, has_gpu, vram_gb


def _auto_profile(ram_gb: float, has_gpu: bool, vram_gb: float) -> str:
    xl_min = float(os.getenv("PROFILE_XL_MIN_VRAM_GB", "40"))
    gpu_min = float(os.getenv("PROFILE_GPU_MIN_VRAM_GB", "8"))
    # A GPU only promotes the tier when we can confirm enough VRAM. A detected
    # GPU with unknown or small VRAM falls through to the RAM-based tiers
    # (Ollama still offloads what fits) — we never auto-select 70b on a small card.
    if has_gpu and vram_gb >= xl_min:
        return "xl"
    if has_gpu and vram_gb >= gpu_min:
        return "gpu"
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
      2. Auto-detect from RAM + GPU/VRAM.

    The recommended (auto) tier is always reported in the info dict, even when
    an override wins, so startup can show "active X (override), recommended Y"."""
    ram_gb, cores, has_gpu, vram_gb = _detect_specs()
    recommended = _auto_profile(ram_gb, has_gpu, vram_gb)

    override = os.getenv("PROFILE_OVERRIDE", "").strip().lower()
    if override and override in PROFILES:
        profile = PROFILES[override]
        source = "override"
    else:
        if override:
            print(
                f"WARNING: PROFILE_OVERRIDE='{override}' is not a known profile "
                f"({'/'.join(PROFILES)}); using auto-detected '{recommended}'."
            )
        profile = PROFILES[recommended]
        source = "auto"

    return profile, {
        "profile": profile.name,
        "recommended": recommended,
        "source": source,
        "ram_gb": round(ram_gb, 1),
        "cpu_cores": cores,
        "gpu": has_gpu,
        "vram_gb": round(vram_gb, 1),
    }


def format_profile_banner(info: dict) -> str:
    """Operator-facing, multi-line summary of the profile decision."""
    gpu_desc = "no"
    if info.get("gpu"):
        gpu_desc = "yes"
        if info.get("vram_gb"):
            gpu_desc += f" ({info['vram_gb']} GB VRAM)"
    src = "PROFILE_OVERRIDE" if info.get("source") == "override" else "auto-detected"
    line = "─" * 62
    body = [
        line,
        " Hardware profile",
        f"   detected   : RAM {info['ram_gb']} GB · {info['cpu_cores']} cores · GPU {gpu_desc}",
        f"   recommended: {info.get('recommended', info['profile'])}",
        f"   active     : {info['profile']}  ({src})",
        "   change with PROFILE_OVERRIDE=tiny|small|medium|large|gpu|xl",
        line,
    ]
    return "\n".join(body)
