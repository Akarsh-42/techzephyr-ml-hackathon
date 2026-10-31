"""Step 1 â€” One-click reproducible weight prefetch.

Judges run `python download_weights.py` once before the live demo; it
pre-fetches all 2D/3D weights into WEIGHTS_CACHE_DIR (point it at a
Kaggle Dataset path on the demo machine) and warms rembg's u2net so the
first real sketch never stalls on Wi-Fi.

Also clones the TripoSR source (the repo is not pip-installable) into
third_party/TripoSR, and installs torchmcubes with --no-build-isolation
(it must compile against the already-installed torch).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

import config
from src.logger import log_feature_progress

# The pipeline never loads SD's safety_checker (Stage 2 instantiates with
# safety_checker=None), so skip its ~3.6 GB of weights at prefetch time.
MODEL_TARGETS = [
    ("Stable Diffusion 1.5", config.SD_MODEL_ID, ["safety_checker/*"]),
    ("ControlNet-Scribble", config.CONTROLNET_MODEL_ID, []),
    ("TripoSR", config.TRIPOSR_MODEL_ID, []),
]

TRIPOSR_SOURCE_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_SOURCE_DIR = Path(__file__).resolve().parent / "third_party" / "TripoSR"
TORCHMCUBES_URL = "git+https://github.com/tatsy/torchmcubes.git"
# --no-build-isolation means the venv itself must supply the build backend.
TORCHMCUBES_BUILD_DEPS = ["scikit-build-core", "ninja", "cmake"]


def prefetch_model_weights() -> None:
    """Snapshot-download each pinned model into WEIGHTS_CACHE_DIR."""
    Path(config.WEIGHTS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for label, model_id, ignore_patterns in MODEL_TARGETS:
        t0 = time.perf_counter()
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=config.WEIGHTS_CACHE_DIR,
            ignore_patterns=ignore_patterns,
        )
        print(f"[OK] {label} ({model_id}) -> {path} ({time.perf_counter() - t0:.1f}s)")


def clone_triposr_source() -> None:
    """Clone the TSR source package once (repo has no setup.py, so it's not pip-installable)."""
    if (TRIPOSR_SOURCE_DIR / "tsr").is_dir():
        print(f"[OK] TripoSR source already present at {TRIPOSR_SOURCE_DIR}")
        return
    TRIPOSR_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        ["git", "clone", "--depth", "1", TRIPOSR_SOURCE_URL, str(TRIPOSR_SOURCE_DIR)],
        check=True,
    )
    print(f"[OK] TripoSR source cloned -> {TRIPOSR_SOURCE_DIR} ({time.perf_counter() - t0:.1f}s)")


def install_torchmcubes() -> None:
    """Install torchmcubes; it must build against the installed torch, so no build isolation."""
    try:
        import torchmcubes  # noqa: F401
        print("[OK] torchmcubes already installed")
        return
    except ImportError:
        pass
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *TORCHMCUBES_BUILD_DEPS],
        check=True,
    )
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", TORCHMCUBES_URL],
        check=True,
    )
    print(f"[OK] torchmcubes installed ({time.perf_counter() - t0:.1f}s)")


def warm_up_rembg() -> None:
    """First rembg.remove() call downloads u2net.onnx; do it now, not mid-demo."""
    from PIL import Image
    from rembg import remove

    t0 = time.perf_counter()
    remove(Image.new("RGB", (64, 64), (255, 255, 255)))
    print(f"[OK] rembg u2net initialized ({time.perf_counter() - t0:.1f}s)")


def main() -> None:
    t0 = time.perf_counter()
    prefetch_model_weights()
    clone_triposr_source()
    install_torchmcubes()
    warm_up_rembg()
    duration = f"{time.perf_counter() - t0:.1f}s"

    log_feature_progress(
        feature_name="Step 1 â€” Configuration & Reproducible Weight Downloader",
        role_owner="Role C (Infra Lead)",
        role_importance_and_job=(
            "Prevents DNF from dead repos & Wi-Fi stalls (Reproducibility 15 pts): locks "
            "active model IDs, per-backend BG colors, and provides 1-click weight caching."
        ),
        files_modified=[
            "config.py", "download_weights.py", "requirements.txt", "src/image_to_3d.py",
        ],
        technical_explanation=(
            "Pinned stable-diffusion-v1-5/stable-diffusion-v1-5 (active repo; runwayml variant "
            "is deleted; safety_checker/* skipped since Stage 2 loads the pipeline with "
            "safety_checker=None, saving ~3.6 GB), lllyasviel/sd-controlnet-scribble, and "
            "stabilityai/TripoSR, fetched via huggingface_hub.snapshot_download into "
            f"{config.WEIGHTS_CACHE_DIR}; rembg u2net.onnx warmed with a dummy 64x64 pass. "
            "TripoSR's GitHub repo is NOT pip-installable (no setup.py/pyproject.toml), so its "
            "tsr/ source is git-cloned (shallow) into third_party/TripoSR. torchmcubes compiles "
            "against the installed PyTorch, so it cannot use pip build isolation and is installed "
            "by this script with --no-build-isolation (deliberately absent from requirements.txt); "
            "its build backend (scikit-build-core, ninja, cmake) is pip-installed first because "
            "--no-build-isolation requires the venv to supply it, and its remaining runtime deps "
            "(omegaconf, einops, imageio) come from requirements.txt. "
            f"Device pinning: Stage 2 on {config.DEVICE_2D}, Stage 4 on {config.DEVICE_3D}; "
            f"MOCK_GPU={config.MOCK_GPU}."
        ),
        duration_or_latency=duration,
    )
    print("All weights cached. Judges: run `python download_weights.py` once before the demo.")


if __name__ == "__main__":
    main()