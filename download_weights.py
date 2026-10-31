"""Step 1 — One-click reproducible weight prefetch.

Judges run `python download_weights.py` once before the live demo; it
pre-fetches all 2D/3D weights into WEIGHTS_CACHE_DIR (point it at a
Kaggle Dataset path on the demo machine) and warms rembg's u2net so the
first real sketch never stalls on Wi-Fi.

Also clones the TripoSR source (the repo is not pip-installable) into
third_party/TripoSR for src/image_to_3d.py to import from.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from huggingface_hub import snapshot_download

import config
from src.logger import log_feature_progress

MODEL_TARGETS = [
    ("Stable Diffusion 1.5", config.SD_MODEL_ID),
    ("ControlNet-Scribble", config.CONTROLNET_MODEL_ID),
    ("TripoSR", config.TRIPOSR_MODEL_ID),
]

TRIPOSR_SOURCE_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_SOURCE_DIR = Path(__file__).resolve().parent / "third_party" / "TripoSR"


def prefetch_model_weights() -> None:
    """Snapshot-download each pinned model into WEIGHTS_CACHE_DIR."""
    Path(config.WEIGHTS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for label, model_id in MODEL_TARGETS:
        t0 = time.perf_counter()
        path = snapshot_download(repo_id=model_id, cache_dir=config.WEIGHTS_CACHE_DIR)
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
    warm_up_rembg()
    duration = f"{time.perf_counter() - t0:.1f}s"

    log_feature_progress(
        feature_name="Step 1 — Configuration & Reproducible Weight Downloader",
        role_owner="Role C (Infra Lead)",
        role_importance_and_job=(
            "Prevents DNF from dead repos & Wi-Fi stalls (Reproducibility 15 pts): locks "
            "active model IDs, per-backend BG colors, and provides 1-click weight caching."
        ),
        files_modified=["config.py", "download_weights.py", "requirements.txt", "src/image_to_3d.py"],
        technical_explanation=(
            "Pinned stable-diffusion-v1-5/stable-diffusion-v1-5 (active repo; runwayml variant "
            "is deleted), lllyasviel/sd-controlnet-scribble, and stabilityai/TripoSR, fetched "
            f"via huggingface_hub.snapshot_download into {config.WEIGHTS_CACHE_DIR}; rembg "
            "u2net.onnx warmed with a dummy 64x64 pass. TripoSR's GitHub repo is NOT "
            "pip-installable (no setup.py/pyproject.toml), so its tsr/ source is git-cloned "
            "(shallow) into third_party/TripoSR by this script; its runtime deps "
            "(omegaconf, einops, imageio, torchmcubes) come from requirements.txt. Device "
            f"pinning: Stage 2 on {config.DEVICE_2D}, Stage 4 on {config.DEVICE_3D}; "
            f"MOCK_GPU={config.MOCK_GPU}."
        ),
        duration_or_latency=duration,
    )
    print("All weights cached. Judges: run `python download_weights.py` once before the demo.")


if __name__ == "__main__":
    main()