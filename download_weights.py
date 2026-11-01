"""Step 1 — One-click reproducible weight prefetch.

Judges run `python download_weights.py` once before the live demo; it
pre-fetches all 2D/3D weights into WEIGHTS_CACHE_DIR and warms rembg's
u2net so the first real sketch never stalls on Wi-Fi.

Also clones the InstantMesh source (primary 3D backend) and TripoSR source
(fallback) into third_party/, and installs torchmcubes with
--no-build-isolation (it must compile against the installed torch).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

import config
from src.logger import log_feature_progress

MODEL_TARGETS = [
    ("Stable Diffusion 1.5", config.SD_MODEL_ID, ["safety_checker/*"]),
    ("ControlNet-Scribble", config.CONTROLNET_MODEL_ID, []),
    ("TripoSR", config.TRIPOSR_MODEL_ID, []),
    ("Zero123++ multiview diffusion", config.ZERO123PLUS_MODEL_ID, []),
]

# Single-file weights from TencentARC/InstantMesh (skip the big unused variants).
INSTANTMESH_FILE_TARGETS = [
    ("InstantMesh custom Zero123++ UNet", config.INSTANTMESH_MODEL_ID, config.INSTANTMESH_UNET_FILE),
    ("InstantMesh FlexiCubes reconstruction", config.INSTANTMESH_MODEL_ID, config.INSTANTMESH_CKPT_FILE),
]

TRIPOSR_SOURCE_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_SOURCE_DIR = Path(__file__).resolve().parent / "third_party" / "TripoSR"

INSTANTMESH_SOURCE_URL = "https://github.com/TencentARC/InstantMesh.git"
INSTANTMESH_SOURCE_DIR = Path(__file__).resolve().parent / "third_party" / "InstantMesh"

TORCHMCUBES_URL = "git+https://github.com/tatsy/torchmcubes.git"
TORCHMCUBES_BUILD_DEPS = ["scikit-build-core", "ninja", "cmake"]


def prefetch_model_weights() -> None:
    """Snapshot-download each pinned model; hf-download InstantMesh single files."""
    Path(config.WEIGHTS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for label, model_id, ignore_patterns in MODEL_TARGETS:
        t0 = time.perf_counter()
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=config.WEIGHTS_CACHE_DIR,
            ignore_patterns=ignore_patterns,
        )
        print(f"[OK] {label} ({model_id}) -> {path} ({time.perf_counter() - t0:.1f}s)")

    for label, repo_id, filename in INSTANTMESH_FILE_TARGETS:
        t0 = time.perf_counter()
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=config.WEIGHTS_CACHE_DIR)
        print(f"[OK] {label} ({filename}) -> {path} ({time.perf_counter() - t0:.1f}s)")


def clone_source(name: str, url: str, target: Path, rename_src: bool = False) -> None:
    """Clone a source-only repo once; optionally rename its top-level src/."""
    marker = target / ("im_src" if rename_src else "tsr")
    if marker.is_dir():
        print(f"[OK] {name} source already present at {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    if rename_src and (target / "src").is_dir():
        # InstantMesh ships a top-level src/ that would shadow OUR src/ package
        # on sys.path; rename it so imports become im_src.utils.*.
        (target / "src").rename(target / "im_src")
        print(f"[OK] {name}: renamed src/ -> im_src/ (avoids package collision)")
    print(f"[OK] {name} source cloned -> {target} ({time.perf_counter() - t0:.1f}s)")


def install_torchmcubes() -> None:
    """Install torchmcubes; builds against installed torch, so no build isolation."""
    try:
        import torchmcubes  # noqa: F401

        print("[OK] torchmcubes already installed")
        return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", *TORCHMCUBES_BUILD_DEPS], check=True)
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
    clone_source("InstantMesh", INSTANTMESH_SOURCE_URL, INSTANTMESH_SOURCE_DIR, rename_src=True)
    clone_source("TripoSR", TRIPOSR_SOURCE_URL, TRIPOSR_SOURCE_DIR)
    install_torchmcubes()
    warm_up_rembg()
    duration = f"{time.perf_counter() - t0:.1f}s"

    log_feature_progress(
        feature_name="Step 1 — Configuration & Reproducible Weight Downloader",
        role_owner="Role C (Infra Lead)",
        role_importance_and_job=(
            "Prevents DNF from dead repos & Wi-Fi stalls (Reproducibility 15 pts): locks "
            "active model IDs, per-backend BG colors, and provides 1-click weight caching."
        ),
        files_modified=[
            "config.py", "download_weights.py", "requirements.txt", "src/image_to_3d.py",
        ],
        technical_explanation=(
            "Pinned active repos snapshot-downloaded into WEIGHTS_CACHE_DIR: "
            "stable-diffusion-v1-5 (safety_checker/* skipped, ~3.6 GB saved), "
            "sd-controlnet-scribble, stabilityai/TripoSR, and sudo-ai/zero123plus-v1.2 "
            "(InstantMesh multiview stage). InstantMesh weights fetched as single files: "
            "diffusion_pytorch_model.bin (custom white-bg Zero123++ UNet) and "
            "instant_mesh_large.ckpt (FlexiCubes recon). Source-only repos shallow-cloned "
            "into third_party/: InstantMesh (top-level src/ renamed to im_src/ so it "
            "cannot shadow our own src/ package) and TripoSR (tsr/). torchmcubes installed "
            "with --no-build-isolation after pip-installing its build backend. rembg "
            "u2net warmed with a dummy 64x64 pass."
        ),
        duration_or_latency=duration,
    )
    print("All weights cached. Judges: run `python download_weights.py` once before the demo.")


if __name__ == "__main__":
    main()