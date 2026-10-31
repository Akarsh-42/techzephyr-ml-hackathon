"""Step 1 — One-click reproducible weight prefetch.

Judges run `python download_weights.py` once before the live demo; it
pre-fetches all 2D/3D weights into WEIGHTS_CACHE_DIR (point it at a
Kaggle Dataset path on the demo machine) and warms rembg's u2net so the
first real sketch never stalls on Wi-Fi.

Also clones the InstantMesh source (the repo is not pip-installable) into
third_party/InstantMesh, and installs nvdiffrast (required by InstantMesh
for FlexiCubes mesh extraction).
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
    ("Stable Diffusion 1.5", config.SD_MODEL_ID),
    ("ControlNet-Scribble", config.CONTROLNET_MODEL_ID),
]

# InstantMesh weights are individual files, not a full snapshot.
INSTANTMESH_FILES = [
    ("InstantMesh UNet", "diffusion_pytorch_model.bin"),
    ("InstantMesh LRM", "instant_mesh_large.ckpt"),
]

INSTANTMESH_SOURCE_URL = "https://github.com/TencentARC/InstantMesh.git"
INSTANTMESH_SOURCE_DIR = Path(__file__).resolve().parent / "third_party" / "InstantMesh"

# Zero123++ diffusion model (used by InstantMesh for multi-view generation)
ZERO123PLUS_MODEL_ID = "sudo-ai/zero123plus-v1.2"


def prefetch_model_weights() -> None:
    """Snapshot-download each pinned model into WEIGHTS_CACHE_DIR."""
    Path(config.WEIGHTS_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # SD1.5 + ControlNet (for Stage 2)
    for label, model_id in MODEL_TARGETS:
        t0 = time.perf_counter()
        path = snapshot_download(repo_id=model_id, cache_dir=config.WEIGHTS_CACHE_DIR)
        print(f"[OK] {label} ({model_id}) -> {path} ({time.perf_counter() - t0:.1f}s)")

    # Zero123++ (for InstantMesh Stage 4 multi-view generation)
    t0 = time.perf_counter()
    path = snapshot_download(repo_id=ZERO123PLUS_MODEL_ID, cache_dir=config.WEIGHTS_CACHE_DIR)
    print(f"[OK] Zero123++ ({ZERO123PLUS_MODEL_ID}) -> {path} ({time.perf_counter() - t0:.1f}s)")

    # InstantMesh individual weight files
    for label, filename in INSTANTMESH_FILES:
        t0 = time.perf_counter()
        path = hf_hub_download(
            repo_id=config.INSTANTMESH_MODEL_ID,
            filename=filename,
            repo_type="model",
        )
        print(f"[OK] {label} ({filename}) -> {path} ({time.perf_counter() - t0:.1f}s)")


def clone_instantmesh_source() -> None:
    """Clone the InstantMesh source package once (repo has no setup.py, so it's not pip-installable)."""
    if (INSTANTMESH_SOURCE_DIR / "src" / "models").is_dir():
        print(f"[OK] InstantMesh source already present at {INSTANTMESH_SOURCE_DIR}")
        return
    INSTANTMESH_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        ["git", "clone", "--depth", "1", INSTANTMESH_SOURCE_URL, str(INSTANTMESH_SOURCE_DIR)],
        check=True,
    )
    print(f"[OK] InstantMesh source cloned -> {INSTANTMESH_SOURCE_DIR} ({time.perf_counter() - t0:.1f}s)")


def install_nvdiffrast() -> None:
    """Install nvdiffrast (required by InstantMesh for differentiable rasterization)."""
    try:
        import nvdiffrast  # noqa: F401
        print("[OK] nvdiffrast already installed")
        return
    except ImportError:
        pass
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "git+https://github.com/NVlabs/nvdiffrast/"],
        check=True,
    )
    print(f"[OK] nvdiffrast installed ({time.perf_counter() - t0:.1f}s)")


def install_extra_deps() -> None:
    """Install xatlas and PyMCubes (InstantMesh FlexiCubes dependencies)."""
    deps = [
        ("xatlas", "xatlas"),
        ("PyMCubes", "mcubes"),
    ]
    for pkg_name, import_name in deps:
        try:
            __import__(import_name)
            print(f"[OK] {pkg_name} already installed")
        except ImportError:
            t0 = time.perf_counter()
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg_name],
                check=True,
            )
            print(f"[OK] {pkg_name} installed ({time.perf_counter() - t0:.1f}s)")


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
    clone_instantmesh_source()
    install_nvdiffrast()
    install_extra_deps()
    warm_up_rembg()
    duration = f"{time.perf_counter() - t0:.1f}s"

    log_feature_progress(
        feature_name="Step 1 — Configuration & Reproducible Weight Downloader (InstantMesh)",
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
            "is deleted), lllyasviel/sd-controlnet-scribble, sudo-ai/zero123plus-v1.2, and "
            "TencentARC/InstantMesh weights, fetched via huggingface_hub.snapshot_download and "
            f"hf_hub_download into {config.WEIGHTS_CACHE_DIR}; rembg u2net.onnx warmed with a "
            "dummy 64x64 pass. InstantMesh source is git-cloned (shallow) into "
            "third_party/InstantMesh. nvdiffrast (for FlexiCubes rasterization), xatlas, and "
            "PyMCubes installed as build deps. "
            f"Device pinning: Stage 2 on {config.DEVICE_2D}, Stage 4 on {config.DEVICE_3D}; "
            f"MOCK_GPU={config.MOCK_GPU}."
        ),
        duration_or_latency=duration,
    )
    print("All weights cached. Judges: run `python download_weights.py` once before the demo.")


if __name__ == "__main__":
    main()