"""Step 1 — Central configuration: model IDs, per-backend BG colors, device pinning, flags.

Non-negotiables enforced here (see blueprint Section 1):
- Active repos only: stable-diffusion-v1-5/* (never the deleted runwayml repo).
- Per-backend background compositing colors for Stage 4.
- Kaggle dual-T4 pinning: Stage 2 -> cuda:0, Stage 4 -> cuda:1.
"""
import os

# Must be configured before importing torch. This helps reuse CUDA allocations
# across the sequential 2D and 3D stages, but cannot make an intrinsically
# oversized operation fit in VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# --- Model IDs (verified active) -------------------------------------------
SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-scribble"
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"
INSTANTMESH_MODEL_ID = "TencentARC/InstantMesh"

# --- Per-backend background compositing (CRITICAL, Stage 4) ----------------
BG_COLOR_TRIPOSR = (255, 255, 255)      # pure WHITE for TripoSR
BG_COLOR_INSTANTMESH = (127, 127, 127)  # neutral GRAY for InstantMesh

# --- 3D backend selection (Hour 5-8 benchmark gate decides) -----------------
PRIMARY_3D_BACKEND = os.getenv("PRIMARY_3D_BACKEND", "instantmesh")  # "instantmesh" | "triposr"

# --- Device pinning (Kaggle T4 x2 does NOT pool VRAM) -----------------------
DEVICE_2D = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_3D = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_2D

# --- Flags -------------------------------------------------------------------
MOCK_GPU = os.getenv("MOCK_GPU", "0") == "1" or not torch.cuda.is_available()

# --- Weight caching (1-click judge reproducibility) ---------------------------
WEIGHTS_CACHE_DIR = os.getenv(
    "WEIGHTS_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_cache"),
)

# --- Stage 3 / Stage 5 thresholds --------------------------------------------
FRAME_OCCUPANCY_RATIO = 0.85  # foreground fills 85% of the recentered RGBA frame
MIN_VERTICES = 100            # strict floor; never >= 4
MIN_FACES = 100               # strict floor; never >= 4
MIN_AXIS_DEPTH = 0.05         # anti-plane check; never 1e-4
LAPLACIAN_ITERATIONS = 1      # controlled smoothing, default 1 iteration

# InstantMesh's upstream large configuration uses a 128^3 FlexiCubes lattice.
# It evaluates more than two million vertices at once and needs about 15 GiB of
# temporary VRAM, which cannot fit on a 15 GB Colab T4. A 64^3 lattice is eight
# times smaller, still produces a dense real mesh, and is T4-safe. Override
# this only on a larger GPU with INSTANTMESH_GRID_RES=128.
INSTANTMESH_GRID_RES = int(os.getenv("INSTANTMESH_GRID_RES", "64"))
if INSTANTMESH_GRID_RES < 32:
    raise ValueError("INSTANTMESH_GRID_RES must be at least 32.")

# Keeping the LRM after exporting competes with the next ControlNet request on
# a single-GPU notebook. Enable only where VRAM headroom is known to be ample.
KEEP_3D_MODEL_LOADED = os.getenv("KEEP_3D_MODEL_LOADED", "0") == "1"
