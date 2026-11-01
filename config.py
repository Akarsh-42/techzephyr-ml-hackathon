"""Step 1 — Central configuration: model IDs, per-backend BG colors, device pinning, flags.

Non-negotiables enforced here (see blueprint Section 1):
- Active repos only: stable-diffusion-v1-5/* (never the deleted runwayml repo).
- Per-backend background compositing colors for Stage 4.
- Kaggle dual-T4 pinning: Stage 2 -> cuda:0, Stage 4 -> cuda:1.
"""
import os

import torch

# --- Model IDs (verified active) -------------------------------------------
SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-scribble"
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"
INSTANTMESH_MODEL_ID = "TencentARC/InstantMesh"
ZERO123PLUS_MODEL_ID = "sudo-ai/zero123plus-v1.2"

# --- Per-backend background compositing (CRITICAL, Stage 4) ----------------
# TripoSR: the official demo (gradio_app.py, fill_background) composites the
# RGBA foreground onto 0.5 GRAY — not white. On pure white the SDF field keeps
# a faint surface shell at the background plane, which surfaces as a thin
# background slab glued behind the object in the extracted mesh.
BG_COLOR_TRIPOSR = (127, 127, 127)      # neutral GRAY for TripoSR (official demo)
# InstantMesh: its CUSTOMIZED Zero123++ UNet was trained to generate
# WHITE-background multiviews (model card: "customized Zero123++ UNet for
# white-background image generation"), and the official Space feeds the
# foreground composited on white. Gray would fight the UNet's training
# distribution and desaturate the multiview colors.
BG_COLOR_INSTANTMESH = (255, 255, 255)  # WHITE for InstantMesh (official demo)

# --- TripoSR marching-cubes threshold ---------------------------------------
# extract_mesh(threshold=...) carves away low-density background fog around the
# object. 25.0 is the official default; raise if background shells persist.
TRIPOSR_MC_THRESHOLD = 25.0

# --- InstantMesh weights (single files inside TencentARC/InstantMesh) --------
INSTANTMESH_UNET_FILE = "diffusion_pytorch_model.bin"   # custom white-bg Zero123++ UNet
INSTANTMESH_CKPT_FILE = "instant_mesh_large.ckpt"       # FlexiCubes reconstruction model
INSTANTMESH_CONFIG_NAME = "instant-mesh-large"           # configs/<name>.yaml in the clone

# Zero123++ diffusion sampling steps. Official demo default is 75 (slider
# 30-75); 50 keeps multiview quality high while trimming ~1/3 of stage-4 time.
INSTANTMESH_DIFFUSION_STEPS = 50

# --- 3D backend selection ----------------------------------------------------
# "instantmesh" is the primary (better geometry + richer vertex colors from 6
# multiview renders); "triposr" stays as the fallback via env override:
#   PRIMARY_3D_BACKEND=triposr python app.py
PRIMARY_3D_BACKEND = os.getenv("PRIMARY_3D_BACKEND", "instantmesh")

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
# Shrink-free Taubin smoothing passes. Each pass = 2 filter sweeps (lambda +
# mu), so 3 passes = 30 total sweeps in mesh_validator.filter_taubin. Plain
# Laplacian shrank meshes into "mushy pillows"; Taubin removes high-frequency
# noise without volume loss, and 3 passes visibly smooths lumpy sketch meshes.
LAPLACIAN_ITERATIONS = 3      # shrink-free smoothing passes (was 1)