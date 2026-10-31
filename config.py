"""SKETCH2SHAPE AI — global config. Single source of truth for model IDs, prompts, constants.
NOTE: model IDs verified Oct 2026. The old 'runwayml/stable-diffusion-v1-5' repo is DEAD — do not use it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
EXAMPLES_DIR = ROOT / "examples"
OUTPUT_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(exist_ok=True)

# ---------- Models ----------
SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # community mirror (OpenRAIL-M)
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-scribble"
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"          # config_name="config.yaml", weight_name="model.ckpt"
INSTANTMESH_REPO = "TencentARC/InstantMesh"        # git clone; weights via its HF repo

# ---------- Backend selection ----------
# "auto" | "tripo" | "instantmesh" | "mock"
import os
BACKEND = os.environ.get("S2S_BACKEND", "auto")
MOCK_GPU = os.environ.get("S2S_MOCK", "0") == "1"   # set S2S_MOCK=1 to develop without GPU

# Background color each 3D backend expects (DO NOT mix these up):
TRIPOSR_BG = (255, 255, 255)      # TripoSR: white
INSTANTMESH_BG = (127, 127, 127)  # InstantMesh/LRM: neutral gray

# ---------- Image constants ----------
IMAGE_SIZE = 512
MIN_STROKE_COVERAGE = 0.02  # if strokes cover < 2% of pixels, skip Otsu (faint-sketch fallback)

# ---------- Diffusion ----------
DIFFUSION_STEPS = 20
DEFAULT_CONTROL_SCALE = 0.85
DEFAULT_SEED = 42
DEFAULT_PROMPT = ("single 3D object, high quality product render, studio lighting, "
                 "smooth solid surface, isometric view, centered, plain white background")
NEGATIVE_PROMPT = ("flat, 2d, sketch, wireframe, text, watermark, blurry, "
                   "multiple objects, cropped, shadow, cluttered background, low quality")

# ---------- Mesh ----------
MIN_VERTICES = 100
MIN_FACES = 100
MC_RESOLUTION = 256
SMOOTH_ITERATIONS = 1  # test 1 vs 2; 2 can over-shrink fine detail

# ---------- Logging ----------
LOG_FILE = ROOT / "done_so_far.md"