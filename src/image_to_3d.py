"""Stage 4 — Image-to-3D reconstruction. Swappable backends:
  - 'tripo'       : TripoSR (fast, low VRAM, vertex colors only)  [implemented]
  - 'instantmesh' : InstantMesh (textured .glb, needs ~10GB VRAM) [TODO: wire repo API at benchmark gate]
  - 'mock'        : procedural icosphere, for UI dev without GPU
Background compositing color is PER-BACKEND (TripoSR=white, InstantMesh=gray)."""
import numpy as np
import torch
from PIL import Image
from config import BACKEND, MOCK_GPU, TRIPOSR_BG, INSTANTMESH_BG, MC_RESOLUTION

_tsr = None


def _prepare_for_3d(segmented_rgba: Image.Image, bg_rgb) -> Image.Image:
    """Tight alpha-bbox crop, pad to ~85% frame occupancy, composite onto backend-specific bg."""
    alpha = np.array(segmented_rgba.split()[3])
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        raise ValueError("No foreground object found after background removal.")
    box = (max(xs.min() - 8, 0), max(ys.min() - 8, 0),
           min(xs.max() + 8, segmented_rgba.width), min(ys.max() + 8, segmented_rgba.height))
    obj = segmented_rgba.crop(box)
    side = int(max(obj.size) / 0.85)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
    bg = Image.new("RGBA", canvas.size, (*bg_rgb, 255))
    return Image.alpha_composite(bg, canvas).convert("RGB")


def _run_triposr(rgb_img: Image.Image):
    global _tsr
    if _tsr is None:
        from tsr.system import TSR  # pip: clone github.com/VAST-AI-Research/TripoSR + its requirements
        from config import TRIPOSR_MODEL_ID
        _tsr = TSR.from_pretrained(TRIPOSR_MODEL_ID, config_name="config.yaml",
                                   weight_name="model.ckpt")
        _tsr.renderer.set_chunk_size(8192)
        if torch.cuda.is_available():
            _tsr.to("cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        scene = _tsr([rgb_img], device=device)
    meshes = _tsr.extract_mesh(scene, resolution=MC_RESOLUTION)
    return meshes[0]  # trimesh.Trimesh with vertex colors


def _run_instantmesh(rgb_img: Image.Image):
    # TODO (benchmark gate, Hour 5-8): wire TencentARC/InstantMesh repo
    # (multiview diffusion + FlexiCubes) behind this same interface, then pick the winner.
    raise NotImplementedError("InstantMesh backend not wired yet — run the benchmark gate first.")


def _run_mock(rgb_img: Image.Image):
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    mesh.visual.vertex_colors = [180, 160, 140, 255]
    return mesh


def reconstruct_3d_mesh(segmented_rgba: Image.Image, backend: str = None) -> "trimesh.Trimesh":
    backend = backend or ("mock" if MOCK_GPU else BACKEND)
    if backend in ("auto", "tripo"):
        return _run_triposr(_prepare_for_3d(segmented_rgba, TRIPOSR_BG))
    if backend == "instantmesh":
        return _run_instantmesh(_prepare_for_3d(segmented_rgba, INSTANTMESH_BG))
    if backend == "mock":
        return _run_mock(segmented_rgba)
    raise ValueError(f"Unknown backend: {backend}")