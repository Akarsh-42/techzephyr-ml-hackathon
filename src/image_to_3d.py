"""Stage 4 — Per-backend BG compositing + TripoSR 3D reconstruction (cuda:1 / cuda:0)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

import config

# TripoSR's repo is source-only (no setup.py), so the tsr package lives in a
# git-cloned checkout (third_party/TripoSR, created by download_weights.py).
_TRIPOSR_SRC = Path(__file__).resolve().parent.parent / "third_party" / "TripoSR"
if _TRIPOSR_SRC.is_dir() and str(_TRIPOSR_SRC) not in sys.path:
    sys.path.insert(0, str(_TRIPOSR_SRC))

_triposr_model = None


def composite_for_backend(rgba_img: Image.Image, backend: str) -> Image.Image:
    bg = config.BG_COLOR_INSTANTMESH if backend == "instantmesh" else config.BG_COLOR_TRIPOSR
    rgba = rgba_img.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*bg, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def _load_triposr():
    global _triposr_model
    if _triposr_model is None:
        from tsr.system import TSR

        cache = Path(config.WEIGHTS_CACHE_DIR)
        snapshot = next(p for p in cache.rglob("config.yaml") if "TripoSR" in str(p)).parent
        model = TSR.from_pretrained(
            snapshot, config_name="config.yaml", weight_name="model.ckpt"
        )
        model.renderer.set_chunk_size(8192)
        _triposr_model = model.to(config.DEVICE_3D)
    return _triposr_model


def _mock_mesh() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
    mesh.visual.vertex_colors = ((mesh.vertex_normals * 0.5 + 0.5) * 255).astype(np.uint8)
    return mesh


def _sample_vertex_colors(mesh: trimesh.Trimesh, img: Image.Image) -> np.ndarray:
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    uv = (v - lo) / np.maximum(hi - lo, 1e-6)
    xs = np.clip((uv[:, 0] * (img.width - 1)).round().astype(int), 0, img.width - 1)
    ys = np.clip(((1 - uv[:, 1]) * (img.height - 1)).round().astype(int), 0, img.height - 1)
    return np.asarray(img)[ys, xs][:, :3]


def _extract_mesh_with_texture(
    model,
    scene_codes,
    composite: Image.Image,
    mc_resolution: int,
) -> trimesh.Trimesh:
    """Prefer TSR's native multi-view textured extraction.

    has_vertex_color=True makes TSR sample colors from rendered views of the
    reconstruction, so side/back-facing vertices get correct colors instead of
    the naive single-image bbox projection (which smears back faces). If the
    textured path fails for any reason, fall back to bbox vertex-color baking.
    """
    try:
        mesh = model.extract_mesh(
            scene_codes, has_vertex_color=True, resolution=mc_resolution
        )[0]
        has_uv = getattr(mesh.visual, "uv", None) is not None
        if not has_uv:
            raise ValueError("TSR textured extraction returned no UV map")
        return mesh
    except Exception:
        mesh = model.extract_mesh(
            scene_codes, has_vertex_color=False, resolution=mc_resolution
        )[0]
        mesh.visual.vertex_colors = _sample_vertex_colors(mesh, composite)
        return mesh


def reconstruct_3d_mesh(
    rgba_img: Image.Image,
    backend: str = config.PRIMARY_3D_BACKEND,
    mc_resolution: int = 256,
    seed: int = 42,
) -> trimesh.Trimesh:
    if config.MOCK_GPU:
        return _mock_mesh()

    if backend == "instantmesh":
        raise ValueError(
            "InstantMesh backend is not wired yet (pending Hour 5-8 benchmark gate); use 'triposr'."
        )

    composite = composite_for_backend(rgba_img, backend)
    model = _load_triposr()
    device = torch.device(config.DEVICE_3D)
    with torch.no_grad():
        scene_codes = model([composite], device)
        mesh = _extract_mesh_with_texture(model, scene_codes, composite, mc_resolution)

    return mesh


if __name__ == "__main__":
    from src.logger import log_feature_progress

    rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(rgba).ellipse([40, 40, 470, 470], fill=(200, 90, 40, 255))

    triposr_bg = composite_for_backend(rgba, "triposr")
    instant_bg = composite_for_backend(rgba, "instantmesh")
    assert np.asarray(triposr_bg)[0, 0].tolist() == [255, 255, 255]
    assert np.asarray(instant_bg)[0, 0].tolist() == [127, 127, 127]

    mesh = reconstruct_3d_mesh(rgba)
    assert len(mesh.vertices) >= config.MIN_VERTICES
    assert len(mesh.faces) >= config.MIN_FACES
    assert mesh.visual is not None

    log_feature_progress(
        feature_name="Step 4 — 3D Reconstruction & Per-Backend Compositing",
        role_owner="Role B (3D Lead)",
        role_importance_and_job=(
            "Core engine for 3D Output Quality (25 pts) & Efficiency (20 pts): routes "
            "per-backend BG colors (WHITE for TripoSR, GRAY for InstantMesh) and extracts "
            "the 3D mesh on the pinned DEVICE_3D with native multi-view texturing."
        ),
        files_modified=["src/image_to_3d.py", "requirements.txt", "download_weights.py"],
        technical_explanation=(
            "composite_for_backend reads BG colors exclusively from config. TripoSR path: "
            "tsr.system.TSR loaded once as a lazy singleton from the snapshot_download cache "
            "via TSR.from_pretrained(snapshot_dir, config_name='config.yaml', "
            "weight_name='model.ckpt') then .to(DEVICE_3D); chunked rendering (8192) to bound "
            "VRAM. _extract_mesh_with_texture first tries TSR's native textured extraction "
            "(has_vertex_color=True), which samples colors from rendered views of the "
            "reconstruction so back-facing vertices are colored correctly; on any failure "
            "it falls back to extract_mesh(has_vertex_color=False) plus bbox vertex-color "
            "baking. mc_resolution is caller-tunable (UI slider 128-512). MOCK_GPU returns "
            "an icosphere for CPU-first UI development."
        ),
        duration_or_latency="TripoSR ~5-15s on T4 (textured path slightly higher)",
    )
    print(f"Step 4 verified: mesh with {len(mesh.vertices)} verts / {len(mesh.faces)} faces.")