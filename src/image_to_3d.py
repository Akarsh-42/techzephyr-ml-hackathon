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


def to_gradio_3d_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """TripoSR's official viewer-orientation transform (tsr/utils.py).

    TSR meshes come out y-up in TSR's internal camera convention; the Gradio
    Model3D viewer (and most glTF consumers) expect a different frame. The
    official demo applies exactly these two rotations after extract_mesh —
    skipping it is why objects appeared rotated 90 degrees in the viewer while
    the vertex colors (baked per-vertex, riding the geometry) looked "straight".
    """
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return mesh


def _load_triposr():
    global _triposr_model
    if _triposr_model is None:
        from tsr.system import TSR

        cache = Path(config.WEIGHTS_CACHE_DIR)
        snapshots = sorted(
            (
                checkpoint.parent
                for checkpoint in cache.rglob("model.ckpt")
                if "triposr" in str(checkpoint).lower()
                and (checkpoint.parent / "config.yaml").is_file()
            ),
            key=str,
        )
        if not snapshots:
            raise FileNotFoundError(
                "TripoSR config.yaml and model.ckpt were not found together under "
                f"WEIGHTS_CACHE_DIR={cache}. Run download_weights.py first."
            )

        snapshot_dir = snapshots[0]
        model = TSR.from_pretrained(str(snapshot_dir), "config.yaml", "model.ckpt")
        model.renderer.set_chunk_size(8192)
        _triposr_model = model.to(config.DEVICE_3D)
        _triposr_model.eval()
    return _triposr_model


def _mock_mesh() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
    mesh.visual.vertex_colors = ((mesh.vertex_normals * 0.5 + 0.5) * 255).astype(np.uint8)
    return mesh


def _has_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    """True if the mesh carries a per-vertex color for every vertex.

    TripoSR's textured extraction (extract_mesh(has_vertex_color=True)) bakes
    VERTEX COLORS from the triplane — it does NOT produce a UV/texture map, so
    checking mesh.visual.uv (as an earlier version did) wrongly rejected the
    native colors every time. An even later edit called extract_mesh(
    has_vertex_color=False) directly and never baked colors at all, which is
    why meshes came out completely grey.
    """
    try:
        vc = mesh.visual.vertex_colors
        return vc is not None and len(vc) == len(mesh.vertices)
    except Exception:
        return False


def _sample_vertex_colors(mesh: trimesh.Trimesh, img: Image.Image) -> np.ndarray:
    """Bbox-projection color baking — last-resort fallback only."""
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    uv = (v - lo) / np.maximum(hi - lo, 1e-6)
    xs = np.clip((uv[:, 0] * (img.width - 1)).round().astype(int), 0, img.width - 1)
    ys = np.clip(((1 - uv[:, 1]) * (img.height - 1)).round().astype(int), 0, img.height - 1)
    return np.asarray(img)[ys, xs][:, :3]


def _extract_mesh_with_texture(
    model,
    scene_codes,
    isolated_rgba: Image.Image,
    mc_resolution: int,
) -> trimesh.Trimesh:
    """Extract the mesh with TripoSR's native per-vertex colors.

    has_vertex_color=True makes TSR sample colors from the triplane itself, so
    side/back-facing vertices get correct object colors (not background gray).
    Only if that path truly fails do we fall back to bbox baking from the
    WHITE-composited isolated foreground (never the gray Stage-4 composite,
    which would paint the object's silhouette gray).
    """
    try:
        mesh = model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=mc_resolution,
            threshold=config.TRIPOSR_MC_THRESHOLD,
        )[0]
        if not _has_vertex_colors(mesh):
            raise ValueError("TSR textured extraction returned no vertex colors")
        return mesh
    except Exception:
        mesh = model.extract_mesh(
            scene_codes,
            has_vertex_color=False,
            resolution=mc_resolution,
            threshold=config.TRIPOSR_MC_THRESHOLD,
        )[0]
        white = Image.new("RGBA", isolated_rgba.size, (255, 255, 255, 255))
        bake_source = Image.alpha_composite(white, isolated_rgba.convert("RGBA"))
        mesh.visual.vertex_colors = _sample_vertex_colors(mesh, bake_source)
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
        mesh = _extract_mesh_with_texture(model, scene_codes, rgba_img, mc_resolution)

    # Official demo applies this right after extraction — required for the
    # Gradio Model3D viewer to display the object upright, colors aligned.
    mesh = to_gradio_3d_orientation(mesh)
    return mesh


if __name__ == "__main__":
    from src.logger import log_feature_progress

    rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(rgba).ellipse([40, 40, 470, 470], fill=(200, 90, 40, 255))

    triposr_bg = composite_for_backend(rgba, "triposr")
    instant_bg = composite_for_backend(rgba, "instantmesh")
    assert np.asarray(triposr_bg)[0, 0].tolist() == [127, 127, 127]
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
            "per-backend BG colors and extracts the 3D mesh on the pinned DEVICE_3D "
            "with native per-vertex colors, oriented for the Gradio viewer."
        ),
        files_modified=["src/image_to_3d.py", "config.py"],
        technical_explanation=(
            "composite_for_backend reads BG colors exclusively from config (0.5 GRAY, "
            "matching the official TripoSR demo's fill_background — pure white left a "
            "faint SDF shell behind the object). TripoSR path: TSR lazy singleton from "
            "the snapshot_download cache, chunked rendering (8192) to bound VRAM. "
            "_extract_mesh_with_texture keeps TSR's native vertex colors (has_vertex_color="
            "True, threshold=TRIPOSR_MC_THRESHOLD); acceptance is verified via "
            "_has_vertex_colors (per-vertex colors), NOT a UV map, which TSR never "
            "produces. A previous edit called extract_mesh(has_vertex_color=False) "
            "directly with no baking step, producing fully grey meshes — restored here. "
            "Fallback bakes from the WHITE-composited isolated RGBA so silhouettes never "
            "absorb the gray background. After extraction, to_gradio_3d_orientation "
            "applies the official -90deg X / +90deg Y rotations. mc_resolution is "
            "caller-tunable (UI slider 128-512). MOCK_GPU returns an icosphere."
        ),
        duration_or_latency="TripoSR ~5-15s on T4 (textured path slightly higher)",
    )
    print(f"Step 4 verified: mesh with {len(mesh.vertices)} verts / {len(mesh.faces)} faces.")