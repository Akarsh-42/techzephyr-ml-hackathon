"""Stage 4 — 3D reconstruction: InstantMesh (primary) / TripoSR (fallback)."""
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

# InstantMesh's repo (third_party/InstantMesh, cloned by download_weights.py)
# also ships a top-level src/ package which would collide with OUR src/ on
# sys.path — the downloader renames it to im_src and exposes the util modules
# we need (im_src.utils.camera_util, im_src.utils.train_util).
_INSTANTMESH_SRC = Path(__file__).resolve().parent.parent / "third_party" / "InstantMesh"
if _INSTANTMESH_SRC.is_dir() and str(_INSTANTMESH_SRC) not in sys.path:
    sys.path.insert(0, str(_INSTANTMESH_SRC))

_triposr_model = None
_instantmesh_pipe = None
_instantmesh_recon = None


def composite_for_backend(rgba_img: Image.Image, backend: str) -> Image.Image:
    bg = config.BG_COLOR_INSTANTMESH if backend == "instantmesh" else config.BG_COLOR_TRIPOSR
    rgba = rgba_img.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*bg, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


# ---------------------------------------------------------------------------
# InstantMesh backend (primary)
# ---------------------------------------------------------------------------
def _load_instantmesh():
    """Load Zero123++ diffusion pipeline + FlexiCubes reconstruction model.

    Mirrors the official Space app.py: DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", custom_pipeline="zero123plus") with the custom
    white-background UNet from TencentARC/InstantMesh, plus instant_mesh_large
    .ckpt loaded via instantiate_from_config. Cached as lazy singletons.
    """
    global _instantmesh_pipe, _instantmesh_recon
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    if _instantmesh_pipe is None:
        unet_path = hf_hub_download(
            repo_id=config.INSTANTMESH_MODEL_ID,
            filename=config.INSTANTMESH_UNET_FILE,
            cache_dir=config.WEIGHTS_CACHE_DIR,
        )
        _instantmesh_pipe = DiffusionPipeline.from_pretrained(
            config.ZERO123PLUS_MODEL_ID,
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
            cache_dir=config.WEIGHTS_CACHE_DIR,
        )
        _instantmesh_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            _instantmesh_pipe.scheduler.config, timestep_spacing="trailing"
        )
        state_dict = torch.load(unet_path, map_location="cpu")
        _instantmesh_pipe.unet.load_state_dict(state_dict, strict=True)
        _instantmesh_pipe = _instantmesh_pipe.to(config.DEVICE_3D)

    if _instantmesh_recon is None:
        from im_src.utils.train_util import instantiate_from_config

        ckpt_path = hf_hub_download(
            repo_id=config.INSTANTMESH_MODEL_ID,
            filename=config.INSTANTMESH_CKPT_FILE,
            cache_dir=config.WEIGHTS_CACHE_DIR,
        )
        cfg_path = _INSTANTMESH_SRC / "configs" / f"{config.INSTANTMESH_CONFIG_NAME}.yaml"
        cfg = OmegaConf.load(cfg_path)
        model = instantiate_from_config(cfg.model_config)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[14:]: v
            for k, v in state_dict.items()
            if k.startswith("lrm_generator.") and "source_camera" not in k
        }
        model.load_state_dict(state_dict, strict=True)
        model = model.to(config.DEVICE_3D).eval()
        model.init_flexicubes_geometry(torch.device(config.DEVICE_3D), use_renderer=False)
        _instantmesh_recon = model

    return _instantmesh_pipe, _instantmesh_recon


def _generate_multiview(pipe, composite: Image.Image, seed: int) -> Image.Image:
    """Run Zero123++ to synthesize the 6-view image grid (960x640)."""
    from pytorch_lightning import seed_everything

    seed_everything(seed)
    return pipe(composite, num_inference_steps=config.INSTANTMESH_DIFFUSION_STEPS).images[0]


def _extract_instantmesh_mesh(recon, z123_image: Image.Image) -> trimesh.Trimesh:
    """FlexiCubes extraction with 6-view vertex colors (official make3d flow)."""
    from einops import rearrange
    from torchvision.transforms import v2

    from im_src.utils.camera_util import get_zero123plus_input_cameras

    images = np.asarray(z123_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)

    device = torch.device(config.DEVICE_3D)
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)

    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    with torch.no_grad():
        planes = recon.forward_planes(images, input_cameras)
        mesh_out = recon.extract_mesh(planes, use_texture_map=False)
    vertices, faces, vertex_colors = mesh_out

    # Official demo axis swap: InstantMesh vertex space -> glTF/gradio frame.
    vertices = vertices[:, [1, 2, 0]]

    return trimesh.Trimesh(
        vertices=vertices.cpu().numpy(),
        faces=faces.cpu().numpy(),
        vertex_colors=(vertex_colors.cpu().numpy() * 255).astype(np.uint8),
    )


def _reconstruct_instantmesh(rgba_img: Image.Image, seed: int) -> trimesh.Trimesh:
    pipe, recon = _load_instantmesh()
    composite = composite_for_backend(rgba_img, "instantmesh")
    z123_image = _generate_multiview(pipe, composite, seed)
    return _extract_instantmesh_mesh(recon, z123_image)


# ---------------------------------------------------------------------------
# TripoSR backend (fallback)
# ---------------------------------------------------------------------------
def to_gradio_3d_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """TripoSR's official viewer-orientation transform (tsr/utils.py).

    TSR meshes come out y-up in TSR's internal camera convention; the Gradio
    Model3D viewer (and most glTF consumers) expect a different frame. The
    official demo applies exactly these two rotations after extract_mesh.
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
    """True if the mesh carries a per-vertex color for every vertex."""
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
    """TripoSR native per-vertex colors; bbox baking as last resort."""
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


def _reconstruct_triposr(rgba_img: Image.Image, mc_resolution: int) -> trimesh.Trimesh:
    composite = composite_for_backend(rgba_img, "triposr")
    model = _load_triposr()
    device = torch.device(config.DEVICE_3D)
    with torch.no_grad():
        scene_codes = model([composite], device)
        mesh = _extract_mesh_with_texture(model, scene_codes, rgba_img, mc_resolution)

    # Official demo applies this right after extraction — required for the
    # Gradio Model3D viewer to display the object upright, colors aligned.
    mesh = to_gradio_3d_orientation(mesh)
    return mesh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def reconstruct_3d_mesh(
    rgba_img: Image.Image,
    backend: str = config.PRIMARY_3D_BACKEND,
    mc_resolution: int = 256,
    seed: int = 42,
) -> trimesh.Trimesh:
    if config.MOCK_GPU:
        return _mock_mesh()

    if backend == "instantmesh":
        return _reconstruct_instantmesh(rgba_img, seed)
    if backend == "triposr":
        return _reconstruct_triposr(rgba_img, mc_resolution)
    raise ValueError(f"Unknown 3D backend: {backend!r} (use 'instantmesh' or 'triposr').")


if __name__ == "__main__":
    rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(rgba).ellipse([40, 40, 470, 470], fill=(200, 90, 40, 255))

    triposr_bg = composite_for_backend(rgba, "triposr")
    instant_bg = composite_for_backend(rgba, "instantmesh")
    assert np.asarray(triposr_bg)[0, 0].tolist() == [127, 127, 127]
    assert np.asarray(instant_bg)[0, 0].tolist() == [255, 255, 255]

    mesh = reconstruct_3d_mesh(rgba)
    assert len(mesh.vertices) >= config.MIN_VERTICES
    assert len(mesh.faces) >= config.MIN_FACES
    assert mesh.visual is not None
    print(f"Step 4 verified: mesh with {len(mesh.vertices)} verts / {len(mesh.faces)} faces.")