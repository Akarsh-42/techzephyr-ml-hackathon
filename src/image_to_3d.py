"""Stage 4 — InstantMesh 3D reconstruction: Zero123++ multi-view diffusion + LRM FlexiCubes.

Replaces the TripoSR backend with InstantMesh for significantly better mesh quality.
Pipeline:
  1. Zero123++ generates 6 consistent multi-view images from the isolated foreground.
  2. LRM (Large Reconstruction Model) with FlexiCubes extracts a vertex-colored mesh.

NOTE: InstantMesh's source tree has its own `src/` package which conflicts with this
project's `src/`. We handle this by temporarily swapping the `src` namespace in
sys.modules during model loading, and inlining small utility functions.
"""
from __future__ import annotations

import gc
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

import config

# InstantMesh's repo is source-only, so it lives in a git-cloned checkout
# (third_party/InstantMesh, created by download_weights.py).
_INSTANTMESH_SRC = Path(__file__).resolve().parent.parent / "third_party" / "InstantMesh"

_diffusion_pipeline = None
_recon_model = None


# ---------------------------------------------------------------------------
# Inlined from InstantMesh's src/utils/camera_util.py (avoids namespace clash)
# ---------------------------------------------------------------------------

def _center_looking_at_camera_pose(camera_position, look_at=None, up_world=None):
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)
    z_axis = F.normalize(camera_position - look_at, dim=-1).float()
    x_axis = F.normalize(torch.linalg.cross(up_world, z_axis, dim=-1), dim=-1).float()
    y_axis = F.normalize(torch.linalg.cross(z_axis, x_axis, dim=-1), dim=-1).float()
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    return torch.cat([extrinsics, padding], dim=-2)


def _get_zero123plus_input_cameras(batch_size=1, radius=4.0, fov=30.0):
    """Build the 6 camera matrices that match Zero123++'s output view layout."""
    azimuths = np.array([30, 90, 150, 210, 270, 330], dtype=float)
    elevations = np.array([20, -10, 20, -10, 20, -10], dtype=float)
    az_rad = np.deg2rad(azimuths)
    el_rad = np.deg2rad(elevations)
    xs = radius * np.cos(el_rad) * np.cos(az_rad)
    ys = radius * np.cos(el_rad) * np.sin(az_rad)
    zs = radius * np.sin(el_rad)
    cam_locs = torch.from_numpy(np.stack([xs, ys, zs], axis=-1)).float()
    c2ws = _center_looking_at_camera_pose(cam_locs).float().flatten(-2)

    focal = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    K = torch.tensor([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]]).float()
    Ks = K.unsqueeze(0).repeat(6, 1, 1).flatten(-2)

    extrinsics = c2ws[:, :12]
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)
    return cameras.unsqueeze(0).repeat(batch_size, 1, 1)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def composite_for_backend(rgba_img: Image.Image, backend: str) -> Image.Image:
    """Composite RGBA onto the correct background color for the chosen backend."""
    bg = config.BG_COLOR_INSTANTMESH if backend == "instantmesh" else config.BG_COLOR_TRIPOSR
    rgba = rgba_img.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*bg, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def _load_diffusion_pipeline():
    """Load the Zero123++ multi-view diffusion pipeline (lazy singleton)."""
    global _diffusion_pipeline
    if _diffusion_pipeline is not None:
        return _diffusion_pipeline

    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    from huggingface_hub import hf_hub_download

    print("Loading Zero123++ diffusion pipeline ...")
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )

    # Load custom white-background UNet from InstantMesh weights
    unet_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="diffusion_pytorch_model.bin",
        repo_type="model",
    )
    state_dict = torch.load(unet_ckpt_path, map_location="cpu", weights_only=False)
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(config.DEVICE_3D)
    _diffusion_pipeline = pipeline
    print("Zero123++ loaded.")
    return _diffusion_pipeline


def _unload_diffusion_pipeline():
    """Free the Zero123++ pipeline to reclaim VRAM for the reconstruction model."""
    global _diffusion_pipeline
    _diffusion_pipeline = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_recon_model():
    """Load the InstantMesh LRM reconstruction model (lazy singleton).

    Uses a temporary sys.modules swap so that InstantMesh's `src.*` imports
    resolve to its own source tree rather than this project's `src/` package.
    """
    global _recon_model
    if _recon_model is not None:
        return _recon_model

    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    # --- Save project's `src` namespace and temporarily hand it to InstantMesh ---
    saved_src_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    }
    im_path = str(_INSTANTMESH_SRC)
    if im_path not in sys.path:
        sys.path.insert(0, im_path)

    try:
        print("Loading InstantMesh reconstruction model ...")
        config_path = _INSTANTMESH_SRC / "configs" / "instant-mesh-large.yaml"
        cfg = OmegaConf.load(str(config_path))

        # Inline instantiate_from_config: import the model class and construct it
        target = cfg.model_config.target  # "src.models.lrm_mesh.InstantMesh"
        module_name, cls_name = target.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        model_cls = getattr(mod, cls_name)
        model = model_cls(**cfg.model_config.get("params", {}))

        # Load weights
        model_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="instant_mesh_large.ckpt",
            repo_type="model",
        )
        state_dict = torch.load(model_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
        model.load_state_dict(state_dict, strict=True)

        device = torch.device(config.DEVICE_3D)
        model = model.to(device)
        model.init_flexicubes_geometry(device, fovy=30.0)
        model = model.eval()
        _recon_model = model
        print("InstantMesh reconstruction model loaded.")
    finally:
        # --- Restore project's `src` namespace ---
        # Remove InstantMesh's src modules from sys.modules
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                sys.modules.pop(k, None)
        # Put the project's src modules back
        sys.modules.update(saved_src_modules)

    return _recon_model


def _mock_mesh() -> trimesh.Trimesh:
    """Return a vertex-colored icosphere for CPU/mock-GPU testing."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
    mesh.visual.vertex_colors = ((mesh.vertex_normals * 0.5 + 0.5) * 255).astype(np.uint8)
    return mesh


def _generate_multiview(rgba_img: Image.Image, seed: int = 42, diffusion_steps: int = 75):
    """Run Zero123++ to generate 6 multi-view images from a single RGBA input.

    Returns a tensor of shape (6, 3, 320, 320) with pixel values in [0, 1].
    """
    from einops import rearrange

    pipeline = _load_diffusion_pipeline()

    # CRITICAL: Composite RGBA onto WHITE before passing to Zero123++.
    # InstantMesh's custom UNet was trained on white backgrounds. Newer
    # diffusers may drop the alpha channel (transparent → black) instead
    # of compositing onto white, causing the model to reconstruct the
    # black background as geometry (the "crumpled paper" artifact).
    input_rgb = Image.alpha_composite(
        Image.new("RGBA", rgba_img.size, (255, 255, 255, 255)),
        rgba_img.convert("RGBA"),
    ).convert("RGB")

    generator = torch.Generator(device=config.DEVICE_3D).manual_seed(seed)
    output_image = pipeline(
        input_rgb,
        num_inference_steps=diffusion_steps,
        generator=generator,
    ).images[0]

    # Split the 960x640 grid into 6 individual 320x320 views (3 rows x 2 cols)
    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)     # (6, 3, 320, 320)

    return images, output_image


def _reconstruct_from_multiview(multiview_images: torch.Tensor) -> trimesh.Trimesh:
    """Run the LRM reconstruction model on 6 multi-view images to extract a mesh.

    Args:
        multiview_images: tensor of shape (6, 3, 320, 320), values in [0, 1].

    Returns:
        A trimesh.Trimesh with vertex colors.
    """
    from torchvision.transforms import v2

    model = _load_recon_model()
    device = torch.device(config.DEVICE_3D)

    input_cameras = _get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)

    images = multiview_images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)

        vertices, faces, vertex_colors = model.extract_mesh(
            planes,
            use_texture_map=False,
            texture_resolution=1024,
            render_resolution=512,
        )

    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()

    # Coordinate flip for GLB compatibility (InstantMesh convention → standard)
    vertices_np = vertices_np @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)

    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_colors=vertex_colors,
    )
    return mesh


def reconstruct_3d_mesh(
    rgba_img: Image.Image,
    backend: str = config.PRIMARY_3D_BACKEND,
    mc_resolution: int = 256,
    seed: int = 42,
) -> trimesh.Trimesh:
    """Full Stage 4: single RGBA image -> 3D mesh via InstantMesh.

    Args:
        rgba_img: Isolated RGBA foreground from Stage 3.
        backend: 3D backend name (kept for pipeline compatibility).
        mc_resolution: Unused (kept for API compatibility with TripoSR calls).
        seed: Random seed for reproducible multi-view generation.

    Returns:
        A trimesh.Trimesh with vertex colors, ready for Stage 5 validation.
    """
    if config.MOCK_GPU:
        return _mock_mesh()

    # Step 1: Generate 6 multi-view images via Zero123++
    multiview_images, _grid_img = _generate_multiview(rgba_img, seed=seed)

    # Step 2: Free the diffusion pipeline to make room for the reconstruction model
    _unload_diffusion_pipeline()

    # Step 3: Reconstruct mesh from multi-view images via LRM + FlexiCubes
    mesh = _reconstruct_from_multiview(multiview_images)

    return mesh


if __name__ == "__main__":
    from src.logger import log_feature_progress

    rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(rgba).ellipse([40, 40, 470, 470], fill=(200, 90, 40, 255))

    mesh = reconstruct_3d_mesh(rgba)
    assert len(mesh.vertices) >= config.MIN_VERTICES
    assert len(mesh.faces) >= config.MIN_FACES
    assert mesh.visual.vertex_colors is not None

    log_feature_progress(
        feature_name="Step 4 — InstantMesh 3D Reconstruction (Zero123++ + LRM FlexiCubes)",
        role_owner="Role B (3D Lead)",
        role_importance_and_job=(
            "Core engine for 3D Output Quality (25 pts) & Efficiency (20 pts): uses Zero123++ "
            "multi-view diffusion to generate 6 consistent views, then LRM with FlexiCubes "
            "to extract high-quality vertex-colored meshes on the pinned DEVICE_3D."
        ),
        files_modified=["src/image_to_3d.py", "requirements.txt", "download_weights.py"],
        technical_explanation=(
            "InstantMesh pipeline: (1) Zero123++ (sudo-ai/zero123plus-v1.2 with custom UNet "
            "from TencentARC/InstantMesh) generates a 960x640 grid of 6 views (3x2 layout, "
            "each 320x320) in fp16 with EulerAncestralDiscreteScheduler. (2) Diffusion pipeline "
            "is unloaded to free VRAM. (3) LRM (instant-mesh-large config: DINOv1-ViT-B/16 "
            "encoder, 16-layer transformer, 64-res triplane, FlexiCubes grid_res=128) predicts "
            "SDF + deformation on a tetrahedral grid, extracts mesh via FlexiCubes, and queries "
            "vertex colors from the triplane. Coordinate flip applied for GLB compatibility. "
            "MOCK_GPU returns a vertex-colored icosphere for CPU testing."
        ),
        duration_or_latency="MOCK (CPU) verified; InstantMesh ~30-60s on T4",
    )
    print(f"Step 4 verified: mesh with {len(mesh.vertices)} verts / {len(mesh.faces)} faces.")
