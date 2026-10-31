"""Stage 2/3 — Sketch-guided 2D synthesis (ControlNet-Scribble + SD 1.5) + rembg isolation.
Empty user prompt falls back to DEFAULT_PROMPT (prompt is OPTIONAL per rules).
VRAM rule: unload this stage before any 3D model loads."""
import gc
import torch
from PIL import Image
from config import (SD_MODEL_ID, CONTROLNET_MODEL_ID, DEFAULT_PROMPT, NEGATIVE_PROMPT,
                   DIFFUSION_STEPS, DEFAULT_CONTROL_SCALE)

_pipe = None  # lazy singleton


def _get_pipeline():
    global _pipe
    if _pipe is None:
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)
        cn = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=torch.float16)
        _pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL_ID, controlnet=cn, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
        _pipe.scheduler = UniPCMultistepScheduler.from_config(_pipe.scheduler.config)
        if torch.cuda.is_available():
            _pipe.enable_model_cpu_offload()  # T4-safe; keeps VRAM low between stages
        else:
            _pipe = _pipe.to("cpu")
    return _pipe


def build_prompt(user_prompt: str) -> str:
    user_prompt = (user_prompt or "").strip()
    return f"{user_prompt}, {DEFAULT_PROMPT}" if user_prompt else DEFAULT_PROMPT


@torch.inference_mode()
def generate_reference_image(scribble_img: Image.Image, user_prompt: str = "",
                             control_scale: float = DEFAULT_CONTROL_SCALE,
                             seed: int = 42):
    """Returns (rendered_rgb, segmented_rgba). Never keeps 2D + 3D models on GPU together."""
    pipe = _get_pipeline()
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    rendered = pipe(prompt=build_prompt(user_prompt),
                    negative_prompt=NEGATIVE_PROMPT,
                    image=scribble_img,
                    num_inference_steps=DIFFUSION_STEPS,
                    controlnet_conditioning_scale=control_scale,
                    generator=gen).images[0]

    from rembg import remove
    segmented = remove(rendered)  # RGBA, transparent background

    # VRAM hygiene: free 2D before Stage 4 loads a 3D model
    _pipe.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rendered, segmented