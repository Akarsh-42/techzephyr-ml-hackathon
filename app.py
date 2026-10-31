"""Step 6 — Gradio UI: Sketchpad, upload tab, prompt, gallery, 3D viewer, downloads."""
from __future__ import annotations

import gradio as gr

# Install the pinned Gradio/gradio-client schema compatibility guard before
# Blocks builds API metadata during launch. This is project code, not a venv edit.
import gradio_compat  # noqa: F401

from pipeline import run_pipeline
# ---------------------------------------------------------------------------
# Compatibility patch: gradio_client 1.3.0 (pinned by gradio 4.44.0) crashes
# with "TypeError: argument of type 'bool' is not iterable" while building API
# info, because Sketchpad/Gallery components emit boolean `additionalProperties`
# flags in their JSON schemas and get_type() assumes every schema is a dict
# (github.com/gradio-app/gradio/issues/11084). Fixed upstream only in gradio 5.x;
# the blueprint pins gradio 4.44.0, so we harden the parser here instead.
# Must run BEFORE any gradio component is instantiated.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


_gcu.get_type = _safe_get_type

from pipeline import run_pipeline  # noqa: E402  (import after patch, on purpose)


def generate(sketch, upload, prompt, seed, mc_resolution):
    try:
        result = run_pipeline(
            sketch,
            upload,
            prompt or "",
            int(seed),
            int(mc_resolution),
        )
    except Exception as exc:
        return None, None, None, None, {"status": "FAILURE", "error": str(exc)}

    gallery = [
        (result["scribble"], "1. Conditioned scribble"),
        (result["shaded_2d"], "2. Shaded 2D render"),
        (result["isolated_rgba"], "3. Isolated foreground (85%)"),
    ]
    return gallery, result["glb_path"], result["glb_path"], result["obj_path"], result["report"]


with gr.Blocks(title="Sketch2Shape AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎨 Sketch2Shape AI — draw a sketch, get a 3D model\n"
        "Draw an object outline on the left (or upload a sketch), optionally describe it, "
        "then hit **Generate 3D Model**. You'll get a textured 3D mesh (.glb / .obj) "
        "you can rotate in the viewer and download."
    )
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Draw"):
                    sketchpad = gr.Sketchpad(label="Sketch here")
                with gr.Tab("Upload"):
                    upload = gr.Image(type="pil", label="Upload a sketch image")
            prompt = gr.Textbox(
                label="Optional prompt (object, material, color, style)",
                placeholder="e.g. a wooden bowl, glossy red ceramic vase, chrome robot toy",
            )
            with gr.Row():
                seed = gr.Number(value=42, label="Seed", precision=0)
                mc_resolution = gr.Slider(
                    minimum=128,
                    maximum=512,
                    value=256,
                    step=128,
                    label="Mesh detail",
                    info="Higher = finer surface, slower generation",
                )
            generate_btn = gr.Button("Generate 3D Model", variant="primary")
            gr.Markdown(
                "💡 **Tips:** draw one clear object with a closed outline; thick strokes "
                "work best. Same seed = same result (deterministic demos)."
            )
        with gr.Column():
            gallery = gr.Gallery(label="Pipeline intermediates", columns=3, height=240)
            viewer = gr.Model3D(label="3D preview")
            with gr.Row():
                glb_out = gr.File(label="Download .glb")
                obj_out = gr.File(label="Download .obj")
            report = gr.JSON(label="Validation & timing report")

    generate_btn.click(
        generate,
        inputs=[sketchpad, upload, prompt, seed, mc_resolution],
        outputs=[gallery, viewer, glb_out, obj_out, report],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=True)
