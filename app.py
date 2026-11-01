"""Step 6 — Gradio 4.44.0 UI: Sketchpad, upload tab, prompt, gallery, 3D viewer, downloads."""
from __future__ import annotations

import gradio as gr

# Install the pinned Gradio/gradio-client schema compatibility guard before
# Blocks builds API metadata during launch. This is project code, not a venv edit.
import gradio_compat  # noqa: F401

from pipeline import run_pipeline


def generate(sketch, upload, prompt, seed):
    try:
        result = run_pipeline(sketch, upload, prompt or "", int(seed))
    except Exception as exc:
        return None, None, None, None, {"status": "FAILURE", "error": str(exc)}

    gallery = [
        (result["scribble"], "1. Conditioned scribble"),
        (result["shaded_2d"], "2. Shaded 2D render"),
        (result["isolated_rgba"], "3. Isolated foreground (85%)"),
    ]
    return gallery, result["glb_path"], result["glb_path"], result["obj_path"], result["report"]


with gr.Blocks(title="Sketch2Shape AI") as demo:
    gr.Markdown("# Sketch2Shape AI — draw a sketch, get a 3D model (.glb / .obj)")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Draw"):
                    sketchpad = gr.Sketchpad(label="Sketch here")
                with gr.Tab("Upload"):
                    upload = gr.Image(type="pil", label="Upload a sketch image")
            prompt = gr.Textbox(
                label="Optional appearance prompt (object, material, color, style)",
                placeholder="Example: a house with a bright pink roof, white walls, and dark windows",
                info="The sketch controls shape; this guides appearance and colour. Leave blank to use the default.",
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
            generate_btn = gr.Button("Generate 3D Model", variant="primary")
        with gr.Column():
            gallery = gr.Gallery(label="Pipeline intermediates", columns=3, height=240)
            viewer = gr.Model3D(label="3D preview")
            with gr.Row():
                glb_out = gr.File(label="Download .glb")
                obj_out = gr.File(label="Download .obj")
            report = gr.JSON(label="Validation & timing report")

    generate_btn.click(
        generate,
        inputs=[sketchpad, upload, prompt, seed],
        outputs=[gallery, viewer, glb_out, obj_out, report],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
