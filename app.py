"""Step 6 — Gradio 4.44.0 UI: Sketchpad, upload tab, prompt, gallery, 3D viewer, downloads."""
from __future__ import annotations

import gradio as gr

# Install the pinned Gradio/gradio-client schema compatibility guard before
# Blocks builds API metadata during launch. This is project code, not a venv edit.
import gradio_compat  # noqa: F401

from pipeline import run_pipeline


APP_CSS = r'''
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root, .dark {
    --color-accent: #22C55E;
    --primary-500: #22C55E;
    --primary-600: #16A34A;
}
body, button, input, textarea, select, .gradio-container {
    font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;
}
#app-title, #app-title * {
    font-family: "Space Grotesk", "Inter", ui-sans-serif, system-ui, sans-serif !important;
}
button.primary {
    background: #15803D !important;
    border-color: #15803D !important;
    color: #fff !important;
}
button.primary:hover {
    background: #166534 !important;
    border-color: #166534 !important;
}
input[type="range"] {
    accent-color: #22C55E !important;
}
#sketchpad {
    background: #fff !important;
}
#sketchpad .image-editor,
#sketchpad .image-container,
#sketchpad .canvas-container {
    background: #fff !important;
}
#sketchpad canvas {
    background: transparent !important;
    transform: none !important;
}
'''


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


def update_viewer_camera(model_path, zoom, orbit, tilt):
    """Update only the Gradio viewer camera; never re-run model generation."""
    if not model_path:
        return gr.update()
    radius = 3.0 / max(float(zoom or 1.0), 0.5)
    return gr.update(
        value=model_path,
        camera_position=(float(orbit or 0), float(tilt or 75), radius),
    )


with gr.Blocks(title="Sketch2Shape AI", css=APP_CSS) as demo:
    gr.Markdown(
        "# Sketch2Shape AI — draw a sketch, get a 3D model (.glb / .obj)",
        elem_id="app-title",
    )
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Draw"):
                    sketchpad = gr.Sketchpad(label="Sketch here", elem_id="sketchpad")
                with gr.Tab("Upload"):
                    upload = gr.Image(type="pil", label="Upload a sketch image")
            prompt = gr.Textbox(
                label="Object, material & color instructions",
                placeholder="e.g. a chair with a red seat, blue backrest, and black metal legs",
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
            generate_btn = gr.Button("Generate 3D Model", variant="primary")
        with gr.Column():
            gallery = gr.Gallery(label="Pipeline intermediates", columns=3, height=240)
            viewer = gr.Model3D(
                label="3D preview",
                elem_id="model-viewer",
                camera_position=(0, 75, 3.0),
            )
            with gr.Row():
                zoom_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                    label="Zoom",
                    info="0.5× to 2×",
                )
                orbit_slider = gr.Slider(
                    minimum=0, maximum=360, value=0, step=1,
                    label="Orbit",
                    info="Horizontal rotation",
                )
                tilt_slider = gr.Slider(
                    minimum=15, maximum=165, value=75, step=1,
                    label="Tilt",
                    info="Vertical rotation",
                )
            with gr.Row():
                glb_out = gr.File(label="Download .glb")
                obj_out = gr.File(label="Download .obj")
            report = gr.JSON(label="Validation & timing report")

    for camera_control in (zoom_slider, orbit_slider, tilt_slider):
        camera_control.release(
            update_viewer_camera,
            inputs=[viewer, zoom_slider, orbit_slider, tilt_slider],
            outputs=[viewer],
            queue=False,
        )

    generate_btn.click(
        generate,
        inputs=[sketchpad, upload, prompt, seed],
        outputs=[gallery, viewer, glb_out, obj_out, report],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
