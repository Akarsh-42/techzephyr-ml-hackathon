"""Step 6 — Gradio 4.44.0 UI: Sketchpad, upload tab, prompt, gallery, 3D viewer, downloads."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import trimesh

# Install the pinned Gradio/gradio-client schema compatibility guard before
# Blocks builds API metadata during launch. This is project code, not a venv edit.
import gradio_compat  # noqa: F401

from pipeline import run_pipeline


PREVIEW_AXIS_REMAP = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def make_preview_model(glb_path):
    """Create a Gradio-only orientation-corrected copy; leave downloads intact."""
    source_path = Path(glb_path)
    preview_path = source_path.with_name(f"{source_path.stem}_viewer.glb")
    mesh = trimesh.load(str(source_path), force="mesh", process=False)
    mesh.apply_transform(PREVIEW_AXIS_REMAP)
    mesh.export(str(preview_path), file_type="glb")
    return str(preview_path)


APP_CSS = '@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap\');\n\n:root, .dark {\n    --color-accent: #22C55E;\n    --primary-500: #22C55E;\n    --primary-600: #16A34A;\n    --body-background-fill: #0B1220;\n    --background-fill-primary: #111827;\n    --background-fill-secondary: #0F172A;\n    --border-color-primary: #263246;\n}\nbody, button, input, textarea, select, .gradio-container {\n    font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;\n}\n.gradio-container {\n    max-width: none !important;\n    min-height: 100vh;\n    color: #E5E7EB !important;\n    background: radial-gradient(ellipse at 15% 0%, rgba(34, 197, 94, .10), transparent 36%), #0B1220 !important;\n}\n#studio-header {\n    max-width: 1440px;\n    margin: 0 auto 1.25rem auto;\n    padding: 1.4rem 1.5rem 1rem;\n    border-bottom: 1px solid #263246;\n}\n#studio-header h1, #studio-header * {\n    font-family: "Space Grotesk", "Inter", ui-sans-serif, system-ui, sans-serif !important;\n}\n#studio-header h1 {\n    color: #F8FAFC;\n    font-size: clamp(1.8rem, 3vw, 2.65rem);\n    letter-spacing: -.04em;\n    margin: .15rem 0;\n}\n#studio-header p {\n    color: #94A3B8;\n    font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;\n    margin: .25rem 0 0;\n}\n#studio-layout {\n    max-width: 1440px;\n    margin: 0 auto !important;\n    gap: 1.1rem !important;\n    align-items: stretch;\n}\n#input-panel, #output-panel {\n    background: rgba(17, 24, 39, .94) !important;\n    border: 1px solid #263246 !important;\n    border-radius: 20px !important;\n    padding: 1.25rem !important;\n    box-shadow: 0 18px 50px rgba(0, 0, 0, .22);\n}\n#input-panel h3, #output-panel h3 {\n    color: #F8FAFC;\n    font-family: "Space Grotesk", "Inter", sans-serif !important;\n}\n#input-panel label, #output-panel label { color: #CBD5E1 !important; }\n#input-panel .tab-nav button.selected {\n    color: #86EFAC !important;\n    border-color: #22C55E !important;\n}\n#sketchpad {\n    width: 100% !important;\n    background: #fff !important;\n    border: 1px solid #CBD5E1 !important;\n    border-radius: 14px !important;\n    overflow: hidden !important;\n}\n#sketchpad .image-editor,\n#sketchpad .image-container,\n#sketchpad .canvas-container { background: #fff !important; }\n#sketchpad canvas {\n    background: transparent !important;\n    touch-action: none;\n}\n#input-panel input, #input-panel textarea {\n    background: #0B1220 !important;\n    color: #F8FAFC !important;\n    border-color: #334155 !important;\n}\n#input-panel input::placeholder, #input-panel textarea::placeholder { color: #94A3B8 !important; }\n#generate-btn {\n    background: #15803D !important;\n    border-color: #15803D !important;\n    color: #fff !important;\n    font-weight: 700 !important;\n    min-height: 3.25rem;\n}\n#generate-btn:hover { background: #166534 !important; border-color: #166534 !important; }\ninput[type="range"] { accent-color: #22C55E !important; }\n#camera-controls {\n    background: #0B1220 !important;\n    border: 1px solid #263246;\n    border-radius: 14px;\n    padding: .65rem;\n}\n#camera-controls label { color: #CBD5E1 !important; }\n#downloads { gap: .75rem; }\n#model-viewer, #pipeline-gallery { border-radius: 14px !important; }\n@media (max-width: 900px) {\n    #studio-layout { flex-direction: column !important; gap: .85rem !important; }\n    #input-panel, #output-panel { padding: 1rem !important; }\n}\n@media (max-width: 600px) {\n    #camera-controls { flex-direction: column !important; }\n    #studio-header { padding: 1rem .5rem; }\n}\n'


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
    try:
        preview_path = make_preview_model(result["glb_path"])
    except Exception as exc:
        preview_path = result["glb_path"]
        result["report"]["preview_orientation_warning"] = str(exc)
    return gallery, preview_path, result["glb_path"], result["obj_path"], result["report"]


def update_viewer_camera(model_path, zoom, orbit, tilt):
    """Update only the Gradio viewer camera; never re-run model generation."""
    if not model_path:
        return gr.update()
    radius = 3.0 / max(float(zoom or 1.0), 0.5)
    return gr.update(
        value=model_path,
        camera_position=(float(orbit or 0), float(tilt or 75), radius),
    )


def set_front_view(model_path, zoom):
    """Aim the viewer at the canonical InstantMesh front without regeneration."""
    if not model_path:
        return gr.update(), 90, 75
    radius = 3.0 / max(float(zoom or 1.0), 0.5)
    return (
        gr.update(value=model_path, camera_position=(90.0, 75.0, radius)),
        90,
        75,
    )


with gr.Blocks(
    title="Sketch2Shape Studio",
    css=APP_CSS,
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
) as demo:
    gr.Markdown(
        "# Sketch2Shape Studio\n### A sketch in. A colored 3D model out.",
        elem_id="studio-header",
    )
    with gr.Row(elem_id="studio-layout"):
        with gr.Column(scale=5, min_width=360, elem_id="input-panel"):
            gr.Markdown("### 01 · Create your sketch")
            gr.Markdown("Draw on the white canvas or switch to upload. Add part-specific color notes below.")
            with gr.Tabs():
                with gr.Tab("Draw"):
                    sketchpad = gr.Sketchpad(
                        label="Sketch canvas",
                        elem_id="sketchpad",
                        canvas_size=(640, 480),
                        width="100%",
                        height=550,
                    )
                with gr.Tab("Upload"):
                    upload = gr.Image(type="pil", label="Upload a sketch image", height=550)
            prompt = gr.Textbox(
                label="Object and color notes",
                placeholder="e.g. a chair with a red seat, blue backrest, and black metal legs",
                lines=2,
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
            generate_btn = gr.Button(
                "Generate 3D model",
                variant="primary",
                elem_id="generate-btn",
            )
        with gr.Column(scale=7, min_width=420, elem_id="output-panel"):
            gr.Markdown("### 02 · Inspect your model")
            gallery = gr.Gallery(
                label="Pipeline",
                columns=3,
                height=200,
                elem_id="pipeline-gallery",
            )
            viewer = gr.Model3D(
                label="Interactive 3D preview",
                elem_id="model-viewer",
                height=430,
                camera_position=(90, 75, 3.0),
            )
            gr.Markdown("#### Camera controls")
            with gr.Row(elem_id="camera-controls"):
                zoom_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                    label="Zoom",
                )
                orbit_slider = gr.Slider(
                    minimum=0, maximum=360, value=90, step=1,
                    label="Orbit",
                )
                tilt_slider = gr.Slider(
                    minimum=15, maximum=165, value=75, step=1,
                    label="Tilt",
                )
            front_view_btn = gr.Button("Front view", variant="secondary")
            with gr.Row(elem_id="downloads"):
                glb_out = gr.File(label="Download .glb")
                obj_out = gr.File(label="Download .obj")
            with gr.Accordion("Validation and timing report", open=False):
                report = gr.JSON(show_label=False)

    for camera_control in (zoom_slider, orbit_slider, tilt_slider):
        camera_control.release(
            update_viewer_camera,
            inputs=[viewer, zoom_slider, orbit_slider, tilt_slider],
            outputs=[viewer],
            queue=False,
        )

    front_view_btn.click(
        set_front_view,
        inputs=[viewer, zoom_slider],
        outputs=[viewer, orbit_slider, tilt_slider],
        queue=False,
    )

    generate_btn.click(
        generate,
        inputs=[sketchpad, upload, prompt, seed],
        outputs=[gallery, viewer, glb_out, obj_out, report],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
