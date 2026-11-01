"""Step 6 — Gradio 4.44.0 UI: Sketchpad, upload tab, prompt, gallery, 3D viewer, downloads."""
from __future__ import annotations

import gradio as gr

# Install the pinned Gradio/gradio-client schema compatibility guard before
# Blocks builds API metadata during launch. This is project code, not a venv edit.
import gradio_compat  # noqa: F401

from pipeline import run_pipeline

APP_CSS = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root, .dark {
    --color-accent: #22C55E;
    --primary-500: #22C55E;
    --primary-600: #16A34A;
    --body-background-fill: #0B1220;
    --background-fill-primary: #111827;
    --background-fill-secondary: #0F172A;
    --border-color-primary: #263246;
}
body, button, input, textarea, select, .gradio-container {
    font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;
}
.gradio-container {
    max-width: none !important;
    min-height: 100vh;
    color: #E5E7EB !important;
    background: radial-gradient(ellipse at 15% 0%, rgba(34, 197, 94, .10), transparent 36%), #0B1220 !important;
}

/* ---------- header ---------- */
#studio-header {
    max-width: 1440px;
    margin: 0 auto .9rem auto;
    padding: 1rem 1.25rem .8rem;
    border-bottom: 1px solid #263246;
}
#studio-header h1 {
    font-family: "Space Grotesk", "Inter", sans-serif !important;
    color: #F8FAFC;
    font-size: clamp(1.5rem, 2.6vw, 2.1rem);
    letter-spacing: -.04em;
    margin: .1rem 0;
}
#studio-header p { color: #94A3B8; margin: .2rem 0 0; }

/* ---------- stage markers (annotation strip) ---------- */
#stage-strip {
    max-width: 1440px;
    margin: 0 auto .9rem auto;
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
}
.stage-chip {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    background: rgba(17, 24, 39, .92);
    border: 1px solid #263246;
    border-radius: 999px;
    padding: .35rem .85rem;
    font-size: .78rem;
    font-weight: 500;
    color: #94A3B8;
    white-space: nowrap;
}
.stage-chip .n {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 50%;
    background: rgba(34, 197, 94, .14);
    border: 1px solid rgba(34, 197, 94, .45);
    color: #86EFAC;
    font-size: .68rem;
    font-weight: 700;
}
.stage-chip b { color: #E2E8F0; font-weight: 600; }

/* ---------- layout panels ---------- */
#studio-layout {
    max-width: 1440px;
    margin: 0 auto !important;
    gap: .9rem !important;
    align-items: stretch;
}
#input-panel, #output-panel {
    background: rgba(17, 24, 39, .94) !important;
    border: 1px solid #263246 !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    box-shadow: 0 14px 40px rgba(0, 0, 0, .22);
}
.section-title {
    display: flex;
    align-items: baseline;
    gap: .5rem;
    margin: 0 0 .55rem 0;
    font-family: "Space Grotesk", "Inter", sans-serif !important;
    color: #F8FAFC;
    font-size: 1.02rem;
    font-weight: 600;
}
.section-title .step {
    color: #22C55E;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    background: rgba(34, 197, 94, .12);
    border: 1px solid rgba(34, 197, 94, .35);
    border-radius: 6px;
    padding: .1rem .4rem;
}
.hint { color: #64748B; font-size: .8rem; margin: -.35rem 0 .55rem 0; }

/* ---------- compact controls ---------- */
#input-panel label, #output-panel label {
    color: #CBD5E1 !important;
    font-size: .85rem !important;
    margin-bottom: .2rem !important;
}
#input-panel .tab-nav button { font-size: .85rem; padding: .4rem .8rem; }
#input-panel .tab-nav button.selected { color: #86EFAC !important; border-color: #22C55E !important; }
#sketchpad {
    width: 100% !important;
    background: #fff !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
#sketchpad .image-editor, #sketchpad .image-container, #sketchpad .canvas-container { background: #fff !important; }
#sketchpad canvas { background: transparent !important; touch-action: none; }
#input-panel input, #input-panel textarea {
    background: #0B1220 !important;
    color: #F8FAFC !important;
    border-color: #334155 !important;
    font-size: .9rem !important;
}
#input-panel input::placeholder, #input-panel textarea::placeholder { color: #94A3B8 !important; }
#seed-box input { max-width: 9rem; }

/* ---------- generate button ---------- */
#generate-btn {
    background: linear-gradient(90deg, #15803D, #16A34A) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: .01em;
    min-height: 2.9rem;
    border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(34, 197, 94, .25);
    transition: transform .15s ease, box-shadow .15s ease;
}
#generate-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(34, 197, 94, .38);
}
input[type="range"] { accent-color: #22C55E !important; }

/* ---------- output side ---------- */
#pipeline-gallery { border-radius: 12px !important; }
#pipeline-gallery .caption { font-size: .72rem !important; }
#model-viewer { border-radius: 12px !important; background: #0B1220 !important; }
#camera-controls {
    background: #0B1220 !important;
    border: 1px solid #263246;
    border-radius: 10px;
    padding: .5rem .7rem;
}
#camera-controls label { font-size: .78rem !important; }
#downloads { gap: .6rem; }
#downloads .file { padding: .35rem .6rem; }
#downloads label { font-size: .78rem !important; }

/* ---------- footer ---------- */
#studio-footer {
    max-width: 1440px;
    margin: .9rem auto 0 auto;
    padding: .7rem 1.25rem;
    border-top: 1px solid #263246;
    color: #64748B;
    font-size: .74rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: .5rem;
}

@media (max-width: 900px) {
    #studio-layout { flex-direction: column !important; gap: .7rem !important; }
    #input-panel, #output-panel { padding: .8rem !important; }
    #studio-footer { flex-direction: column; }
}
"""

STAGE_STRIP = """
<div id="stage-strip">
  <span class="stage-chip"><span class="n">1</span><b>Sketch Prep</b>&nbsp;denoise &amp; binarize</span>
  <span class="stage-chip"><span class="n">2</span><b>ControlNet</b>&nbsp;shaded 2D render</span>
  <span class="stage-chip"><span class="n">3</span><b>Isolation</b>&nbsp;rembg, 85% frame</span>
  <span class="stage-chip"><span class="n">4</span><b>InstantMesh</b>&nbsp;6-view 3D + colors</span>
  <span class="stage-chip"><span class="n">5</span><b>Validate</b>&nbsp;export &amp; reload-check</span>
</div>
"""


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


with gr.Blocks(
    title="Sketch2Shape Studio",
    css=APP_CSS,
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
) as demo:
    gr.Markdown(
        "# Sketch2Shape Studio\n### A sketch in. A colored 3D model out.",
        elem_id="studio-header",
    )
    gr.HTML(STAGE_STRIP)

    with gr.Row(elem_id="studio-layout"):
        # ---------------- left: input ----------------
        with gr.Column(scale=5, min_width=340, elem_id="input-panel"):
            gr.HTML(
                "<div class='section-title'><span class='step'>Step 1</span>Create your sketch</div>"
            )
            gr.HTML(
                "<div class='hint'>Draw one object with a closed outline — thick strokes work best. "
                "Or switch to <b>Upload</b> for an existing image.</div>"
            )
            with gr.Tabs():
                with gr.Tab("Draw"):
                    sketchpad = gr.Sketchpad(
                        label="Sketch canvas",
                        elem_id="sketchpad",
                        canvas_size=(640, 480),
                        width="100%",
                        height=440,
                    )
                with gr.Tab("Upload"):
                    upload = gr.Image(type="pil", label="Upload a sketch image", height=440)
            gr.HTML(
                "<div class='section-title'><span class='step'>Step 2</span>Describe it</div>"
            )
            prompt = gr.Textbox(
                show_label=False,
                placeholder="Optional — e.g. a chair with a red seat, blue backrest, black metal legs",
                lines=2,
            )
            with gr.Row(elem_id="seed-box"):
                seed = gr.Number(value=42, label="Seed (same seed = same model)", precision=0)
            generate_btn = gr.Button("Generate 3D model", variant="primary", elem_id="generate-btn")

        # ---------------- right: output ----------------
        with gr.Column(scale=7, min_width=420, elem_id="output-panel"):
            gr.HTML(
                "<div class='section-title'><span class='step'>Step 3</span>Inspect your model</div>"
            )
            viewer = gr.Model3D(
                label="Interactive 3D preview",
                elem_id="model-viewer",
                height=400,
                camera_position=(0, 75, 3.0),
            )
            with gr.Accordion("Camera controls", open=False):
                with gr.Row(elem_id="camera-controls"):
                    zoom_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Zoom"
                    )
                    orbit_slider = gr.Slider(
                        minimum=0, maximum=360, value=0, step=1, label="Orbit"
                    )
                    tilt_slider = gr.Slider(
                        minimum=15, maximum=165, value=75, step=1, label="Tilt"
                    )
            gr.HTML(
                "<div class='section-title'><span class='step'>Pipeline</span>How your sketch became 3D</div>"
            )
            gallery = gr.Gallery(
                show_label=False,
                columns=3,
                height=170,
                elem_id="pipeline-gallery",
            )
            with gr.Row(elem_id="downloads"):
                glb_out = gr.File(label="Download .glb")
                obj_out = gr.File(label="Download .obj")
            with gr.Accordion("Validation & timing report", open=False):
                report = gr.JSON(show_label=False)

    gr.HTML(
        "<div id='studio-footer'><span>Sketch2Shape Studio — TechZephyr 4.0</span>"
        "<span>Sketch &#8594; ControlNet &#8594; InstantMesh &#8594; GLB/OBJ</span></div>"
    )

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