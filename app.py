"""SKETCH2SHAPE AI — Gradio web app. Canvas OR upload, optional prompt, 3D preview, downloads.
Run:  S2S_MOCK=1 python app.py   (mock mode, no GPU)  |  python app.py  (full pipeline)
"""
import gradio as gr
from config import DEFAULT_CONTROL_SCALE, DEFAULT_SEED, MC_RESOLUTION
from pipeline import run_pipeline


def generate(sketch_tab, upload_tab, prompt, control_scale, seed):
    sketch = sketch_tab if sketch_tab is not None else upload_tab
    if sketch is None:
        raise gr.Error("Please draw a sketch or upload an image first.")
    out = run_pipeline(sketch, prompt, control_scale, int(seed))
    if out.get("error"):
        raise gr.Error(f"Pipeline failed: {out['error']}")
    rep = out["report"]
    runtime = {"total_seconds": out["total_s"], **out["timings"]}
    return (out["preprocessed"], out["render"], out["segmented"],
            rep["glb_path"], rep["glb_path"], rep["obj_path"],
            {**rep, **runtime})


def regenerate(sketch_tab, upload_tab, prompt, control_scale, seed):
    """Regenerate = same input, seed+1. Cheap product-feel differentiator."""
    return generate(sketch_tab, upload_tab, prompt, control_scale, int(seed) + 1)


with gr.Blocks(theme=gr.themes.Soft(), title="Sketch2Shape AI") as demo:
    gr.Markdown("# Sketch2Shape AI — 2D Sketch to Textured 3D Asset")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Draw Sketch"):
                    sketch_in = gr.Sketchpad(label="Draw your concept", brush=True, eraser=True)
                with gr.Tab("Upload Image"):
                    upload_in = gr.Image(label="Upload sketch", type="pil")
            prompt_in = gr.Textbox(label="Optional prompt (material, color, style)",
                                   placeholder="e.g. ergonomic wooden chair, matte oak finish — or leave blank")
            with gr.Accordion("Advanced", open=False):
                scale_in = gr.Slider(0.1, 1.0, DEFAULT_CONTROL_SCALE, label="ControlNet conditioning")
                seed_in = gr.Number(DEFAULT_SEED, label="Seed", precision=0)
            go_btn = gr.Button("Generate 3D Asset", variant="primary")
            regen_btn = gr.Button("Regenerate (new seed)")
        with gr.Column():
            with gr.Row():
                pre_view = gr.Image(label="Preprocessed sketch")
                render_view = gr.Image(label="ControlNet render")
                seg_view = gr.Image(label="Isolated foreground")
            viewer = gr.Model3D(label="Interactive 3D preview (.glb)")
            with gr.Row():
                glb_dl = gr.File(label="Download .glb")
                obj_dl = gr.File(label="Download .obj")
            report_view = gr.JSON(label="Mesh validation & runtime report")

    go_btn.click(generate, [sketch_in, upload_in, prompt_in, scale_in, seed_in],
                 [pre_view, render_view, seg_view, viewer, glb_dl, obj_dl, report_view])
    regen_btn.click(regenerate, [sketch_in, upload_in, prompt_in, scale_in, seed_in],
                    [pre_view, render_view, seg_view, viewer, glb_dl, obj_dl, report_view])

if __name__ == "__main__":
    demo.launch()