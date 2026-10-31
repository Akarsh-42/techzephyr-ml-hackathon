# Sketch2Shape AI — Continuous Audit Log (done_so_far.md)

### [2026-10-31 17:46:58 IST] — Feature: Step 2 — Sketch Normalization & Faint-Sketch Fallback
- Role Owner: Role A (2D Vision Lead)
- Role Importance & Job: Protects Technical Implementation (25 pts) against bad inputs: normalizes the Gradio 4.44.0 Sketchpad dict, denoises, binarizes via Otsu, and never loses faint pencil strokes thanks to the adaptive + Canny(30,100) + 3x3 dilation fallback.
- Files Touched: src/preprocess.py
- Time Taken / Latency: ~0.2s per sketch on CPU
- Technical Explanation: normalize_sketch_input prefers the upload tab, falls back through Sketchpad dict keys (composite/background), alpha-composites RGBA onto white, rejects blank canvases (pixel std < 3). preprocess_sketch: bilateralFilter(9,75,75), Otsu with auto-inversion to white strokes on black, fallback when fg ratio < 0.5%, aspect-safe square padding to 512x512 RGB.
---
### [2026-10-31 18:12:53 IST] — Feature: Step 1 — Configuration & Reproducible Weight Downloader
- Role Owner: Role C (Infra Lead)
- Role Importance & Job: Prevents DNF from dead repos & Wi-Fi stalls (Reproducibility 15 pts): locks active model IDs, per-backend BG colors, and provides 1-click weight caching.
- Files Touched: config.py, download_weights.py, requirements.txt
- Time Taken / Latency: 1196.5s
- Technical Explanation: Pinned stable-diffusion-v1-5/stable-diffusion-v1-5 (active repo; runwayml variant is deleted), lllyasviel/sd-controlnet-scribble, and stabilityai/TripoSR, fetched via huggingface_hub.snapshot_download into C:\iit_techzyphr\techzephyr-ml-hackathon\weights_cache; rembg u2net.onnx warmed with a dummy 64x64 pass. Device pinning: Stage 2 on cpu, Stage 4 on cpu; MOCK_GPU=True.
---
### [2026-10-31 18:48:53 IST] — Pipeline Run: run_20261031_184852
- Status: SUCCESS
- Stage Timings (s): {
  "stage1_preprocess_s": 0.03,
  "stage2_controlnet_s": 0.01,
  "stage3_rembg_s": 1.26,
  "stage4_3d_recon_s": 0.01,
  "stage5_validate_export_s": 0.04,
  "total_runtime_s": 1.34
}
- Mesh Report: {
  "bbox_extents": [
    1.0,
    1.0,
    1.0
  ],
  "face_count": 1280,
  "glb_path": "C:\\iit_techzyphr\\techzephyr-ml-hackathon\\outputs\\run_20261031_184852.glb",
  "glb_size_kb": 26.0,
  "is_watertight": true,
  "min_axis_depth": 1.0,
  "obj_path": "C:\\iit_techzyphr\\techzephyr-ml-hackathon\\outputs\\run_20261031_184852.obj",
  "obj_size_kb": 60.5,
  "reload_verified": true,
  "vertex_count": 642
}
- Notes: None
---
### [2026-10-31 18:48:53 IST] — Feature: Step 6 — Master Orchestrator (pipeline.py)
- Role Owner: Role C (UI & Integration)
- Role Importance & Job: Delivers End-to-End Pipeline Validity (20 pts) with zero live-demo crashes: per-stage timing, silent retry-once (seed+1) insurance, and audit logging of every run, success and failure.
- Files Touched: pipeline.py, src/sketch_to_2d.py
- Time Taken / Latency: 1.34
- Technical Explanation: Stage 1 runs once (deterministic); Stages 2-5 are wrapped in a two-attempt loop: attempt 2 silently re-runs with seed+1 on any Stage 2-5 exception (Rule 4) before the error ever reaches the UI. Stage 2/3 split timings come from sketch_to_2d.LAST_TIMINGS; stages 4-5 use time.perf_counter(). Every run calls log_pipeline_run with status SUCCESS/RETRY_SUCCESS/FAILURE.
---
### [2026-10-31 19:40:07 IST] — Feature: Fix Gradio Sketchpad NumPy Input Normalization
- Role Owner: Role C (UI & Integration)
- Role Importance & Job: Prevents the Generate action from failing at Stage 1 when Gradio 4.44.0 returns a Sketchpad composite as a NumPy array; supports end-to-end pipeline validity (20 pts).
- Files Touched: src/preprocess.py, what_we_did.md, done_so_far.md
- Time Taken / Latency: N/A (diagnosis and correction during this session)
- Technical Explanation: Replaced boolean fallback on `sketch_input.get("composite")` with an explicit `is None` check before using `background`. NumPy arrays do not define a single truth value, which caused the reported UI error. Checked the installed Gradio 4.44.0 package against its RECORD: 1,287 hashes matched with no mismatches; `pip check` found no broken requirements. No Gradio package files were changed. Application rerun was not performed.
---
### [2026-10-31 19:46:36 IST] — Feature: Recover Sketchpad Strokes from Gradio Layers
- Role Owner: Role C (UI & Integration)
- Role Importance & Job: Prevents Stage 1 from rejecting a visibly drawn Sketchpad input as blank; improves end-to-end pipeline validity and UI reliability.
- Files Touched: src/preprocess.py, what_we_did.md, done_so_far.md
- Time Taken / Latency: N/A (follow-up diagnosis from UI screenshot)
- Technical Explanation: Gradio 4.44.0 Sketchpad uses an ImageEditor payload with optional `composite` plus `background` and `layers`. Updated normalization to preserve four-channel NumPy RGBA data and to alpha-composite available layers over white when the supplied composite is missing or blank. This addresses the screenshot's blank-canvas rejection while retaining blank-input detection when no visible strokes exist. No Gradio package files were changed. App rerun was not performed.
---
### [2026-10-31 19:48:01 IST] — Pipeline Run: run_20261031_194755
- Status: SUCCESS
- Stage Timings (s): {
  "stage1_preprocess_s": 0.2,
  "stage2_controlnet_s": 0.01,
  "stage3_rembg_s": 4.88,
  "stage4_3d_recon_s": 0.01,
  "stage5_validate_export_s": 0.09,
  "total_runtime_s": 5.2
}
- Mesh Report: {
  "bbox_extents": [
    1.0,
    1.0,
    1.0
  ],
  "face_count": 1280,
  "glb_path": "C:\\iit_techzyphr\\techzephyr-ml-hackathon\\outputs\\run_20261031_194755.glb",
  "glb_size_kb": 26.0,
  "is_watertight": true,
  "min_axis_depth": 1.0,
  "obj_path": "C:\\iit_techzyphr\\techzephyr-ml-hackathon\\outputs\\run_20261031_194755.obj",
  "obj_size_kb": 60.5,
  "reload_verified": true,
  "vertex_count": 642
}
- Notes: None
---
### [2026-10-31 19:49:16 IST] — Pipeline Run: run_20261031_194912
- Status: SUCCESS
- Stage Timings (s): {
  "stage1_preprocess_s": 0.16,
  "stage2_controlnet_s": 0.01,
  "stage3_rembg_s": 4.2,
  "stage4_3d_recon_s": 0.01,
  "stage5_validate_export_s": 0.08,
  "total_runtime_s": 4.46
}
- Mesh Report: {
  "bbox_extents": [
    1.0,
    1.0,
    1.0
  ],
  "face_count": 1280,
  "glb_path": "C:\\iit_techzyphr\\techzephyr-ml-hackathon\\outputs\\run_20261031_194912.glb",
  "glb_size_kb": 26.0,
  "is_watertight": true,
  "min_axis_depth": 1.0,
  "obj_path": "C:\\iit_techzyphr\\techzephyr-ml-hackathon\\outputs\\run_20261031_194912.obj",
  "obj_size_kb": 60.5,
  "reload_verified": true,
  "vertex_count": 642
}
- Notes: None
---
