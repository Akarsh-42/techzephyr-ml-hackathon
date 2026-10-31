# Sketch2Shape AI — Role Importance, Jobs & Architecture (what_we_did.md)

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
### [2026-10-31 18:48:53 IST] — Feature: Step 6 — Master Orchestrator (pipeline.py)
- Role Owner: Role C (UI & Integration)
- Role Importance & Job: Delivers End-to-End Pipeline Validity (20 pts) with zero live-demo crashes: per-stage timing, silent retry-once (seed+1) insurance, and audit logging of every run, success and failure.
- Files Touched: pipeline.py, src/sketch_to_2d.py
- Time Taken / Latency: 1.34
- Technical Explanation: Stage 1 runs once (deterministic); Stages 2-5 are wrapped in a two-attempt loop: attempt 2 silently re-runs with seed+1 on any Stage 2-5 exception (Rule 4) before the error ever reaches the UI. Stage 2/3 split timings come from sketch_to_2d.LAST_TIMINGS; stages 4-5 use time.perf_counter(). Every run calls log_pipeline_run with status SUCCESS/RETRY_SUCCESS/FAILURE.
---
### [2026-10-31 19:40:07 IST] — Feature: Fix Gradio Sketchpad NumPy Input Normalization
- Role Owner: Role C (UI & Integration)
- Role Importance & Job: Keeps the Generate action usable with Gradio 4.44.0 Sketchpad dictionary outputs, preventing an input-normalization exception before the image pipeline starts.
- Files Touched: src/preprocess.py, what_we_did.md, done_so_far.md
- Time Taken / Latency: N/A (diagnosis and correction during this session)
- Technical Explanation: Gradio can supply the Sketchpad composite as a NumPy array. The old `sketch_input.get("composite") or sketch_input.get("background")` expression implicitly asked NumPy for the array's truth value and raised `The truth value of an array with more than one element is ambiguous`. Replaced it with an explicit `is None` fallback. Audited the venv's Gradio 4.44.0 installation: 1,287 recorded file hashes matched, and `pip check` reported no broken dependencies. No Gradio package files were changed. The application was not rerun after this correction.
---
### [2026-10-31 19:46:36 IST] — Feature: Recover Sketchpad Strokes from Gradio Layers
- Role Owner: Role C (UI & Integration)
- Role Importance & Job: Prevents valid in-app drawings from being rejected as blank and keeps Stage 1 compatible with Gradio 4.44.0 ImageEditor/Sketchpad payloads.
- Files Touched: src/preprocess.py, what_we_did.md, done_so_far.md
- Time Taken / Latency: N/A (follow-up diagnosis from UI screenshot)
- Technical Explanation: The UI showed `Sketch canvas is completely blank` despite visible strokes. Gradio 4.44.0's Sketchpad payload can carry background, layer images, and an optional composite. The old normalizer ignored `layers` and could select an empty background if the composite was missing or blank. NumPy RGBA inputs were also being truncated to RGB, losing alpha. Updated `_to_pil` to preserve RGBA, added alpha-aware layer compositing over a white canvas, and fall back to that reconstructed image when the composite is missing or visually blank. No Gradio library files were changed. App rerun was not performed.
---
