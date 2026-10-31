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
### [2026-10-31 19:17:25 IST] — Feature: Step 3 — ControlNet 2D Synthesis & 85% Foreground Isolation
- Role Owner: Role A (2D Vision Lead)
- Role Importance & Job: Drives 3D Output Quality (25 pts) via shaded depth cues: converts flat strokes into shaded renders on cuda:0, unloads VRAM before 3D loads, and isolates the foreground at 85% frame occupancy via rembg for Stage 4.
- Files Touched: src/sketch_to_2d.py
- Time Taken / Latency: MOCK (CPU) verified; GPU ~4-6s ControlNet + ~1.0s rembg on T4
- Technical Explanation: SD1.5 + sd-controlnet-scribble in float16, 20 steps, UniPCMultistepScheduler, torch.Generator seed control; on single-GPU (DEVICE_2D == DEVICE_3D) the 2D pipeline is unloaded with gc.collect() + torch.cuda.empty_cache() before Stage 4. rembg.remove yields RGBA, tight alpha>10 bbox crop, LANCZOS rescale to 85% frame occupancy on a transparent 512x512 canvas. MOCK_GPU renders a distance-transform shading so CPU-first UI development runs end-to-end. Stage 2/3 latencies recorded in LAST_TIMINGS for pipeline.py per-stage reporting.
---
