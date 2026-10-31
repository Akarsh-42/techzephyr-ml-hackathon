# Sketch2Shape AI — Continuous Audit Log (done_so_far.md)

### [2026-10-31 18:46:35 IST] — Feature: Step 0 — Continuous Audit Logger & Role Tracker
- Role Owner: Role C (Docs & Audit)
- Role Importance & Job: Secures Documentation (15 pts) & Reproducibility (15 pts): records every coded feature, role job, and pipeline run (success and failure) with IST timestamps.
- Files Touched: src/logger.py, src/__init__.py
- Time Taken / Latency: N/A
- Technical Explanation: Dual appender writing structured Markdown blocks to done_so_far.md and what_we_did.md; IST computed via zoneinfo.ZoneInfo('Asia/Kolkata'), the standard-library tz database replacement for the manual UTC+5:30 timedelta.
---
### [2026-10-31 18:46:36 IST] — Feature: Step 2 — Sketch Normalization & Faint-Sketch Fallback
- Role Owner: Role A (2D Vision Lead)
- Role Importance & Job: Protects Technical Implementation (25 pts) against bad inputs: normalizes the Gradio 4.44.0 Sketchpad dict, denoises, binarizes via Otsu, and never loses faint pencil strokes thanks to the adaptive + Canny(30,100) + 3x3 dilation fallback.
- Files Touched: src/preprocess.py
- Time Taken / Latency: ~0.2s per sketch on CPU
- Technical Explanation: normalize_sketch_input prefers the upload tab, falls back through Sketchpad dict keys (composite/background), alpha-composites RGBA onto white, rejects blank canvases (pixel std < 3). preprocess_sketch: bilateralFilter(9,75,75), Otsu with auto-inversion to white strokes on black, fallback when fg ratio < 0.5%, aspect-safe square padding to 512x512 RGB.
---
### [2026-10-31 18:46:48 IST] — Feature: Step 4 — 3D Reconstruction & Per-Backend Compositing
- Role Owner: Role B (3D Lead)
- Role Importance & Job: Core engine for 3D Output Quality (25 pts) & Efficiency (20 pts): routes per-backend BG colors (WHITE for TripoSR, GRAY for InstantMesh) and extracts the 3D mesh on the pinned DEVICE_3D.
- Files Touched: src/image_to_3d.py, requirements.txt, download_weights.py
- Time Taken / Latency: MOCK (CPU) verified; TripoSR ~5-15s on T4
- Technical Explanation: composite_for_backend reads BG colors exclusively from config (Rule 2, never hardcoded per call). TripoSR path: tsr.system.TSR loaded once as a lazy singleton from the snapshot_download cache (config.yaml + model.ckpt), chunked rendering (8192) to bound VRAM, Marching Cubes at mc_resolution=256. TSR.extract_mesh returns a list of trimesh.Trimesh; we take the first mesh, then bake vertex colors by sampling the composited image via normalized bbox projection. seed is accepted for the pipeline retry-once contract; TSR inference itself is deterministic. MOCK_GPU returns a vertex-colored icosphere(subdivisions=3, radius=0.8) for CPU-first UI development. InstantMesh raises a clear ValueError until the benchmark gate picks the primary backend. TripoSR's GitHub repo is NOT pip-installable (no setup.py), so third_party/TripoSR (cloned by download_weights.py) is inserted into sys.path before the tsr import.
---
