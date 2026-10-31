# SKETCH2SHAPE AI — Canonical Step-by-Step Execution Blueprint

This document serves as the single source of truth for both human programmers and AI coding agents building the TechZephyr 4.0 Sketch2Shape AI pipeline. Every module must strictly follow the file structure, function contracts, and non-negotiable engineering safeguards defined below.

## 1. The 10 Non-Negotiable Engineering Rules (Do Not Regress)

1. Verified Active Model IDs Only: NEVER use the deleted runwayml/stable-diffusion-v1-5 repository. Always use stable-diffusion-v1-5/stable-diffusion-v1-5 alongside lllyasviel/sd-controlnet-scribble, stabilityai/TripoSR, and TencentARC/InstantMesh.
2. Per-Backend Background Compositing (CRITICAL): In Stage 4 (src/image_to_3d.py), background compositing of the isolated RGBA foreground must be routed via config.py constants per backend: TripoSR → WHITE (255, 255, 255); InstantMesh → GRAY (127, 127, 127). NEVER hardcode a single background color for both 3D backends.
3. Faint-Sketch Fallback (src/preprocess.py): If Otsu thresholding produces an almost empty binary map (foreground stroke pixel ratio < 0.5%) due to faint pencil lines, automatically fall back to Canny edge detection (cv2.Canny(blurred, 30, 100)) + morphological dilation (3x3 kernel) so the user's sketch is never lost.
4. Silent Retry-Once Insurance (pipeline.py): Judges hand you ONE fresh sketch during live evaluation. If Stage 4/5 mesh generation or validation fails on the first attempt, pipeline.py must silently re-run once with seed + 1 (and/or fallback 3D backend) before any error ever reaches the user UI.
5. Strict Vertex & Face Floor (src/mesh_validator.py): Reject any mesh where vertices < 100 OR faces < 100. Never use >= 4, as a 4-vertex mesh can be a flat plane.
6. Strict Anti-Plane Depth Check (src/mesh_validator.py): After normalizing the mesh into a unit bounding box, enforce min_axis_depth >= 0.05 across all three X, Y, Z bounding-box extents. 1e-4 lets a nearly-flat plane pass—judges explicitly test for and disqualify flat planes.
7. Post-Export Reload Verification (src/mesh_validator.py): After exporting .glb (primary) and .obj (compatibility fallback), verify both files are > 1024 bytes and execute trimesh.load(glb_path) to prove the artifact a judge downloads actually opens without corruption.
8. CPU-First Development & VRAM Discipline: Develop UI and pipeline wiring with MOCK_GPU=True on CPU. On GPU, unload the 2D pipeline (gc.collect() + torch.cuda.empty_cache()) before loading any 3D model. In the single live demo session on Kaggle T4 x2, pin Stage 2 to cuda:0 and Stage 4 to cuda:1. Dual T4s do not pool VRAM in solo dev sessions.
9. Gradio Pinned at 4.44.0 Everywhere: Keep gradio==4.44.0 locked across all environments and normalize gr.Sketchpad dictionary outputs ({"background": ..., "layers": [...], "composite": PIL.Image}) via normalize_sketch_input().
10. Mandatory Continuous Audit Log (done_so_far.md / what_we_did.md) & README Disclosures: Every coded feature and every pipeline run, both success and failure, must be logged with IST timestamps (UTC+5:30), role importance, files touched, duration, and technical explanation. The final README.md must disclose all model licenses (SD 1.5: CreativeML OpenRAIL-M, TripoSR: MIT, rembg: MIT), the stage runtime table, hardware specs, and limitations.

## 2. Target Repository Structure

All programmers and AI coding agents must write code strictly into these designated files:

sketch2shape/
├── app.py                      # Entry point: Gradio 4.44.0 UI layout, 3D viewer & event bindings
├── pipeline.py                 # Master orchestrator: per-stage timing, retry-once (seed+1), audit logging
├── config.py                   # Model IDs, per-backend BG colors, device pinning (cuda:0/cuda:1), flags
├── download_weights.py         # Judge 1-click reproducibility script & Kaggle Dataset cache builder
├── done_so_far.md              # Continuous IST-timestamped log of features, roles & pipeline runs
├── what_we_did.md              # Auto-synced role importance, job breakdown & architectural summary
├── requirements.txt            # Pinned dependencies (gradio==4.44.0, diffusers, trimesh, rembg, etc.)
├── README.md                   # Rubric-aligned documentation (15 pts): licenses, runtime table, limits
├── src/
│   ├── __init__.py
│   ├── logger.py               # Dual-logger updating done_so_far.md & what_we_did.md with IST timestamps
│   ├── preprocess.py           # Stage 1: Gradio dict normalization, Otsu + faint-sketch Canny fallback
│   ├── sketch_to_2d.py         # Stage 2 & 3: ControlNet-Scribble + SD1.5 (cuda:0) + rembg 85% crop
│   ├── image_to_3d.py          # Stage 4: Per-backend BG routing + TripoSR / InstantMesh (cuda:1)
│   └── mesh_validator.py       # Stage 5: 8-step validation, anti-plane >=0.05, 1-iter smooth, reload check
├── examples/                   # Pre-tested sketch inputs + verified .glb/.obj outputs
└── outputs/                    # Runtime output directory (auto-created at startup)

## 3. Step-by-Step Implementation Instructions for Coding Agents

### Step 0: Continuous Audit Logger & Role Tracker (src/logger.py → done_so_far.md & what_we_did.md)

Objective: Maintain a real-time, IST-timestamped (Asia/Kolkata, UTC+5:30) engineering audit log that records every coded feature, its role importance, its exact job in the architecture, and every pipeline execution, both successes and failures. This doubles as instant, copy-paste source material for the 15-point README.md rubric requirement.

Function Contracts (src/logger.py):

def log_feature_progress(
    feature_name: str,
    role_owner: str,
    role_importance_and_job: str,
    files_modified: list[str],
    technical_explanation: str,
    duration_or_latency: str = "N/A"
) -> None:
    """Appends a timestamped feature completion entry (in IST) to done_so_far.md and what_we_did.md."""

def log_pipeline_run(
    run_id: str,
    status: str,  # "SUCCESS", "RETRY_SUCCESS", or "FAILURE"
    stage_timings: dict,
    mesh_report: dict,
    notes: str = ""
) -> None:
    """Logs every pipeline execution (success AND failure) to done_so_far.md with IST timestamps."""

1. Compute the exact IST timestamp using datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S IST").
2. Whenever a file is created or updated, append the specified structured Markdown feature block to both done_so_far.md and what_we_did.md.
3. Immediately call log_feature_progress(...) after completing each step below.

Required feature log block:
### [2026-10-31 16:10:00 IST] — Feature: <feature_name>
- Role Owner: <role_owner>
- Role Importance & Job: <why this component is critical to the 100-point rubric and what job it performs>
- Files Touched: <files_modified>
- Time Taken / Latency: <duration_or_latency>
- Technical Explanation: <exact algorithms, fallback logic, and integration contract>
---

### Step 1: Configuration & Reproducible Weight Downloader (config.py & download_weights.py)

- SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5" (Active repo; never use runwayml).
- CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-scribble".
- TRIPOSR_MODEL_ID = "stabilityai/TripoSR".
- INSTANTMESH_MODEL_ID = "TencentARC/InstantMesh".
- BG_COLOR_TRIPOSR = (255, 255, 255), pure WHITE required for TripoSR.
- BG_COLOR_INSTANTMESH = (127, 127, 127), neutral GRAY required for InstantMesh.
- PRIMARY_3D_BACKEND = os.getenv("PRIMARY_3D_BACKEND", "triposr"), decided at the Hour 5–8 benchmark gate ("triposr" or "instantmesh").
- DEVICE_2D = "cuda:0" if torch.cuda.is_available() else "cpu".
- DEVICE_3D = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_2D.
- MOCK_GPU = os.getenv("MOCK_GPU", "0") == "1" or not torch.cuda.is_available().
- FRAME_OCCUPANCY_RATIO = 0.85.
- MIN_VERTICES = 100.
- MIN_FACES = 100.
- MIN_AXIS_DEPTH = 0.05.
- LAPLACIAN_ITERATIONS = 1, default 1 iteration.

download_weights.py must use huggingface_hub.snapshot_download to pre-fetch SD_MODEL_ID, CONTROLNET_MODEL_ID, and TRIPOSR_MODEL_ID into WEIGHTS_CACHE_DIR and initialize rembg (u2net.onnx). Log completion of Step 1 to done_so_far.md and what_we_did.md.

### Step 2: Stage 1 — Sketch Normalization & Faint-Sketch Fallback (src/preprocess.py)

Function Contracts:

def normalize_sketch_input(sketch_input: dict | Image.Image | np.ndarray | None, upload_input: Image.Image | None) -> Image.Image:
def preprocess_sketch(raw_image: Image.Image, target_size: int = 512) -> Image.Image:

1. normalize_sketch_input: If upload_input is present, use it; otherwise inspect sketch_input. Handle the gradio==4.44.0 gr.Sketchpad dictionary using sketch_input.get("composite"). Alpha-composite RGBA images onto a solid white (255, 255, 255) background. Raise ValueError if the canvas is completely blank.
2. preprocess_sketch: Convert to grayscale using cv2.cvtColor and denoise with cv2.bilateralFilter(gray, 9, 75, 75). Apply Otsu binarization using cv2.threshold(..., cv2.THRESH_BINARY + cv2.THRESH_OTSU). Invert if mean pixel intensity > 127 so strokes are white (255) on a black (0) background.
3. Faint-Sketch Fallback: Calculate fg_ratio = np.count_nonzero(binary) / binary.size. If fg_ratio < 0.005, run cv2.adaptiveThreshold + cv2.Canny(gray, 30, 100) and dilate with a 3x3 ones kernel using cv2.dilate.
4. Aspect-ratio-safely pad and resize to 512x512 RGB PIL.Image.
5. Log completion of Step 2 to done_so_far.md and what_we_did.md.

### Step 3: Stage 2 & Stage 3 — 2D Synthesis & 85% Foreground Isolation (src/sketch_to_2d.py)

Function Contract:

def generate_2d_and_isolate_foreground(
    scribble_img: Image.Image,
    user_prompt: str = "",
    control_scale: float = 0.85,
    seed: int = 42
) -> tuple[Image.Image, Image.Image]:

1. Build the prompt: substitute DEFAULT_PROMPT if user_prompt is empty; otherwise combine f"{user_prompt.strip()}, {DEFAULT_PROMPT}".
2. Run StableDiffusionControlNetPipeline using stable-diffusion-v1-5/stable-diffusion-v1-5 and lllyasviel/sd-controlnet-scribble in torch.float16 on DEVICE_2D (cuda:0), for 20 steps with UniPCMultistepScheduler and torch.Generator(DEVICE_2D).manual_seed(seed).
3. If running in a single-GPU environment (DEVICE_2D == DEVICE_3D), immediately unload/offload the 2D pipeline and call gc.collect() + torch.cuda.empty_cache() before any 3D model loads.
4. Run rembg.remove(shaded_2d_img) to obtain a raw RGBA image.
5. Find the bounding box of non-zero alpha pixels (alpha > 10), crop tightly around the foreground object, and recenter it onto a square transparent RGBA canvas (512x512) such that the object's longest dimension occupies 85% (FRAME_OCCUPANCY_RATIO) of the frame.
6. Return (shaded_2d_img, recentered_rgba_img).
7. Log completion of Step 3 to done_so_far.md and what_we_did.md.

### Step 4: Stage 4 — 3D Reconstruction & Per-Backend Compositing (src/image_to_3d.py)

Function Contracts:

def composite_for_backend(rgba_img: Image.Image, backend: str) -> Image.Image:
def reconstruct_3d_mesh(
    rgba_img: Image.Image,
    backend: str = PRIMARY_3D_BACKEND,
    mc_resolution: int = 256,
    seed: int = 42
) -> trimesh.Trimesh:

1. Inside composite_for_backend(rgba_img, backend), if backend == "triposr", composite the transparent RGBA image onto config.BG_COLOR_TRIPOSR, WHITE (255, 255, 255). If backend == "instantmesh", composite onto config.BG_COLOR_INSTANTMESH, GRAY (127, 127, 127).
2. At Hour 5–8, run the scheduled benchmark gate across 3 test sketches comparing TripoSR (fast 5–15s on T4, vertex colors, Marching Cubes res=256) versus InstantMesh (textured .glb, fp16 + CPU offload fits 15GB T4). The winner is set as PRIMARY_3D_BACKEND; the runner-up remains wired as the automatic fallback.
3. Pin 3D inference to DEVICE_3D (cuda:1 only inside the single dual-T4 demo session; cuda:0 in solo single-GPU sessions).
4. If MOCK_GPU=True, return a vertex-colored trimesh.creation.icosphere(subdivisions=3, radius=0.8) for CPU UI testing.
5. Log completion of Step 4 to done_so_far.md and what_we_did.md.

### Step 5: Stage 5 — 8-Step Mesh Validation & Export (src/mesh_validator.py)

Function Contract:

def validate_and_export_mesh(
    mesh: trimesh.Trimesh,
    output_dir: str,
    run_id: str,
    smoothing_iterations: int = config.LAPLACIAN_ITERATIONS
) -> dict:

1. Strict Vertex/Face Floor Reject: Check len(mesh.vertices) < 100 or len(mesh.faces) < 100. If true, raise ValueError("Mesh failed minimum geometry threshold (<100 verts/faces).").
2. Topology Purge: Remove degenerate and duplicate faces, remove unreferenced vertices, and merge close vertices.
3. Centroid & Unit Normalization: Shift mesh.vertices -= mesh.centroid to center at (0, 0, 0), then scale uniformly so the longest bounding-box axis fits inside a unit bounding box [-0.5, 0.5] or [-1.0, 1.0].
4. Anti-Plane Depth Check: Compute extents = mesh.extents after normalization and check min_axis_depth = float(np.min(extents) / np.max(extents)). If min_axis_depth < 0.05, raise ValueError(f"Flat plane detected: min_axis_depth={min_axis_depth:.4f} < 0.05").
5. Normals & Controlled Laplacian Smoothing: Fix face windings/normals using trimesh.repair.fix_normals(mesh). Apply Laplacian smoothing with 1 iteration by default using trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations).
6. Dual Export: Export outputs/{run_id}.glb, preserving vertex colors/textures, and outputs/{run_id}.obj. Verify both files exist and have os.path.getsize() > 1024 bytes.
7. Post-Export Reload Verification: Execute reloaded = trimesh.load(glb_path, force="mesh") and verify len(reloaded.vertices) >= 100.
8. Return a structured validation report containing glb_path, obj_path, vertex_count, face_count, is_watertight, bbox_extents, min_axis_depth, glb_size_kb, obj_size_kb, and reload_verified: True.
9. Log completion of Step 5 to done_so_far.md and what_we_did.md.

### Step 6: Master Orchestrator (pipeline.py), UI (app.py), & Rubric Docs (README.md)

- pipeline.py: Record high-precision time.perf_counter() durations for stage1_preprocess_s, stage2_controlnet_s, stage3_rembg_s, stage4_3d_recon_s, stage5_validate_export_s, and total_runtime_s. Surface the full breakdown in the UI JSON box.
- pipeline.py: Wrap Stages 2–5 in a try/except block. If Stage 4 or Stage 5 raises a validation, anti-plane, or reload exception on Attempt 1, silently trigger Attempt 2 with seed = seed + 1 and/or fallback 3D backend before surfacing any error to the user.
- pipeline.py: Call log_pipeline_run(...) at the end of every run, both success and failure, so done_so_far.md has a complete IST-timestamped record.
- app.py: Pin gradio==4.44.0. Provide Draw tab (gr.Sketchpad), Upload tab (gr.Image), optional prompt (gr.Textbox), Intermediate 2D Gallery (Conditioned Scribble, Shaded 2D Render, 85% Isolated RGBA), Interactive 3D Viewer (gr.Model3D), .glb and .obj download buttons, and Validation & Timing JSON report.
- README.md must include architecture overview and diagram; model license disclosures; realistic T4 runtime table; hardware assumptions and known limitations; and the role summary from what_we_did.md and done_so_far.md.

## 4. What what_we_did.md & done_so_far.md Track (Role Importance & Job Matrix)

| Module / File | Role Owner | Role Importance (Rubric Mapping) | Exact Job Performed in Pipeline |
| --- | --- | --- | --- |
| src/logger.py, done_so_far.md, what_we_did.md | Role C (Docs & Audit) | Secures Documentation (15 pts) & Reproducibility (15 pts) | Logs every coded feature, role job, and live pipeline run (success & failure) with IST timestamps. |
| config.py, download_weights.py | Role C (Infra Lead) | Prevents DNF from dead repos & Wi-Fi stalls (Reproducibility 15 pts) | Locks stable-diffusion-v1-5/stable-diffusion-v1-5, per-backend BG colors (WHITE vs GRAY), and 1-click weight caching. |
| src/preprocess.py | Role A (2D Vision Lead) | Protects Technical Implementation (25 pts) against bad inputs | Normalizes Gradio 4.44.0 dicts, denoises sketches, and triggers Canny + dilation fallback when Otsu loses faint pencil lines. |
| src/sketch_to_2d.py | Role A (2D Vision Lead) | Drives 3D Output Quality (25 pts) via shaded depth cues | Converts flat 2D strokes into shaded 3D-like renders via ControlNet-Scribble (cuda:0), unloads VRAM, and isolates foreground at 85% occupancy via rembg. |
| src/image_to_3d.py | Role B (3D Lead) | Core engine for 3D Output Quality (25 pts) & Efficiency (20 pts) | Routes per-backend BG colors (WHITE for TripoSR, GRAY for InstantMesh) and extracts 3D mesh on cuda:1 / cuda:0. |
| src/mesh_validator.py | Role B (3D Lead) | Guarantees Pipeline Validity (20 pts) & judge file opening | Enforces ≥100 verts/faces, min_axis_depth ≥0.05 anti-plane check, 1-iter Laplacian smooth, .glb/.obj export, and trimesh.load() reload verification. |
| pipeline.py, app.py | Role C (UI & Integration) | Delivers End-to-End Prototype (20 pts) with zero live-demo crashes | Wraps stages with per-stage timing, silent seed + 1 retry-once insurance on validation failure, and Gradio 4.44.0 3D UI. |

## 5. End-to-End Canonical System Architecture Diagram

USER INTERFACE LAYER: Gradio 4.44.0 Web App (app.py), with an in-app sketchpad, upload sketch image tab, optional text prompt, 2D gallery, interactive 3D viewer, download buttons, and validation/timing JSON report.

STAGE 1: INPUT NORMALIZATION & FAINT-SKETCH PREPROCESSING (src/preprocess.py — CPU): normalize_sketch_input → RGBA to white background → bilateral denoise → Otsu binarization. If foreground stroke ratio is below 0.5%, use adaptive threshold + Canny(30,100) + 3x3 dilation. Then aspect-safe pad and resize to 512x512.

STAGE 2: SKETCH-GUIDED 2D SYNTHESIS (src/sketch_to_2d.py — cuda:0): stable-diffusion-v1-5/stable-diffusion-v1-5 + lllyasviel/sd-controlnet-scribble, FP16, 20 steps, UniPCMultistepScheduler, Seed=N, approximately 4–6 seconds on T4. Unload 2D pipeline and clear VRAM before 3D loads.

STAGE 3: FOREGROUND ISOLATION & 85% RECENTERING: rembg.remove() → RGBA → tight alpha bounding-box crop → recenter at 85% frame occupancy on an RGBA canvas.

STAGE 4: PER-BACKEND COMPOSITING & 3D RECONSTRUCTION: TripoSR composites onto pure white (255,255,255); InstantMesh composites onto neutral gray (127,127,127). TripoSR uses Marching Cubes at resolution 256 and produces a vertex-colored Trimesh. InstantMesh produces textured GLB output with FP16 and CPU offload. MOCK_GPU mode uses a procedural vertex-colored icosphere.

STAGE 5: 8-STEP MESH VALIDATION, SMOOTHING & VERIFIED EXPORT: floor check, topology purge, centroid/unit normalization, anti-plane check with min_axis_depth >=0.05, normal repair and one-iteration Laplacian smoothing, dual GLB/OBJ export with file-size checks, post-export trimesh.load() verification, and structured output report.

VALIDATION PASSED: Interactive 3D viewer, verified GLB and OBJ download buttons, and per-stage and total runtime JSON report.

VALIDATION FAILED ON ATTEMPT 1: Silent retry-once insurance re-runs with seed = seed + 1 and/or switches to the fallback 3D backend before exposing an error to the judge UI.

CONTINUOUS IST AUDIT LOG & ROLE TRACKER: src/logger.py writes IST timestamps, role owner, role importance and job, files touched, stage timings, and mesh metrics to done_so_far.md and what_we_did.md. These records feed README.md with model licenses, runtime table, hardware, and limitations.

## README.md Required Disclosures

- Architecture overview and diagram.
- Model License Disclosures: Stable Diffusion 1.5 (CreativeML OpenRAIL-M), ControlNet-Scribble (Apache-2.0 / OpenRAIL-M), TripoSR (MIT License), InstantMesh (Apache-2.0), rembg (MIT License), trimesh (MIT License).
- Realistic T4 Runtime Table: Preprocess ~0.2s; ControlNet ~4–6s; rembg ~1.0s; TripoSR ~5–15s; Validation/Export ~0.5s; Total ~12–22s.
- Hardware Assumptions & Known Limitations: single-view backside occlusion hallucination and thin-wireframe quantization at 256^3 marching cubes.
- Role Summary from what_we_did.md and done_so_far.md.