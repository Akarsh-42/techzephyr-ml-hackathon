# SKETCH2SHAPE AI — Canonical Architecture Plan (v2)

**Supersedes:** execution_blueprint_architecture (1).md, and planning.md v1. This + setup.md are the ONLY two planning documents (this = what to build; setup.md = where/how to run). All other docs move to `archive/`.

## Stage numbering (use everywhere: code comments, README, demo narration)

Stage 1 Preprocess → Stage 2 2D Synthesis → Stage 3 Isolation → Stage 4 3D Reconstruction → Stage 5 Validation & Export. UI and logger wrap all five.

## UI Layer (`app.py` — Gradio 4.44.0, LOCKED)

- Tab 1: `gr.Sketchpad` (draw) | Tab 2: `gr.Image` upload — both routed through `normalize_sketch_input()`
- Optional prompt box (blank → DEFAULT_PROMPT; user text → appended with studio/isometric modifiers)
- Advanced: ControlNet scale (0.1–1.0), diffusion steps (20), MC resolution (256), seed (42)
- Buttons: **Generate** + **Regenerate (seed+1)**
- Outputs: preprocess/render/segmented previews, `gr.Model3D` viewport, .glb + .obj downloads, JSON report (mesh health + per-stage latency)

## Stage 1 — Preprocessing (`src/preprocess.py`)

1. `normalize_sketch_input()`: unpack Sketchpad dict (`composite`/`background`), PIL, or ndarray → RGB PIL.
2. **Blank-canvas rejection:** std of grayscale < 2.0 → friendly error, no GPU burn.
3. Grayscale → bilateral filter (denoise, keeps strokes) → **Otsu binarization**.
4. **Faint-sketch fallback (MANDATORY):** if stroke coverage < 2% of pixels after binarization, skip binarization and use normalized inverted grayscale — Otsu destroys faint pencil.
5. Morphological opening then dilation (remove specks, close stroke gaps).
6. Auto-invert so strokes are WHITE on BLACK (ControlNet-Scribble convention); aspect-ratio pad → 512×512.

## Stage 2 — 2D Synthesis (`src/sketch_to_2d.py`)

- ControlNet-Scribble (`lllyasviel/sd-controlnet-scribble`) + SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5` — NEVER `runwayml/...`, dead repo). fp16, 20 steps, UniPC scheduler, `cpu_offload` on <16GB GPUs.
- Prompt fallback engine per rules (prompt is optional).
- **VRAM rule:** unload 2D pipeline → `gc.collect()` + `torch.cuda.empty_cache()` before any 3D model loads.

## Stage 3 — Foreground Isolation (`rembg` / u2net.onnx)

- `rembg.remove()` → RGBA → tight alpha-bbox crop → recenter at **85% frame occupancy** on transparent RGBA.

## Stage 4 — 3D Reconstruction (`src/image_to_3d.py`)

**Background compositing is PER-BACKEND (critical):** TripoSR → WHITE (255,255,255). InstantMesh → GRAY (127,127,127). Route via config constants; NEVER hardcode one color for both.

- **Primary: decided at the Hour 5–8 benchmark gate** — 3 sketches × {TripoSR (fast, vertex colors) vs InstantMesh (textured .glb, fp16+offload fits 15GB T4)}. Winner becomes primary, loser stays as automatic fallback. This is a SCHEDULED decision, not "if time permits" — it settles the 25-point Output Quality row.
- TripoSR (`stabilityai/TripoSR`) → Marching Cubes (res 256) → vertex-colored trimesh. Realistic budget on T4: 5–15s (not "~2s").
- Device pinning: Stage 2 → cuda:0, Stage 4 → cuda:1 **only inside the single demo session** (dual T4s do NOT pool VRAM; in solo dev sessions cuda:1 is just that session's 2nd GPU).
- Mock mode (`S2S_MOCK=1`): vertex-colored icosphere, CPU-only UI development.

## Stage 5 — Validation & Export (`src/mesh_validator.py`)

1. **Reject:** verts < 100 OR faces < 100 (NOT ≥4 — a 4-vert mesh can be a flat plane).
2. Purge degenerate/duplicate faces + unreferenced vertices; merge vertices.
3. Centroid → origin; scale → unit bounding box.
4. **Anti-plane check: `min_axis_depth >= 0.05`** after normalization (1e-4 would let a nearly-flat plane pass — judges explicitly test for flat planes).
5. Fix face windings/normals; Laplacian smoothing — **1 iteration default, test 1 vs 2 at benchmark gate** (2 over-shrinks detail).
6. Export .glb (PRIMARY — preserves textures) + .obj (compat fallback); size check > 1KB.
7. **Post-export reload verification:** `trimesh.load(glb_path)` must succeed — proves the artifact a judge downloads actually opens.
8. Return report: vertex/face counts, watertight, bbox extents, file sizes.

## Pipeline Orchestration (`pipeline.py`)

- Per-stage timing; total runtime surfaced in UI JSON.
- **Retry-once insurance:** validation failure → silent 3D re-run with seed+1 before any error reaches the user (judges hand you ONE fresh sketch).
- Every run (success AND failure) logged to `done_so_far.md` with timestamps.

## Continuous Audit Log (`src/logger.py` → `done_so_far.md`)

Timestamp, author/agent, files touched, duration, technical explanation. Doubles as README source material (Docs 15 pts).

## Non-negotiables (from all reviews — do not regress)

1. Model IDs: `stable-diffusion-v1-5/stable-diffusion-v1-5` only. 2. Per-backend bg colors. 3. Faint-sketch fallback. 4. Retry-once. 5. ≥100 verts/faces. 6. Anti-plane depth ≥ 0.05. 7. Reload-verify .glb. 8. CPU-first dev, GPU only in benchmark blocks. 9. Gradio pinned 4.44.0 everywhere. 10. README must disclose model licenses (SD 1.5 OpenRAIL-M, TripoSR MIT, rembg MIT) + runtime table + hardware + limitations.