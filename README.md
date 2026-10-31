# Sketch2Shape AI — TechZephyr 4.0 ML Hackathon

Turn a single-view 2D sketch into a textured, downloadable 3D asset (.glb / .obj).

- **Problem statement:** see `ML_Hackathonv2.txt`
- **Canonical execution blueprint:** see `execution_blueprint_architecture.md`

## Objective

Anyone with a rough pencil sketch can produce a usable 3D model — draw or upload a sketch, optionally describe the object with a text prompt, and receive an interactive 3D preview plus verified `.glb`/`.obj` downloads. The pipeline preserves the sketch's structure (via ControlNet conditioning) while adding plausible depth, shading, and color. 3D reconstruction uses InstantMesh's multi-view diffusion + Large Reconstruction Model for high-quality geometry.

## Architecture

```
Sketch (draw / upload) + optional text prompt
        │
        ▼
[Stage 1] src/preprocess.py      Gradio dict normalization → bilateral denoise → Otsu binarization
        │                        faint-sketch fallback: adaptive threshold + Canny(30,100) + 3x3 dilate
        ▼
[Stage 2] src/sketch_to_2d.py    SD 1.5 + ControlNet-Scribble, FP16, 20 steps, UniPC scheduler (cuda:0)
        │                        unload 2D pipeline + torch.cuda.empty_cache() before 3D load
        ▼
[Stage 3] src/sketch_to_2d.py    rembg background removal → tight alpha crop → 85% frame occupancy
        ▼
[Stage 4] src/image_to_3d.py     InstantMesh: Zero123++ multi-view diffusion (6 views, 75 steps)
        │                        → LRM FlexiCubes reconstruction (grid_res=128) on cuda:1/cuda:0
        │                        Zero123++ unloaded between diffusion and reconstruction to fit VRAM
        ▼
[Stage 5] src/mesh_validator.py  8-step validation: floor reject → topology purge → unit normalize
        │                        → anti-plane (min_axis_depth ≥ 0.05) → Laplacian smooth
        │                        → dual GLB/OBJ export → trimesh.load() reload verification
        ▼
app.py (Gradio 4.44.0)           3D viewer, .glb/.obj downloads, per-stage timing JSON report
```

**Retry insurance:** if any of Stages 2–5 fails, `pipeline.py` silently re-runs once with seed + 1 before any error reaches the UI.

## Quickstart

```bash
pip install -r requirements.txt     # torch first; nvdiffrast installed by the next step
python download_weights.py          # one-time: prefetch SD1.5/ControlNet/Zero123++/InstantMesh
                                    # weights, clone InstantMesh source into third_party/,
                                    # install nvdiffrast, xatlas, PyMCubes, warm rembg u2net
python app.py                       # UI at http://localhost:7860
```

**CPU-only machines:** `MOCK_GPU` activates automatically (no GPU detected), so the entire pipeline runs end-to-end with a distance-transform mock render and a vertex-colored icosphere mesh — perfect for UI development and pipeline wiring checks. Set `MOCK_GPU=1` to force it even on GPU machines.

**Self-verify individual stages (no UI):**

```bash
python src/logger.py
python -m src.preprocess
python -m src.sketch_to_2d
python -m src.image_to_3d
python -m src.mesh_validator
python -m pipeline              # full end-to-end, exports outputs/run_*.glb/.obj
```

## Expected inputs & outputs

- **Input:** in-app sketch (Sketchpad), uploaded sketch image (PNG/JPG), optional text prompt (object, material, color, style), seed.
- **Output:** `outputs/run_YYYYMMDD_HHMMSS.glb` (primary, vertex-colored), `outputs/run_*.obj` (compatibility fallback), plus the pipeline intermediate gallery and a validation/timing JSON report.

## Dependencies & licenses

| Component | Model ID / Package | License |
| --- | --- | --- |
| Stable Diffusion 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML OpenRAIL-M |
| ControlNet-Scribble | `lllyasviel/sd-controlnet-scribble` | Apache-2.0 / OpenRAIL-M |
| Zero123++ | `sudo-ai/zero123plus-v1.2` | Apache-2.0 |
| InstantMesh (weights) | `TencentARC/InstantMesh` | Apache-2.0 |
| InstantMesh (source) | `TencentARC/InstantMesh` — cloned to `third_party/` by `download_weights.py` (repo is not pip-installable; runtime deps in requirements.txt) | Apache-2.0 |
| nvdiffrast | `git+https://github.com/NVlabs/nvdiffrast/` — installed by `download_weights.py` (required for FlexiCubes rasterization) | NVIDIA Source Code License |
| rembg | PyPI `rembg` (u2net) | MIT |
| trimesh | PyPI `trimesh` | MIT |
| Gradio (pinned) | `gradio==4.44.0` | Apache-2.0 |

## Runtime table (Colab/Kaggle T4, measured assumptions)

| Stage | Component | Time |
| --- | --- | --- |
| 1 | Preprocess (Otsu + fallback) | ~0.2 s |
| 2 | ControlNet 2D synthesis | ~4–6 s |
| 3 | rembg isolation + recenter | ~1.0 s |
| 4a | Zero123++ multi-view generation (75 steps) | ~15–25 s |
| 4b | LRM FlexiCubes mesh extraction | ~10–20 s |
| 5 | Validation + export + reload | ~0.5 s |
| — | **Total** | **~30–55 s** |

CPU mock mode: full pipeline ~2–3 s.

## Hardware assumptions & limitations

- **Target hardware:** NVIDIA T4 16GB (Colab or Kaggle). Stage 2 pinned to `cuda:0`, Stage 4 to `cuda:1` when available; both share `cuda:0` on single-GPU machines. On single GPU, the 2D pipeline is unloaded before 3D loads, and Zero123++ is unloaded before the LRM loads.
- **Single-view occlusion:** unseen backside geometry is hallucinated from multi-view priors; InstantMesh generates 6 consistent views to reduce ambiguity, but it is plausible, not exact.
- **FlexiCubes resolution:** grid_res=128 captures most geometric detail; very thin features (wireframes, antennas) may be simplified.
- **Vertex-color texturing:** InstantMesh queries vertex colors from the triplane; fine texture detail is represented as vertex colors, not UV-mapped textures.
- **VRAM floor:** ~12GB free needed for the full GPU path (Zero123++ and LRM are loaded/unloaded sequentially); below that, force `MOCK_GPU=1`.

## Reproducibility

- All model weights are fetched from the verified active Hugging Face repos listed above via `download_weights.py` (`huggingface_hub.snapshot_download` and `hf_hub_download`); InstantMesh's source is shallow-cloned from GitHub into `third_party/InstantMesh` by the same script, and nvdiffrast is installed by the same script from GitHub.
- `gradio==4.44.0`, `numpy<3.0`, and all other versions pinned in `requirements.txt`.
- Every pipeline run (success **and** failure) is logged with IST timestamps, stage timings, and mesh metrics in `done_so_far.md` / `what_we_did.md` — the project's continuous audit trail, generated by `src/logger.py`.
- Deterministic seeds: same sketch + same seed → same 2D render and multi-view generation → same mesh.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: config` / `src` | Run as a module from project root: `python -m pipeline`, not `python pipeline/pipeline.py` |
| First rembg call hangs | Downloading u2net.onnx (~170 MB, one time) — wait or pre-warm via `python download_weights.py` |
| NumPy 2.x import errors from torch/rembg | `pip install "numpy<2.0"` |
| CUDA OOM on Stage 4 | Zero123++ should be unloaded before LRM loads; if still OOM, try on Colab or force `MOCK_GPU=1` |
| Flat plane rejected on valid sketch | Retry insurance re-runs with seed+1 automatically; if it persists, the sketch may need stronger depth cues (shading lines) |
| `ModuleNotFoundError: src.models` | Run `python download_weights.py` once (clones InstantMesh source into `third_party/`) |
| nvdiffrast build fails | Requires CUDA toolkit matching your PyTorch; install via `pip install git+https://github.com/NVlabs/nvdiffrast/` |
| nvdiffrast build fails on Windows | Install Visual Studio Build Tools (C++ workload) and CUDA toolkit; or develop on CPU mock mode (`MOCK_GPU=1`) |

## Role summary (from the audit log)

| Module | Role | Rubric mapping |
| --- | --- | --- |
| `src/logger.py` + audit files | Role C (Docs & Audit) | Documentation 15 pts, Reproducibility 15 pts |
| `config.py`, `download_weights.py` | Role C (Infra Lead) | Reproducibility 15 pts |
| `src/preprocess.py` | Role A (2D Vision Lead) | Technical Implementation 25 pts |
| `src/sketch_to_2d.py` | Role A (2D Vision Lead) | 3D Output Quality 25 pts |
| `src/image_to_3d.py` | Role B (3D Lead) | 3D Output Quality 25 pts, Efficiency 20 pts |
| `src/mesh_validator.py` | Role B (3D Lead) | Pipeline Validity 20 pts |
| `pipeline.py`, `app.py` | Role C (UI & Integration) | End-to-End Prototype 20 pts |