# Sketch2Shape AI — TechZephyr 4.0 ML Hackathon

Turn a single-view 2D sketch into a textured, downloadable 3D asset (.glb / .obj).

- **Problem statement:** see `ML_Hackathonv2.txt`
- **Canonical execution blueprint:** see `execution_blueprint_architecture.md`

## Objective

Anyone with a rough pencil sketch can produce a usable 3D model — draw or upload a sketch, optionally describe the object with a text prompt, and receive an interactive 3D preview plus verified `.glb`/`.obj` downloads. The pipeline preserves the sketch's structure (via ControlNet conditioning) while adding plausible depth, shading, and color.

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
[Stage 4] src/image_to_3d.py     TripoSR Marching Cubes (res 256) on cuda:1/cuda:0
        │                        BG routing via config: TripoSR → white, InstantMesh → gray
        ▼
[Stage 5] src/mesh_validator.py  8-step validation: floor reject → topology purge → unit normalize
        │                        → anti-plane (min_axis_depth ≥ 0.05) → Laplacian smooth
        │                        → dual GLB/OBJ export → trimesh.load() reload verification
        ▼
app.py (Gradio 4.4.0)           3D viewer, .glb/.obj downloads, per-stage timing JSON report
```

**Retry insurance:** if any of Stages 2–5 fails, `pipeline.py` silently re-runs once with seed + 1 before any error reaches the UI.

## Quickstart

```bash
pip install -r requirements.txt     # includes TripoSR's runtime deps + torchmcubes
python download_weights.py          # one-time: prefetch SD1.5/ControlNet/TripoSR weights, clone
                                    # TripoSR source into third_party/, warm rembg u2net
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
| TripoSR (weights) | `stabilityai/TripoSR` | MIT |
| TripoSR (source) | `VAST-AI-Research/TripoSR` — cloned to `third_party/` by `download_weights.py` (repo is not pip-installable; runtime deps in requirements.txt) | MIT |
| torchmcubes | `git+https://github.com/tatsy/torchmcubes.git` | MIT |
| rembg | PyPI `rembg` (u2net) | MIT |
| trimesh | PyPI `trimesh` | MIT |
| Gradio (pinned) | `gradio==4.44.0` | Apache-2.0 |

## Runtime table (Kaggle T4 x2, measured assumptions)

| Stage | Component | Time |
| --- | --- | --- |
| 1 | Preprocess (Otsu + fallback) | ~0.2 s |
| 2 | ControlNet 2D synthesis | ~4–6 s |
| 3 | rembg isolation + recenter | ~1.0 s |
| 4 | TripoSR reconstruction | ~5–15 s |
| 5 | Validation + export + reload | ~0.5 s |
| — | **Total** | **~12–22 s** |

CPU mock mode: full pipeline ~2–3 s.

## Hardware assumptions & limitations

- **Target hardware:** NVIDIA T4 16GB (x2 on Kaggle). Stage 2 pinned to `cuda:0`, Stage 4 to `cuda:1` when available; both share `cuda:0` on single-GPU machines. Dual T4s do **not** pool VRAM — hence the strict 2D-pipeline unload before 3D load.
- **Single-view occlusion:** unseen backside geometry is hallucinated from priors; it is plausible, not exact.
- **Marching-cubes quantization:** very thin features (wireframes, antennas) below the 256³ grid resolution may be lost or merged.
- **Vertex-color texturing:** TripoSR output carries vertex colors sampled from the generated image; fine texture detail is not preserved.
- **VRAM floor:** ~6GB free needed for the GPU path; below that, force `MOCK_GPU=1`.
- **InstantMesh backend:** wired for background compositing but pending the benchmark gate; `triposr` is the default `PRIMARY_3D_BACKEND`.

## Reproducibility

- All model weights are fetched from the verified active Hugging Face repos listed above via `download_weights.py` (`huggingface_hub.snapshot_download` into `weights_cache/`); TripoSR's `tsr` source is shallow-cloned from GitHub into `third_party/TripoSR` by the same script.
- `gradio==4.44.0`, `numpy<2.0`, and all other versions pinned in `requirements.txt`.
- Every pipeline run (success **and** failure) is logged with IST timestamps, stage timings, and mesh metrics in `done_so_far.md` / `what_we_did.md` — the project's continuous audit trail, generated by `src/logger.py`.
- Deterministic seeds: same sketch + same seed → same 2D render and mesh.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: config` / `src` | Run as a module from project root: `python -m pipeline`, not `python pipeline/pipeline.py` |
| First rembg call hangs | Downloading u2net.onnx (~170 MB, one time) — wait or pre-warm via `python download_weights.py` |
| NumPy 2.x import errors from torch/rembg | `pip install "numpy<2.0"` |
| CUDA OOM on Stage 4 | 2D pipeline unload should have freed VRAM; if not, reduce `mc_resolution` or force `MOCK_GPU=1` |
| Flat plane rejected on valid sketch | Retry insurance re-runs with seed+1 automatically; if it persists, the sketch may need stronger depth cues (shading lines) |
| `ModuleNotFoundError: tsr` | Run `python download_weights.py` once (clones TripoSR source into `third_party/`) |
| torchmcubes build fails | Requires C++ compiler (`build-essential` on Linux / VS Build Tools on Windows) — it compiles CUDA/C++ marching cubes on install |

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