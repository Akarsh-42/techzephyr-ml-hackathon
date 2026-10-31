# Sketch2Shape AI — Sketch to Textured 3D Asset

Single-view 2D sketch → validated, textured, exportable 3D asset (.glb/.obj) in one click. Built for the TechZephyr 4.0 24-hour hackathon.

## Pipeline (5 stages)

```
Sketch (canvas/upload) ─► Stage 1 Preprocess ─► Stage 2 ControlNet-Scribble + SD 1.5
        │                        (OpenCV)              (sketch-guided render)
        ▼
 Stage 3 rembg isolation ─► Stage 4 3D Reconstruction ─► Stage 5 Validation & Export
                          (TripoSR / InstantMesh)        (trimesh → .glb + .obj)
```

## Quickstart

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_weights.py      # pre-cache model weights (~8.5GB; HF_TOKEN env var if needed)
python app.py                   # full pipeline (needs CUDA GPU)
S2S_MOCK=1 python app.py        # mock mode — full UI, no GPU (Windows: $env:S2S_MOCK="1")
python tests/test_pipeline_mock.py   # automated smoke test, no GPU
```

## Environment variables

| Variable | Purpose |
|---|---|
| `S2S_MOCK=1` | Mock mode: full UI/pipeline without GPU (uses procedural icosphere) |
| `S2S_BACKEND` | `auto` \| `tripo` \| `instantmesh` \| `mock` — 3D backend selection |
| `HF_HOME` | Point HF cache at the mounted weights (e.g. `/kaggle/input/sketch2shape-weights`) |
| `HF_TOKEN` | Read-only Hugging Face token, private env var — NEVER commit |

## Models & licenses

| Component | Model / Repo | License |
|---|---|---|
| 2D synthesis base | `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML OpenRAIL-M |
| Sketch control | `lllyasviel/sd-controlnet-scribble` | OpenRAIL |
| 3D reconstruction (baseline) | `stabilityai/TripoSR` | MIT |
| 3D reconstruction (benchmark) | `TencentARC/InstantMesh` | apache-2.0 (code) — verify weights card |
| Foreground isolation | rembg / u2net.onnx | MIT (Apache-2.0 model) |

<!-- TODO (Hour 14–20): measured per-stage runtime table on the demo GPU, e.g.
| Stage | Hardware | Time |
|---|---|---|
| Preprocess | CPU | 0.1s |
| Diffusion (20 steps) | T4 cuda:0 | ~6–10s |
| rembg | CPU | ~5s |
| TripoSR (MC 256) | T4 cuda:1 | 5–15s |
| Validate + export | CPU | <1s |
-->

## Hardware assumptions

- Development: any CPU (mock mode). GPU needed only for integration/benchmark blocks.
- Demo: Kaggle T4 x2 — Stage 2 pinned to `cuda:0`, Stage 4 to `cuda:1`. Dual T4s do NOT pool VRAM.

## Known limitations (be candid — judges reward this)

- TripoSR outputs vertex colors, not baked UV textures; InstantMesh exports textured meshes but is slower on T4.
- Single-view reconstruction cannot recover true occluded geometry; backs of objects are inferred.
- Very faint sketches skip binarization (fallback path) and may lose stroke fidelity.
- Mock mode returns a procedural sphere and is for UI development only.

## Repository layout

```
app.py  pipeline.py  config.py  download_weights.py  requirements.txt
src/  preprocess.py  sketch_to_2d.py  image_to_3d.py  mesh_validator.py  logger.py
tests/  test_pipeline_mock.py
docs/  planning_v2.md  setup.md
examples/  outputs/  (gitignored)
```

## Credits

Built on TripoSR (VAST-AI), InstantMesh (TencentARC), Stable Diffusion v1.5, ControlNet, rembg, trimesh, Gradio. Engineering log: `done_so_far.md`.