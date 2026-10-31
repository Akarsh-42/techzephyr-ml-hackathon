# Sketch2Shape AI — TechZephyr 4.0 ML Hackathon

Turn a single-view 2D sketch into a textured, downloadable 3D asset (.glb/.obj).

**Status:** Steps 0–3 complete (audit logger, config/weights, sketch preprocessing, ControlNet 2D synthesis + rembg foreground isolation). Steps 4–6 (3D reconstruction, mesh validation, pipeline + Gradio UI) in progress.

## Quickstart

```bash
pip install -r requirements.txt
python download_weights.py        # one-time weight prefetch (SD1.5, ControlNet-Scribble, TripoSR) + rembg warm-up
python -m src.preprocess          # self-verify Stage 1
python -m src.sketch_to_2d        # self-verify Stages 2-3 (CPU mock mode works out of the box)
```

On machines without a GPU, `MOCK_GPU` activates automatically so the whole 2D path runs on CPU.

## Architecture (per execution blueprint)

```
Sketch (draw/upload)
  → Stage 1  src/preprocess.py     Otsu binarization + faint-sketch Canny fallback, 512x512
  → Stage 2  src/sketch_to_2d.py    SD1.5 + ControlNet-Scribble (cuda:0), 20 steps, UniPC
  → Stage 3  src/sketch_to_2d.py    rembg RGBA isolation, 85% frame occupancy recentering
  → Stage 4  src/image_to_3d.py     TripoSR (white BG) / InstantMesh (gray BG), cuda:1   [todo]
  → Stage 5  src/mesh_validator.py  validation, anti-plane check, .glb/.obj export      [todo]
  → Gradio 4.44.0 UI (app.py)       viewer + downloads + timing report                   [todo]
```

## Project layout

| Path | Purpose |
| --- | --- |
| `config.py` | Model IDs, per-backend BG colors, device pinning, thresholds |
| `download_weights.py` | 1-click reproducible weight caching + rembg warm-up |
| `src/logger.py` | IST-timestamped audit log → `done_so_far.md` / `what_we_did.md` |
| `src/preprocess.py` | Stage 1: normalization + Otsu + faint-sketch fallback |
| `src/sketch_to_2d.py` | Stages 2-3: ControlNet synthesis + 85% foreground isolation |
| `ML_Hackathonv2.txt` | Problem statement |
| `execution_blueprint_architecture.md` | Canonical execution blueprint (single source of truth) |

## References

- Problem statement: `ML_Hackathonv2.txt`
- Execution blueprint: `execution_blueprint_architecture.md`

Full documentation (licenses, runtime table, hardware, limitations) ships with Step 6.