# Environment Setup & Team Logistics Guide (TechZephyr 4.0) — v2

Operational runbook: pre-hackathon verification, cloud GPU limits, role split, persistent weight caching, and 3-tier demo-day failover. Companion doc to `planning_v2.md` (that = what to build; this = where/how to run it).

## 1. Hour -1: Pre-Hackathon Verification Checklist

- **Individual phone-verified Kaggle accounts** (one per teammate). Phone verification unlocks GPU + internet. Never share an account across two active GPU notebooks — it burns the ~30h/week quota at double speed.
- **Scoped read-only Hugging Face token** (HF_TOKEN) stored in private env vars only. Never commit. Rotate after the final demo.
- **Attach-and-terminate test:** open a blank Kaggle notebook, enable Internet + T4 x2, verify both GPUs register, Stop Session immediately.
- **Golden GPU rule (CPU-first):** author all code with `S2S_MOCK=1` on CPU. Attach GPU only during scheduled 30–60 min integration/benchmark blocks. Per-role budget: max 6 GPU-hours each across the event.
- ⚠️ **Verify all quotas on event day** — free-tier limits (session caps, weekly hours) shift without notice. Check account dashboards, not yesterday's numbers.

## 2. Cloud Platform Reality

| Platform | Hardware | Verified limits | Strategic role |
| --- | --- | --- | --- |
| Kaggle Notebooks | GPU T4 x2 (15GB each) | 12h max session; ~30 GPU-hrs/week; quota ticks from GPU attach | Primary integration + live demo host; pin Stage 2→cuda:0, Stage 4→cuda:1 (GPUs do NOT pool VRAM) |
| Lightning AI Studios | T4 16GB (credit-backed) | ~15 credits/mo (~22 GPU-hrs); 4h forced restart; 50GB persistent SSD | Persistent disk base + Tier-2 demo standby; stay CPU mode until failover |
| Google Colab Free | T4 15GB | ~90-min idle disconnect; unpredictable cutoffs; tunnels throttled | Emergency scratchpad only — never host the demo here |
| Local laptop | CPU | — | Frontend/geometry sandbox (mock mode) |

## 3. Role Split

- **Role A (2D track):** `src/preprocess.py` + `src/sketch_to_2d.py`. Kaggle #1, cuda:0. Budget ≤6 GPU-hrs.
- **Role B (3D track):** `src/image_to_3d.py` + `src/mesh_validator.py`. Kaggle #2 or Lightning. Benchmark TripoSR vs InstantMesh at the Hour 5–8 gate. Budget ≤6 GPU-hrs.
- **Role C (frontend/repro/docs):** `config.py`, `app.py`, `pipeline.py`, `download_weights.py`, README, examples/, `done_so_far.md` init. Local CPU + Kaggle #3 reserved for demo host.

## 4. System & Package Rules

- apt (Ubuntu 22.04+): `libgl1 libglib2.0-0 git-lfs`. **`libgl1-mesa-glx` no longer exists — never use it.**
- Pin everything via `requirements.txt` (v2 pins). After the FIRST successful end-to-end run, re-pin from `pip freeze` and freeze permanently.
- Commit + push every 60 minutes; log progress in `done_so_far.md`. `.gitignore` excludes weights/caches/outputs.

## 5. Shared Weight Cache (Zero Venue Wi-Fi Dependency)

~8.5GB stack. Cloud sessions wipe `~/.cache/huggingface` on restart.

1. `download_weights.py` fetches: SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5` — never `runwayml/...`, dead repo), `lllyasviel/sd-controlnet-scribble`, TripoSR, u2net via rembg.
2. Role C runs it in Kaggle with output `/kaggle/working/weights_cache` → Save Version → New Dataset `sketch2shape-weights` (Private) → add A & B as collaborators.
3. Every session: **+ Add Input** → dataset mounts read-only at `/kaggle/input/sketch2shape-weights` → export `HF_HOME=/kaggle/input/sketch2shape-weights` and `U2NET_HOME=.../u2net`. Zero-second restore, no network.

## 6. 3-Tier Demo-Day Failover (Hours 20–24)

Judges may provide a fresh sketch live. Three layers:

- **Tier 1 (primary):** Hour 20 — Role C kills old sessions, starts a FRESH Kaggle T4 x2 session (full 12h window), mounts weights dataset, pulls main branch, verifies cuda:0/cuda:1 pinning, `demo.launch(share=True)`. Run 2 warm-up sketches (CUDA compile + hot weights).
- **Tier 2 (hot standby):** final main synced on Lightning AI with cached weights on the 50GB SSD; keep in CPU mode until needed; flip to T4 + native Port Viewer (7860) on trouble. Failover <90s, zero downloads.
- **Tier 3 (insurance):** uncut 2-min screen recording of the full flow (draw → terminal logs stage 1–5 → viewport rotation → .glb/.obj opened in Blender/3D Viewer) + pre-generated example pairs in `examples/` on the local laptop SSD. Never present a cached example as live — label it.

## 7. Pre-Event Preflight (T-0 checklist)

1. Smoke test passes locally: `python tests/test_pipeline_mock.py` → ALL TESTS PASSED.
2. One real end-to-end GPU run on Kaggle done before the event; `.glb` verified in a viewer.
3. Quotas re-checked; weights dataset published and mounted clean in a fresh session.
4. 10 judge-simulation sketches collected (faint pencil, photo-of-paper, colored marker, gaps, blank).
5. Backup video recorded; GitHub repo pushed.