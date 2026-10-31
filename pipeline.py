"""Step 6 — Master orchestrator: per-stage timing, silent retry-once (seed+1), audit logging."""
from __future__ import annotations

import os
import time
from datetime import datetime

import config
from src import sketch_to_2d
from src.image_to_3d import reconstruct_3d_mesh
from src.logger import log_pipeline_run
from src.mesh_validator import validate_and_export_mesh
from src.preprocess import normalize_sketch_input, preprocess_sketch

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def run_pipeline(
    sketch_input,
    upload_input,
    user_prompt: str = "",
    seed: int = 42,
    mc_resolution: int = 256,
) -> dict:
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timings: dict = {}
    t0 = time.perf_counter()

    raw = normalize_sketch_input(sketch_input, upload_input)
    scribble = preprocess_sketch(raw)
    timings["stage1_preprocess_s"] = round(time.perf_counter() - t0, 2)

    status, shaded_2d, isolated, mesh_report, error = "FAILURE", None, None, None, ""

    for attempt, run_seed in enumerate((seed, seed + 1), start=1):
        try:
            shaded_2d, isolated = sketch_to_2d.generate_2d_and_isolate_foreground(
                scribble, user_prompt, seed=run_seed
            )
            timings["stage2_controlnet_s"] = round(sketch_to_2d.LAST_TIMINGS["controlnet_s"], 2)
            timings["stage3_rembg_s"] = round(sketch_to_2d.LAST_TIMINGS["rembg_s"], 2)

            t = time.perf_counter()
            mesh = reconstruct_3d_mesh(
                isolated,
                backend=config.PRIMARY_3D_BACKEND,
                seed=run_seed,
                mc_resolution=int(mc_resolution),
            )
            timings["stage4_3d_recon_s"] = round(time.perf_counter() - t, 2)

            t = time.perf_counter()
            mesh_report = validate_and_export_mesh(mesh, OUTPUT_DIR, run_id)
            timings["stage5_validate_export_s"] = round(time.perf_counter() - t, 2)

            status = "SUCCESS" if attempt == 1 else "RETRY_SUCCESS"
            break
        except Exception as exc:
            error = str(exc)
            if attempt > 1:
                break

    timings["total_runtime_s"] = round(time.perf_counter() - t0, 2)
    log_pipeline_run(run_id, status, timings, mesh_report or {}, notes=error)

    if status == "FAILURE":
        raise RuntimeError(f"Pipeline failed after retry (seeds {seed} and {seed + 1}): {error}")

    return {
        "scribble": scribble,
        "shaded_2d": shaded_2d,
        "isolated_rgba": isolated,
        "glb_path": mesh_report["glb_path"],
        "obj_path": mesh_report["obj_path"],
        "report": {**mesh_report, "run_id": run_id, "status": status, "timings": timings},
    }


if __name__ == "__main__":
    from PIL import Image, ImageDraw

    from src.logger import log_feature_progress

    canvas = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(canvas).ellipse([180, 120, 460, 360], outline=(20, 20, 20), width=4)

    result = run_pipeline({"composite": canvas}, None, "a wooden bowl", seed=42)
    assert result["report"]["reload_verified"] is True
    assert os.path.getsize(result["glb_path"]) > 1024

    log_feature_progress(
        feature_name="Step 6 — Master Orchestrator (pipeline.py)",
        role_owner="Role C (UI & Integration)",
        role_importance_and_job=(
            "Delivers End-to-End Pipeline Validity (20 pts) with zero live-demo crashes: "
            "per-stage timing, silent retry-once (seed+1) insurance, and audit logging of "
            "every run, success and failure."
        ),
        files_modified=["pipeline.py", "src/sketch_to_2d.py"],
        technical_explanation=(
            "Stage 1 runs once (deterministic); Stages 2-5 are wrapped in a two-attempt loop: "
            "attempt 2 silently re-runs with seed+1 on any Stage 2-5 exception (Rule 4) before "
            "the error ever reaches the UI. Stage 2/3 split timings come from "
            "sketch_to_2d.LAST_TIMINGS; stages 4-5 use time.perf_counter(). Every run calls "
            "log_pipeline_run with status SUCCESS/RETRY_SUCCESS/FAILURE. mc_resolution (UI "
            "slider, 128-512) is passed through to reconstruct_3d_mesh so judges can trade "
            "mesh detail against latency live."
        ),
        duration_or_latency=result["report"]["timings"]["total_runtime_s"],
    )
    print(f"Pipeline verified: {result['report']['status']} -> {result['glb_path']}")