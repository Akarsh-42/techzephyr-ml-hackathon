"""Master orchestrator: sketch -> preprocess -> ControlNet -> rembg -> 3D -> validate/export.
Includes: per-stage timing, retry-once on validation failure (seed+1), mock mode."""
import time
import uuid
import traceback
from config import DEFAULT_CONTROL_SCALE, DEFAULT_SEED, MOCK_GPU
from src.preprocess import preprocess_sketch
from src.logger import log_feature_progress


def run_pipeline(sketch_input, prompt="", control_scale=DEFAULT_CONTROL_SCALE,
                 seed=DEFAULT_SEED) -> dict:
    """Returns dict: preprocessed, render, segmented, report, timings, error."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    timings, result = {}, {"error": None}
    t_all = time.time()

    try:
        # Stage 1
        t = time.time()
        prepped = preprocess_sketch(sketch_input)
        timings["preprocess_s"] = round(time.time() - t, 2)
        result["preprocessed"] = prepped

        if MOCK_GPU:
            from src.image_to_3d import reconstruct_3d_mesh
            t = time.time()
            mesh = reconstruct_3d_mesh(prepped, backend="mock")
            timings["reconstruct_s"] = round(time.time() - t, 2)
            result["render"] = result["segmented"] = prepped
        else:
            # Stage 2/3 (ControlNet + rembg)
            from src.sketch_to_2d import generate_reference_image
            t = time.time()
            rendered, segmented = generate_reference_image(prepped, prompt,
                                                            control_scale, seed)
            timings["diffusion_s"] = round(time.time() - t, 2)
            result["render"], result["segmented"] = rendered, segmented

            # Stage 4 (3D reconstruction)
            from src.image_to_3d import reconstruct_3d_mesh
            t = time.time()
            mesh = reconstruct_3d_mesh(segmented)
            timings["reconstruct_s"] = round(time.time() - t, 2)

        # Stage 5 (validate + export); retry 3D once if mesh is unusable
        from src.mesh_validator import validate_and_export_mesh, is_valid
        t = time.time()
        try:
            report = validate_and_export_mesh(mesh, run_id=run_id)
        except ValueError:
            if MOCK_GPU:
                raise
            from src.image_to_3d import reconstruct_3d_mesh
            mesh = reconstruct_3d_mesh(result["segmented"])  # implicit new random draw
            report = validate_and_export_mesh(mesh, run_id=run_id + "_retry")
            report["retried"] = True
        timings["export_s"] = round(time.time() - t, 2)

        result["report"] = report
        result["timings"] = timings
        result["total_s"] = round(time.time() - t_all, 2)
        log_feature_progress("Pipeline run", [run_id],
                             f"End-to-end run OK in {result['total_s']}s; timings={timings}",
                             duration_or_latency=f"{result['total_s']}s", author_or_agent="pipeline")
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        log_feature_progress("Pipeline FAILURE", [run_id], result["error"],
                             author_or_agent="pipeline")
        return result