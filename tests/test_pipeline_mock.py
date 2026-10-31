"""Automated smoke test — runs the full pipeline in MOCK mode on a synthetic sketch.
No GPU, no model downloads, no Gradio. Verifies Stage 1 preprocessing, blank-canvas
rejection, all input types, orchestration, mock reconstruction, Stage 5 validation
(incl. anti-plane check + reload verify), and dual .glb/.obj export.

Usage:  python tests/test_pipeline_mock.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["S2S_MOCK"] = "1"  # must be set before importing config/pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

PASS, FAIL = "\u2705 PASS", "\u274c FAIL"


def synth_sketch():
    """Simulate a Gradio Sketchpad dict: white canvas, black scribble strokes."""
    img = np.full((512, 512, 3), 255, dtype=np.uint8)  # white canvas
    img[120:400, 150:180, :] = 0   # vertical stroke
    img[250:280, 150:400, :] = 0   # horizontal stroke
    img[120:150, 150:400, :] = 0   # top stroke -> forms a closed "chair-ish" shape
    return {"composite": img, "background": img, "layers": []}


def main() -> int:
    failures = []
    print("=== Sketch2Shape mock smoke test ===")

    # --- Stage 1: preprocessing (real code path, no GPU) ---
    try:
        from src.preprocess import preprocess_sketch, normalize_sketch_input
        prepped = preprocess_sketch(synth_sketch())
        assert prepped.size == (512, 512), f"expected 512x512, got {prepped.size}"
        arr = np.asarray(prepped)
        assert arr.std() > 2.0, "preprocessed image looks blank"
        print(f"{PASS} Stage 1 preprocess -> 512x512, std={arr.std():.1f}")
    except Exception as e:
        failures.append(("Stage 1 preprocess", repr(e)))
        print(f"{FAIL} Stage 1 preprocess: {e}")
        return _summary(failures)

    # --- Blank-canvas rejection ---
    try:
        blank = np.full((512, 512, 3), 255, dtype=np.uint8)
        try:
            preprocess_sketch({"composite": blank})
            raise AssertionError("blank canvas was NOT rejected")
        except ValueError:
            print(f"{PASS} Blank-canvas rejection works")
    except Exception as e:
        failures.append(("blank rejection", repr(e)))
        print(f"{FAIL} Blank-canvas rejection: {e}")

    # --- Input-type flexibility ---
    try:
        from PIL import Image
        normalize_sketch_input(Image.new("RGB", (64, 64), (0, 0, 0)))
        normalize_sketch_input(np.zeros((64, 64, 3), dtype=np.uint8))
        print(f"{PASS} normalize_sketch_input accepts dict / PIL / ndarray")
    except Exception as e:
        failures.append(("normalize types", repr(e)))
        print(f"{FAIL} normalize_sketch_input: {e}")

    # --- Full pipeline in mock mode ---
    try:
        from pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as td:
            out = run_pipeline(synth_sketch(), prompt="", seed=42)
        assert out.get("error") is None, f"pipeline error: {out.get('error')}"
        rep = out["report"]
        assert rep["vertices"] > 100 and rep["faces"] > 100, "mesh below thresholds"
        assert rep.get("min_axis_depth", 0) >= 0.05, "anti-plane check failed"
        assert rep.get("reload_verified") is True, "reload verification missing"
        assert Path(rep["glb_path"]).exists() and Path(rep["obj_path"]).exists()
        print(f"{PASS} Pipeline (mock): verts={rep['vertices']}, faces={rep['faces']}, "
              f"depth={rep['min_axis_depth']:.3f}, glb={rep['glb_kb']}KB, total={out['total_s']}s")
    except Exception as e:
        failures.append(("pipeline mock", repr(e)))
        print(f"{FAIL} Pipeline (mock): {e}")

    # --- Logger wrote an entry ---
    try:
        from config import LOG_FILE
        assert LOG_FILE.exists() and "Pipeline" in LOG_FILE.read_text(encoding="utf-8")
        print(f"{PASS} done_so_far.md logging works")
    except Exception as e:
        failures.append(("logger", repr(e)))
        print(f"{FAIL} Logger: {e}")

    return _summary(failures)


def _summary(failures):
    print("\n=== RESULT:", "ALL TESTS PASSED — repo is healthy, proceed to GPU run" if not failures
          else f"{len(failures)} FAILURE(S) — fix before proceeding ===")
    for name, err in failures:
        print(f"  - {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())