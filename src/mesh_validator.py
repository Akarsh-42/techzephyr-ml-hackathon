"""Stage 5 — 8-step mesh validation, smoothing, dual GLB/OBJ export & reload verification."""
from __future__ import annotations

import os

import numpy as np
import trimesh

import config


def validate_and_export_mesh(
    mesh: trimesh.Trimesh,
    output_dir: str,
    run_id: str,
    smoothing_iterations: int = config.LAPLACIAN_ITERATIONS,
) -> dict:
    # 1. Strict vertex/face floor reject
    if len(mesh.vertices) < config.MIN_VERTICES or len(mesh.faces) < config.MIN_FACES:
        raise ValueError("Mesh failed minimum geometry threshold (<100 verts/faces).")

    # 2. Topology purge
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()

    # 3. Centroid & unit normalization (longest axis fits [-0.5, 0.5])
    mesh.vertices -= mesh.centroid
    mesh.vertices /= float(np.max(mesh.extents))

    # 4. Anti-plane depth check
    min_axis_depth = float(np.min(mesh.extents) / np.max(mesh.extents))
    if min_axis_depth < config.MIN_AXIS_DEPTH:
        raise ValueError(
            f"Flat plane detected: min_axis_depth={min_axis_depth:.4f} < {config.MIN_AXIS_DEPTH}"
        )

    # 5. Normals & controlled Laplacian smoothing
    trimesh.repair.fix_normals(mesh)
    trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)

    # 6. Dual export + file-size floor check
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{run_id}.glb")
    obj_path = os.path.join(output_dir, f"{run_id}.obj")
    mesh.export(glb_path)
    mesh.export(obj_path)
    for p in (glb_path, obj_path):
        if not os.path.isfile(p) or os.path.getsize(p) <= 1024:
            raise ValueError(f"Export failed or file too small: {p}")

    # 7. Post-export reload verification
    reloaded = trimesh.load(glb_path, force="mesh")
    reload_verified = len(reloaded.vertices) >= config.MIN_VERTICES

    # 8. Structured validation report
    return {
        "glb_path": glb_path,
        "obj_path": obj_path,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "bbox_extents": np.asarray(mesh.extents).round(4).tolist(),
        "min_axis_depth": round(min_axis_depth, 4),
        "glb_size_kb": round(os.path.getsize(glb_path) / 1024, 1),
        "obj_size_kb": round(os.path.getsize(obj_path) / 1024, 1),
        "reload_verified": reload_verified,
    }


if __name__ == "__main__":
    import tempfile

    from src.image_to_3d import _mock_mesh
    from src.logger import log_feature_progress

    report = validate_and_export_mesh(
        _mock_mesh(), tempfile.mkdtemp(prefix="s2s_step5_"), "step5_selfcheck"
    )
    assert report["vertex_count"] >= config.MIN_VERTICES
    assert report["face_count"] >= config.MIN_FACES
    assert report["min_axis_depth"] >= config.MIN_AXIS_DEPTH
    assert report["reload_verified"] is True

    flat = trimesh.creation.box(extents=(1.0, 1.0, 0.001))
    try:
        validate_and_export_mesh(flat, tempfile.mkdtemp(prefix="s2s_flat_"), "flat_plane")
        raise AssertionError("Anti-plane check failed to reject a flat mesh.")
    except ValueError as e:
        assert "Flat plane detected" in str(e)

    tiny = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    tiny.vertices = tiny.vertices[:50]
    tiny.faces = tiny.faces[:10]
    try:
        validate_and_export_mesh(tiny, tempfile.mkdtemp(prefix="s2s_tiny_"), "tiny_mesh")
        raise AssertionError("Vertex/face floor failed to reject a tiny mesh.")
    except ValueError as e:
        assert "minimum geometry threshold" in str(e)

    log_feature_progress(
        feature_name="Step 5 — 8-Step Mesh Validation & Verified Export",
        role_owner="Role B (3D Lead)",
        role_importance_and_job=(
            "Guarantees Pipeline Validity (20 pts) & judge file opening (Reproducibility 15 pts): "
            "enforces >=100 verts/faces, min_axis_depth >=0.05 anti-plane check, 1-iter "
            "Laplacian smooth, .glb/.obj dual export, and trimesh.load() reload verification."
        ),
        files_modified=["src/mesh_validator.py"],
        technical_explanation=(
            "Floor reject (<100 verts OR faces) per Rule 5; topology purge via nondegenerate/"
            "unique face masks + merge_vertices + unreferenced removal; centroid centering then "
            "unit-normalization (longest axis -> [-0.5, 0.5]); anti-plane ratio min/max extent "
            "with hard 0.05 floor per Rule 6 (never 1e-4); trimesh.repair.fix_normals + "
            "filter_laplacian(iterations=1); dual GLB (colors preserved) and OBJ export with "
            ">1024-byte checks per Rule 7; trimesh.load(force='mesh') reload check proving the "
            "downloaded artifact opens. Self-check also asserts a flat plane and a tiny mesh "
            "are both rejected."
        ),
        duration_or_latency="~0.5s per mesh on CPU",
    )
    print(f"Step 5 verified: {report['vertex_count']} verts, reload_verified={report['reload_verified']}")