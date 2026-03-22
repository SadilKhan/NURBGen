# system imports
import json
import sys
import os
import argparse
import signal
import traceback
import multiprocessing
from pathlib import Path
from glob import glob
from functools import partial

# local imports
from utils.utils import read_step_file, normalize_shape
from utils.approximation_score import compare_faces
from utils.logger import setup_logger
from functions.reconstruct_shape import (
    reconstructBSplineSurface, build_face_from_loops,
    reconstruct_from_faces
)
from functions.shape_extraction import (
    extract_faces, convert_whole_shape_to_nurbs, extract_geometrical_surface, extract_loops_from_face
)
from functions.normalize import compressedJson

# OCC imports
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.ShapeBuild import ShapeBuild_ReShape
from OCC.Core.BRepLib import breplib


# ─────────────────────────────────────────────────────────────────────────────
# Timeout helper (SIGALRM — Linux only)
# ─────────────────────────────────────────────────────────────────────────────

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Processing exceeded time limit.")


# ─────────────────────────────────────────────────────────────────────────────
# Original single-file helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def replace_elliptical_edges_with_bspline(shape: TopoDS_Shape) -> TopoDS_Shape:
    reshaper = ShapeBuild_ReShape()
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        try:
            h_curve, first, last = BRep_Tool.Curve(edge)
        except:
            h_curve = None
        if h_curve is None:
            explorer.Next()
            continue
        curve = h_curve.DynamicType().Name()
        if curve == "Geom_Ellipse":
            try:
                bspline = geomconvert.CurveToBSplineCurve(h_curve)
                new_edge = BRepBuilderAPI_MakeEdge(bspline, first, last).Edge()
                reshaper.Replace(edge, new_edge)
            except Exception:
                pass
        explorer.Next()
    new_shape = reshaper.Apply(shape)
    breplib.BuildCurves3d(new_shape)
    return new_shape


def shape_has_conic_edges(shape):
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        try:
            curve_handle, first, last = BRep_Tool.Curve(edge)
        except:
            curve_handle = None
        if curve_handle is None:
            continue
        if curve_handle:
            curve = curve_handle.DynamicType().Name()
            if curve in ["Geom_Ellipse", "Geom_Parabola", "Geom_Hyperbola"]:
                return True
        explorer.Next()
    return False


def load_original_shape(step_file, logger):
    shape = read_step_file(step_file, logger)
    faces = extract_faces(shape)
    if len(faces) <= 0:
        logger.error("No faces found in original shape.")
        return None, []
    return shape, faces


def convert_to_nurbs(original_shape, logger):
    nurbs_shape = convert_whole_shape_to_nurbs(original_shape)
    if nurbs_shape is None:
        logger.error("Conversion to NURBS failed.")
        return None
    faces = extract_faces(nurbs_shape)
    if len(faces) <= 0:
        logger.error("No faces found in NURBS shape.")
        return None
    logger.info(f"Converted {len(faces)} faces to NURBS.")
    return faces


def extract_surface_parameters(faces):
    extracted_params = {}
    extracted_params_json = {}
    for i, face in enumerate(faces):
        extracted_params[f"face_{i}"] = extract_geometrical_surface(face)
        extracted_params_json[f"face_{i}"] = extracted_params[f"face_{i}"].to_json()
    return extracted_params, extracted_params_json


def reconstruct_surfaces(extracted_params):
    return [reconstructBSplineSurface(params) for params in extracted_params.values()]


def evaluate_faces(original_faces, reconstructed_faces, logger, threshold=1.0):
    error_indices = []
    for i in range(len(reconstructed_faces)):
        error = compare_faces(
            reconstruct_from_faces([original_faces[i]]),
            reconstruct_from_faces([reconstructed_faces[i]]),
            logger=logger,
            samples=8192,
            face_index=i,
            threshold=threshold,
        )
        if error > threshold:
            error_indices.append(i)
    return error_indices


def handle_failed_faces(error_indices, original_faces, reconstructed_faces, extracted_params_json, logger):
    for i in error_indices:
        try:
            params = extract_loops_from_face(original_faces[i])
            try:
                recon_face = build_face_from_loops(params, i)
            except Exception:
                continue
            if recon_face is None or recon_face.IsNull():
                logger.warning(f"Reconstructed face is null at index {i}")
                continue
            try:
                reconstruct_from_faces([recon_face])
            except Exception:
                logger.error(f"Reconstructed face is not valid at index {i}")
                continue
            extracted_params_json[f"face_{i}"] = params
            reconstructed_faces[i] = recon_face
        except Exception as e:
            logger.error(f"Error reconstructing face at index {i}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Core single-file extraction (runs inside a worker process)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_single(step_file: Path, output_json: Path, threshold: float, timeout: int) -> str:
    """
    Extract NURBS parameters from one STEP file and write to output_json.
    Returns a status string: "ok" | "skipped" | "timeout" | "error: <msg>"
    Runs in its own subprocess so any segfault is isolated.
    SIGALRM enforces a per-file timeout (Linux only).
    """
    logger = setup_logger()

    # ── SIGALRM timeout ──────────────────────────────────────────────────────
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        output_json.parent.mkdir(parents=True, exist_ok=True)

        original_shape, original_faces = load_original_shape(str(step_file), logger)
        if original_shape is None:
            return f"error: no faces in {step_file.name}"

        nurbs_faces = convert_to_nurbs(original_shape, logger)
        if nurbs_faces is None:
            return f"error: NURBS conversion failed for {step_file.name}"

        extracted_params, extracted_params_json = extract_surface_parameters(nurbs_faces)
        reconstructed_faces = reconstruct_surfaces(extracted_params)

        error_indices = evaluate_faces(original_faces, reconstructed_faces, logger, threshold)
        handle_failed_faces(error_indices, original_faces, reconstructed_faces, extracted_params_json, logger)

        extracted_params_json = compressedJson(extracted_params_json)

        with open(output_json, "w") as f:
            json.dump(extracted_params_json, f, separators=(",", ":"))

        return "ok"

    except TimeoutError:
        return "timeout"
    except Exception as e:
        return f"error: {e}\n{traceback.format_exc()}"
    finally:
        signal.alarm(0)   # cancel alarm


def _worker(task: tuple) -> tuple[str, str]:
    """
    Unpacks (step_file, output_json, threshold, timeout) and calls _extract_single.
    Returns (step_file_str, status).
    Runs in a subprocess — a segfault here won't kill the main process.
    """
    step_file, output_json, threshold, timeout = task
    status = _extract_single(Path(step_file), Path(output_json), threshold, timeout)
    return step_file, status


# ─────────────────────────────────────────────────────────────────────────────
# Directory-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_dir(args, logger=None):
    if logger is None:
        logger = setup_logger()

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # ── Collect all STEP files (preserving subdirectory structure) ───────────
    step_files = sorted(
        Path(p) for p in glob(str(input_dir / "**" / "*.step"), recursive=True)
        + glob(str(input_dir / "**" / "*.stp"), recursive=True)
    )

    if not step_files:
        logger.error(f"No .step / .stp files found under {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(step_files)} STEP file(s) under {input_dir}")

    # ── Build task list ──────────────────────────────────────────────────────
    tasks = []
    skipped = []
    for step_file in step_files:
        # Mirror input directory structure under output_dir
        relative    = step_file.relative_to(input_dir)
        output_json = (output_dir / relative).with_suffix(".json")

        if output_json.exists() and not args.overwrite:
            skipped.append(str(step_file))
            continue

        tasks.append((str(step_file), str(output_json), args.threshold, args.timeout))

    if skipped:
        logger.info(f"Skipping {len(skipped)} already-processed file(s) (use --overwrite to reprocess).")

    if not tasks:
        logger.info("Nothing to process.")
        return

    logger.info(f"Processing {len(tasks)} file(s) with {args.num_workers} worker(s)...")

    # ── Parallel processing via multiprocessing (segfault-safe) ─────────────
    # Each task runs in its own subprocess. A segfault in a worker is caught
    # by multiprocessing and reported as a non-zero exit code.
    ok_count      = 0
    timeout_count = 0
    error_count   = 0
    failed_files  = []

    # 'spawn' context avoids OCC fork-safety issues
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=args.num_workers) as pool:
        for step_file_str, status in pool.imap_unordered(_worker, tasks):
            name = Path(step_file_str).name
            if status == "ok":
                ok_count += 1
                logger.info(f"[OK]      {name}")
            elif status == "timeout":
                timeout_count += 1
                failed_files.append(step_file_str)
                logger.warning(f"[TIMEOUT] {name} — exceeded {args.timeout}s")
            elif status.startswith("error"):
                error_count += 1
                failed_files.append(step_file_str)
                logger.error(f"[ERROR]   {name} — {status}")
            else:
                # Worker process crashed (segfault → non-zero exit)
                error_count += 1
                failed_files.append(step_file_str)
                logger.error(f"[CRASH]   {name} — worker exited unexpectedly")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(tasks)
    logger.info("─" * 60)
    logger.info(f"Done.  Total: {total}  |  OK: {ok_count}  |  "
                f"Timeout: {timeout_count}  |  Error/Crash: {error_count}  |  "
                f"Skipped: {len(skipped)}")

    if failed_files:
        failed_log = output_dir / "failed_files.txt"
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_log, "w") as f:
            f.write("\n".join(failed_files))
        logger.info(f"Failed file paths saved to: {failed_log}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-file entry point (original behaviour, preserved)
# ─────────────────────────────────────────────────────────────────────────────

def extract(args, logger=None):
    if logger is None:
        logger = setup_logger()
    logger.info("Starting single-file extraction...")

    step_file   = Path(args.step_file)
    output_json = Path(args.output_json)

    status = _extract_single(step_file, output_json, args.threshold, args.timeout)

    if status == "ok":
        logger.info(f"Saved: {output_json}")
    elif status == "timeout":
        logger.error(f"Timed out after {args.timeout}s.")
        sys.exit(1)
    else:
        logger.error(status)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and reconstruct NURBS surfaces from STEP file(s)."
    )

    # ── Input mode (mutually exclusive) ──────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--step_file", type=str,
        help="Path to a single STEP file."
    )
    input_group.add_argument(
        "--input_dir", type=str,
        help="Root directory to search recursively for .step / .stp files."
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_json", type=str, default="extracted_params.json",
        help="Output JSON path (single-file mode only)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/json",
        help="Root output directory (directory mode). Mirrors input structure."
    )

    # ── Processing options ────────────────────────────────────────────────────
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Chamfer distance threshold for face evaluation (default: 0.6)."
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Per-file timeout in seconds (default: 60)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=max(1, os.cpu_count() - 1),
        help="Number of parallel worker processes (default: cpu_count - 1)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess files even if the output JSON already exists."
    )

    args = parser.parse_args()
    logger = setup_logger()

    if args.input_dir:
        extract_dir(args, logger)
    else:
        extract(args, logger)