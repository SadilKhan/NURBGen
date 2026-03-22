import os
import json
import argparse
from model.Bspline import BsplineParameters
from utils.utils import save_shape_as_stl, save_shape, load_json
from utils.logger import setup_logger
from functions.reconstruct_shape import (
    reconstructBSplineSurface, build_face_from_loops,
    reconstruct_from_faces
)
from functions.normalize import decompressedJson
from glob import glob

def reconstruct_surfaces(extracted_params):
    return [reconstructBSplineSurface(params) for params in extracted_params]


def build_geometric_faces(extract_geometrical_params, logger):
    geometric_faces = []
    for params in extract_geometrical_params:
        try:
            recon_face = build_face_from_loops(params)
            geometric_faces.append(recon_face)
        except Exception as e:
            logger.error(f"Error reconstructing face: {e}")
            geometric_faces.append(None)
    return geometric_faces


def save_final_outputs(shape, step_path, stl_path, logger):
    try:
        save_shape(shape, step_path, logger)
    except Exception as e:
        logger.error(f"Error saving shape as STEP: {e}")
        raise e
    try:
        save_shape_as_stl(shape, stl_path, logger)
    except Exception as e:
        logger.error(f"Error saving shape as STL: {e} but STEP was saved successfully.")

def extract_bspline_and_geometrical_params(extracted_params_json):
    extract_bspline_params = []
    extract_geometrical_params = []
    index_for_geometrical_params = []
    
    # Fix for 'list' object has no attribute 'keys'
    if isinstance(extracted_params_json, list):
        iterable = enumerate(extracted_params_json)
    else:
        iterable = extracted_params_json.items()

    for i, pram in iterable:
        if isinstance(pram, list):
            extract_geometrical_params.append(pram)
            # Handle list index (int) vs dict key (str like "part_0")
            idx = int(i.split("_")[1]) if isinstance(i, str) and "_" in i else int(i)
            index_for_geometrical_params.append(idx)
        else:
            bspline_params = BsplineParameters.from_json(pram)
            extract_bspline_params.append(bspline_params)
    return extract_bspline_params, extract_geometrical_params, index_for_geometrical_params


def reconstruct_from_json(json_path, input_folder, output_dir, logger):
    filename = os.path.relpath(json_path, input_folder)
    output_stl = filename.replace(".json", ".stl")
    output_step = filename.replace(".json", ".step")
    output_stl = os.path.join(output_dir, output_stl)
    output_step = os.path.join(output_dir, output_step)
    os.makedirs(output_dir, exist_ok=True, mode=0o777)

    extracted_params_json = load_json(json_path)
    extracted_params_json = decompressedJson(extracted_params_json)

    extract_bspline_params, extract_geometrical_params, indexes = extract_bspline_and_geometrical_params(extracted_params_json)
    reconstructed_faces = reconstruct_surfaces(extract_bspline_params)
    geometric_faces = build_geometric_faces(extract_geometrical_params, logger)

    # combine reconstructed and geometric faces
    faces = []
    for i in range(len(reconstructed_faces) + len(geometric_faces)):
        if i in indexes:
            faces.append(geometric_faces[indexes.index(i)])
        else:
            faces.append(reconstructed_faces.pop(0))

    reconstructed_shape = reconstruct_from_faces(faces)
    save_final_outputs(reconstructed_shape, output_step, output_stl, logger)


def main(args):
    logger = setup_logger()
    input_folder = args.json_folder

    json_paths = glob(os.path.join(input_folder, "**", "*.json"), recursive=True)

    if not json_paths:
        logger.warning("No JSON files found in the specified folder (recursively).")
        return

    for json_path in json_paths:
        # --- SKIP LOGIC ---
        filename = os.path.splitext(os.path.basename(json_path))[0]
        output_dir = os.path.dirname(json_path)
        output_stl = os.path.join(output_dir, f"{filename}.stl")
        output_step = os.path.join(output_dir, f"{filename}.step")
        
        if os.path.exists(output_stl):
            logger.info(f"Skipping (already exists): {json_path}")
            continue
        # ------------------

        logger.info(f"Processing file: {json_path}")
        try:
            reconstruct_from_json(json_path, input_folder, output_dir, logger)
        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch reconstruct NURBS surfaces from JSON files.")
    parser.add_argument('--json_folder', type=str, required=True, help='Folder path containing JSON parameter files.')
    parser.add_argument('--output_dir', type=str, default="./output", help='Directory to save STL and STEP outputs.')
    args = parser.parse_args()

    main(args)