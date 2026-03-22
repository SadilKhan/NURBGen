
# ----------------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------------
import os
import sys
import json
import logging
import numpy as np

# ----------------------------------------------------------------------------
# Compress and Normalize Json
# ----------------------------------------------------------------------------

def compress_orientation(data):
    def recurse(obj):
        if isinstance(obj, dict):
            return {k: ("R" if v == "REVERSED" else "F") if k == "orientation" else recurse(v)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        else:
            return obj
    return recurse(data)


def round_floats_except_poles(data, decimal_places=6):
    def recurse(obj , skip_key=None):
        if isinstance(obj, dict):
            return {k: recurse(v, skip_key="poles" if k == "poles" else skip_key)
                for k, v in obj.items()}
        elif isinstance(obj, list):
            if skip_key == "poles":
                return obj  # Leave poles as-is
            return [recurse(v, skip_key) for v in obj]
        elif isinstance(obj, np.ndarray):
            return recurse(obj.tolist())  # convert array to list and recurse
        elif isinstance(obj, (float, np.floating, int, np.integer)):
            return round(obj, decimal_places)
        else:
            return obj  # leave strings, bools, None, etc.

    return recurse(data)


def remove_bbox_area(data):
    def recurse(obj):
        if isinstance(obj, dict):
            return {
                k: recurse(v)
                for k, v in obj.items()
                if k != "bbox_area"  # 🚫 remove this field
            }
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        else:
            return obj
    return recurse(data)

def compress_uniform_weights(data):
    def recurse(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == "weights" and isinstance(v, list):
                    # Only compress if all rows are uniform
                    if all(isinstance(row, list) and row and all(x == row[0] for x in row) for row in v):
                        compressed_rows = [[round(row[0], 3), len(row)] for row in v]
                        new_obj[k] = compressed_rows
                    else:
                        print(f"⚠️ Skipping 'weights' field due to non-uniform rows.")
                        # SKIP weights completely
                        continue
                else:
                    new_obj[k] = recurse(v)
            return new_obj
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        else:
            return obj
    return recurse(data)


def encode_poles(poles):
    """
    Subtracts the first pole vector from all subsequent pole vectors.
    The first vector is preserved.
    """
    poles = np.array(poles)
    base = poles[0]
    diffs = [base]  # Keep the first vector
    for i in range(1, len(poles)):
        diffs.append(poles[i] - base)
    return np.array(diffs).tolist()


def encode_poles_2_step(poles):
    """
    Encodes the poles using two-step relative encoding:
    1. Local difference encoding within each sub-array.
    2. Global difference encoding from the first sub-array.
    """
    poles = np.array(poles)
    
    # Step 1: local encoding within each group
    local_encoded = []
    for group in poles:
        base = group[0]
        encoded_group = [base]
        for i in range(1, len(group)):
            encoded_group.append(base - group[i])
        local_encoded.append(encoded_group)

    local_encoded = np.array(local_encoded)

    # Step 2: global encoding using the first group
    global_encoded = [local_encoded[0]]  # first group stays the same
    for group in local_encoded[1:]:
        encoded_group = [local_encoded[0][i] - group[i] for i in range(len(group))]
        global_encoded.append(encoded_group)

    return np.array(global_encoded).tolist()

# ---------------------------------------------------------------------------------
# Decompress and Normalize Json
# ---------------------------------------------------------------------------------
def decode_poles(encoded_poles):
    """
    Reconstructs the original poles from the encoded representation.
    """
    encoded_poles = np.array(encoded_poles)
    base = encoded_poles[0]
    original = [base]
    for i in range(1, len(encoded_poles)):
        original.append(base + encoded_poles[i])
    return np.array(original).tolist()
def decode_poles_2_step(encoded_poles):
    """
    Decodes the poles from the two-step relative encoding:
    1. Undo global difference (reverse of Step 2).
    2. Undo local difference (reverse of Step 1).
    """
    encoded = np.array(encoded_poles)

    # Step 1: undo global encoding
    decoded_global = [encoded[0]]
    for group in encoded[1:]:
        decoded_group = [encoded[0][i] - group[i] for i in range(len(group))]
        decoded_global.append(decoded_group)

    decoded_global = np.array(decoded_global)

    # Step 2: undo local encoding
    decoded = []
    for group in decoded_global:
        base = group[0]
        restored_group = [base]
        for i in range(1, len(group)):
            restored_group.append(base - group[i])
        decoded.append(restored_group)

    return np.array(decoded).tolist()


def decompress_orientation(data):
    def recurse(obj):
        if isinstance(obj, dict):
            return {k: ("REVERSED" if v == "R" else "FORWARD") if k == "orientation" else recurse(v)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        else:
            return obj
    return recurse(data)

def decompress_uniform_weights(data):
    def recurse(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == "weights" and isinstance(v, dict) and "val" in v and "shape" in v:
                    rows, cols = v["shape"]
                    new_obj[k] = [[v["val"]] * cols for _ in range(rows)]
                else:
                    new_obj[k] = recurse(v)
            return new_obj
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        else:
            return obj
    return recurse(data)

def decompress_rowwise_weights(data):
    def recurse(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == "weights" and isinstance(v, list) and all(isinstance(r, list) and len(r) == 2 for r in v):
                    new_obj[k] = [[val] * count for val, count in v]
                else:
                    new_obj[k] = recurse(v)
            return new_obj
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        else:
            return obj
    return recurse(data)

def remove_empty_weights(data):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key == "weights" and (value is None or value == []):
                continue
            new_data[key] = remove_empty_weights(value)
        return new_data
    elif isinstance(data, list):
        return [remove_empty_weights(item) for item in data]
    else:
        return data


def compressedJson(data, min_val=-1000.0, max_val=1000.0):
    import copy
    data = copy.deepcopy(data)  # Avoid mutating original input

    def encode_surface_poles(surface):
        """Handles encoding poles in both dict and list-based surfaces."""
        if isinstance(surface, dict):
            poles = surface.get("poles")
            if poles is not None:
                surface["poles"] = encode_poles_2_step(poles)

        elif isinstance(surface, list):
            for sub in surface:
                if not isinstance(sub, dict):
                    continue
                for edge in sub.get("edges", []):
                    poles = edge.get("poles")
                    if poles is not None:
                        edge["poles"] = encode_poles(poles)

        else:
            print(f"⛔ Unsupported surface structure: {surface}")

    # Step 1: Encode poles safely
    for surf_id in list(data.keys()):
        encode_surface_poles(data[surf_id])

    # Step 2: Apply other transformations
    data = round_floats_except_poles(data, decimal_places=6)
    data = compress_orientation(data)
    data = compress_uniform_weights(data)
    data = remove_bbox_area(data)
    data = fix_negative_zeros_nested(data)
    data = remove_empty_weights(data)

    return data

def decompressedJson(data, min_val=-1000.0, max_val=1000.0):
   

    # Step 1: Decompress orientation
    data = decompress_orientation(data)

    # Step 2: Decompress uniform weights (both formats supported)
    data = decompress_rowwise_weights(data)

    # Step 3: Dequantize poles
    def decode_surface_poles(surface):
        if isinstance(surface, dict):
            if "poles" in surface:
                surface["poles"] = decode_poles_2_step(surface["poles"])

        elif isinstance(surface, list):
            for i, sub in enumerate(surface):
                if not isinstance(sub, dict):
                    continue
                for j, edge in enumerate(sub.get("edges", [])):
                    if "poles" in edge:
                        edge["poles"] = decode_poles(edge["poles"])
        else:
            print(f"⛔ Surface  has unsupported structure. Skipping.")

    for surf_id in list(data.keys()):
        decode_surface_poles(data[surf_id])
    return data 

# ---------------------------------------------------------------------------------
# Evaluate JSON
# ---------------------------------------------------------------------------------
import numpy as np

def calculate_dequantization_error(original_poles, dequantized_poles, min_val=-1000.0, max_val=1000.0):
    """
    Compute mean absolute error normalized to the value range.
    Returns relative error in % (0–100 range is ideal).
    """
    original = np.array(original_poles, dtype=np.float64).reshape(-1, 3)
    dequantized = np.array(dequantized_poles, dtype=np.float64).reshape(-1, 3)

    if original.shape != dequantized.shape:
        raise ValueError("Shape mismatch between original and dequantized data.")

    range_val = max_val - min_val
    abs_error = np.abs(original - dequantized) / range_val  # Normalize to range
    mean_error = np.mean(abs_error, axis=0) * 100  # As %

    return {
        "x": round(mean_error[0], 4),
        "y": round(mean_error[1], 4),
        "z": round(mean_error[2], 4),
        "overall": round(np.mean(mean_error), 4)
    }


def fix_negative_zeros_nested(obj):
    if isinstance(obj, dict):
        return {k: fix_negative_zeros_nested(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_negative_zeros_nested(item) for item in obj]
    elif isinstance(obj, float):
        return 0.0 if obj == -0.0 else obj
    else:
        return obj


def main_compare_error(original_path, decompressed_path):
    import numpy as np

    with open(original_path, 'r') as f:
        original_data = json.load(f)
    with open(decompressed_path, 'r') as f:
        decompressed_data = json.load(f)

    print("🔍 Surface-wise Dequantization Error (%):\n")

    total_error = []
    for surf_id in original_data:
        orig_surface = original_data[surf_id]
        decomp_surface = decompressed_data.get(surf_id)
        if not decomp_surface:
            continue

        all_orig, all_decomp = [], []

        if isinstance(orig_surface, dict) and "poles" in orig_surface:
            all_orig.extend(np.array(orig_surface["poles"]).reshape(-1, 3).tolist())
            all_decomp.extend(np.array(decomp_surface["poles"]).reshape(-1, 3).tolist())

        elif isinstance(orig_surface, list):
            for orig_sub, decomp_sub in zip(orig_surface, decomp_surface):
                for orig_edge, decomp_edge in zip(orig_sub.get("edges", []), decomp_sub.get("edges", [])):
                    if "poles" in orig_edge and "poles" in decomp_edge:
                        all_orig.extend(np.array(orig_edge["poles"]).reshape(-1, 3).tolist())
                        all_decomp.extend(np.array(decomp_edge["poles"]).reshape(-1, 3).tolist())

        if all_orig and all_decomp:
            try:
                error = calculate_dequantization_error(all_orig, all_decomp)
                overall = float(error["overall"])
                total_error.append(overall)
                print(f"Surface {surf_id}: {overall:.4f}%")
            except Exception as e:
                print(f"⚠️ Surface {surf_id} error: {e}")

    if total_error:
        average_error = sum(total_error) / len(total_error)
        print(f"\n🔢 Total Average Error Across Surfaces: {average_error:.4f}%")
    else:
        print("⚠️ No surfaces processed.")

    print("\n✅ Finished error comparison.")



