import os
import numpy as np
import torch
import trimesh
import pandas as pd
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process
from scipy.spatial import cKDTree
import point_cloud_utils as pcu
from kaolin.metrics.pointcloud import chamfer_distance, f_score
import StructuralLosses
import argparse
import gc

# =====================================================
# Utility
# =====================================================

def scale_pc(pc):
    """Center and scale point cloud to unit cube [-0.5, 0.5]."""
    min_bound = pc.min(axis=0)
    max_bound = pc.max(axis=0)
    center = (min_bound + max_bound) / 2
    scale = (max_bound - min_bound).max()
    return (pc - center) / scale


def safe_process(target, timeout=60, **kwargs):
    with Manager() as manager:
        return_dict = manager.dict()

        def wrapper(return_dict, **kwargs):
            return_dict["result"] = target(**kwargs)

        process = Process(target=wrapper, args=(return_dict,), kwargs=kwargs)
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return None
        return return_dict.get("result", None)


# =====================================================
# CPU METRICS
# =====================================================

def normal_consistency(pred_points, pred_normals, gt_points, gt_normals, symmetric=True):
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)

    dists, idx = tree_gt.query(pred_points, k=1)
    matched_gt_normals = gt_normals[idx]
    cos_sim_pred = np.abs(
        np.sum(pred_normals * matched_gt_normals, axis=1)
        / (np.linalg.norm(pred_normals, axis=1) * np.linalg.norm(matched_gt_normals, axis=1) + 1e-8)
    )

    if not symmetric:
        return cos_sim_pred.mean()

    dists, idx = tree_pred.query(gt_points, k=1)
    matched_pred_normals = pred_normals[idx]
    cos_sim_gt = np.abs(
        np.sum(gt_normals * matched_pred_normals, axis=1)
        / (np.linalg.norm(gt_normals, axis=1) * np.linalg.norm(matched_pred_normals, axis=1) + 1e-8)
    )

    return 0.5 * (cos_sim_pred.mean() + cos_sim_gt.mean())


def get_jsd(pt1, pt2):
    if len(pt1.shape) == 2 and len(pt2.shape) == 2:
        pt1 = pt1[np.newaxis, ...]
        pt2 = pt2[np.newaxis, ...]
    return StructuralLosses.jsd_between_point_cloud_sets(pt1, pt2)

def sample_points_from_mesh(mesh_path, num_points=8192):
    try:
        pred_mesh = safe_process(trimesh.load, timeout=30, file_obj=mesh_path, process=False, force="mesh")
        if pred_mesh is None or not isinstance(pred_mesh, trimesh.Trimesh):
            return None
        pred_mesh.fix_normals()

        pred_points, face_indices = trimesh.sample.sample_surface(pred_mesh, 8192)
        pred_normals = pred_mesh.face_normals[face_indices]
        pred_points = scale_pc(pred_points).astype(np.float32)
        del pred_mesh
        gc.collect()
        return pred_points, pred_normals
    except Exception as e:
        print(f"Error sampling points from {mesh_path}: {e}")
        return None, None


def process_cpu_metrics(step_path, pred_step_dir, gt_step_dir, dataset_type):
    """Compute CPU-bound metrics."""
    try:
        pred_points, pred_normals = sample_points_from_mesh(step_path, num_points=8192)
        gt_step_path = os.path.join(gt_step_dir, os.path.relpath(step_path, pred_step_dir)).replace("NURBS", "STEP")
        
        uid = os.path.relpath(step_path, pred_step_dir).strip(".step")
        gt_points, gt_normals = sample_points_from_mesh(gt_step_path, num_points=8192)
        
        if pred_points is None:
            print(f"Skipping {step_path} due to loading/sampling error.")
            return None
        if gt_points is None:
            print(f"Skipping {step_path} due to missing GT file: {gt_step_path}")
            return None

        nc = normal_consistency(pred_points, pred_normals, gt_points, gt_normals)
        jsd = get_jsd(pred_points, gt_points) * 1e2
        hd = pcu.hausdorff_distance(gt_points, pred_points)
        
        del pred_normals, gt_normals
        gc.collect()
        return {
            "uid": uid,
            "pred_points": pred_points,
            "gt_points": gt_points,
            "nc": nc,
            "jsd": jsd,
            "hd": hd
        }

    except Exception as e:
        print(f"CPU metric error on {step_path}: {e}")
        return None


# =====================================================
# GPU METRICS - CD & F1
# =====================================================

def compute_cd_f1(batch, device):
    """Compute Chamfer Distance and F1 in batches."""
    pred_tensors = torch.from_numpy(np.stack([x["pred_points"] for x in batch])).to(device)
    gt_tensors = torch.from_numpy(np.stack([x["gt_points"] for x in batch])).to(device)

    cd = chamfer_distance(gt_tensors, pred_tensors) * 1e3
    f1 = f_score(gt_tensors, pred_tensors, 0.02) * 1e2

    for i, x in enumerate(batch):
        x["cd"] = cd[i].item() if cd.ndim > 0 else cd.item()
        x["f1"] = f1[i].item() if f1.ndim > 0 else f1.item()
    return batch


# =====================================================
# GPU METRICS - MMD
# =====================================================

def get_mmd(pt1, pt2):
    if len(pt1.shape) == 2 and len(pt2.shape) == 2:
        pt1 = pt1.unsqueeze(0)
        pt2 = pt2.unsqueeze(0)
    bs = pt1.shape[0]
    results = StructuralLosses.compute_all_metrics(pt1, pt2, bs)
    return results['lgan_mmd-CD'].item()


def compute_mmd(batch, device):
    """Compute MMD separately, since it's slow."""
    pred_tensors = torch.from_numpy(np.stack([x["pred_points"] for x in batch])).to(device)
    gt_tensors = torch.from_numpy(np.stack([x["gt_points"] for x in batch])).to(device)
    mmd = get_mmd(pred_tensors, gt_tensors) * 1e3
    for x in batch:
        x["mmd"] = mmd
    return batch


# =====================================================
# MAIN
# =====================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_step_dir", type=str, required=True, help="Directory containing predicted STEP files")
    ap.add_argument("--gt_step_dir", type=str, required=True, help="Directory containing ground truth STEP files")
    ap.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save metric CSVs")
    ap.add_argument("--dataset_type", type=str, default="abc", choices=["abc"])
    ap.add_argument("--method", type=str, default="nurbgen")
    ap.add_argument("--max_workers", type=int, default=32)
    ap.add_argument("--batch_size_cd", type=int, default=512)
    ap.add_argument("--batch_size_mmd", type=int, default=4)
    ap.add_argument("--extension", type=str, default="step")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_steps = sorted(glob(os.path.join(args.pred_step_dir, "**", f"*.{args.extension}"), recursive=True))
    print(f"Found {len(all_steps)} files in {args.pred_step_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------- PART 1: CPU METRICS -----------
    cpu_results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_cpu_metrics, step, args.pred_step_dir, args.gt_step_dir, args.dataset_type)
            for step in all_steps
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="CPU Metrics"):
            res = f.result()
            if res is not None:
                cpu_results.append(res)

    df_cpu = pd.DataFrame([{k: v for k, v in r.items() if k not in ["pred_points", "gt_points"]} for r in cpu_results])
    cpu_csv_path = os.path.join(args.output_dir, args.method, args.dataset_type, "metrics_cpu.csv")
    os.makedirs(os.path.dirname(cpu_csv_path), exist_ok=True)
    df_cpu.to_csv(cpu_csv_path, index=False)
    print(f"✅ Saved CPU metrics: {cpu_csv_path}")

    # ----------- PART 2: GPU METRICS (CD + F1) -----------
    cd_f1_results = []
    for i in tqdm(range(0, len(cpu_results), args.batch_size_cd), desc="GPU Metrics (CD/F1)"):
        batch = cpu_results[i:i + args.batch_size_cd]
        processed = compute_cd_f1(batch, device)
        cd_f1_results.extend(processed)

    df_cd_f1 = pd.DataFrame([{k: v for k, v in r.items() if k not in ["pred_points", "gt_points"]} for r in cd_f1_results])
    gpu_cd_path = os.path.join(args.output_dir, args.method, args.dataset_type, "metrics_gpu_cd_f1.csv")
    df_cd_f1.to_csv(gpu_cd_path, index=False)
    print(f"✅ Saved GPU CD/F1 metrics: {gpu_cd_path}")
    
    # Print numerical summaries
    print("\n=== Numerical Summaries ===")
    print("CPU Metrics:")
    print(df_cpu.mean(numeric_only=True).round(2).to_string())
    print("\nGPU CD/F1 Metrics:")
    print(df_cd_f1.mean(numeric_only=True).round(2).to_string())

    # ----------- PART 3: GPU METRICS (MMD) -----------
    mmd_results = []
    for i in tqdm(range(0, len(cpu_results), args.batch_size_mmd), desc="GPU Metrics (MMD)"):
        batch = cpu_results[i:i + args.batch_size_mmd]
        processed = compute_mmd(batch, device)
        mmd_results.extend(processed)

    df_mmd = pd.DataFrame([{k: v for k, v in r.items() if k not in ["pred_points", "gt_points"]} for r in mmd_results])
    gpu_mmd_path = os.path.join(args.output_dir, args.method, args.dataset_type, "metrics_gpu_mmd.csv")
    df_mmd.to_csv(gpu_mmd_path, index=False)
    print(f"✅ Saved GPU MMD metrics: {gpu_mmd_path}")
    
    # Print numerical summaries
    print("\n=== Numerical Summaries ===")
    print("GPU MMD Metrics:")
    print(df_mmd.mean(numeric_only=True).round(2).to_string())


if __name__ == "__main__":
    main()
