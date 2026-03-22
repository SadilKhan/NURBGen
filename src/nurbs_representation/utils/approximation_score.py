
import numpy as np
import torch
import trimesh
from pathlib import Path

from OCC.Core.TopoDS import topods
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend.TopologyUtils import TopologyExplorer
from scipy.spatial import cKDTree

# -------------------------
# Chamfer Distance (PyTorch)
# -------------------------
def chamfer_distance(p1, p2):
    """
    Compute Chamfer Distance between two point clouds.
    p1, p2: [B, N, 3] and [B, M, 3]
    Returns: scalar chamfer loss
    """
    B, N, _ = p1.shape
    _, M, _ = p2.shape

    p1_expand = p1.unsqueeze(2).expand(B, N, M, 3)
    p2_expand = p2.unsqueeze(1).expand(B, N, M, 3)

    dist = torch.norm(p1_expand - p2_expand, dim=3)

    min_dist_p1, _ = dist.min(dim=2)  # [B, N]
    min_dist_p2, _ = dist.min(dim=1)  # [B, M]

    loss = min_dist_p1.mean(dim=1) + min_dist_p2.mean(dim=1)  # [B]
    return loss.mean()  # scalar


# -------------------------
# Extract mesh from face
# -------------------------
def extract_mesh_from_face(face):
    face = topods.Face(face)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face, loc)

    if tri is None:
        raise ValueError("Face has no triangulation. Run BRepMesh_IncrementalMesh first.")

    # Extract vertices
    vertices = []
    for i in range(tri.NbNodes()):
        pnt = tri.Node(i + 1).Transformed(loc.Transformation())
        vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
    vertices = np.array(vertices)

    # Extract triangle indices (1-based → 0-based)
    triangles = []
    for i in range(tri.NbTriangles()):
        t = tri.Triangle(i + 1).Get()
        triangles.append([t[0] - 1, t[1] - 1, t[2] - 1])
    triangles = np.array(triangles)

    return vertices, triangles


# -------------------------
# Sample points from triangle mesh
# -------------------------
def sample_surface_points_from_mesh(vertices, triangles, num_samples=2048):
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    return mesh.sample(num_samples)


# -------------------------
# Extract first face from compound or shape
# -------------------------
def extract_first_face(shape):
    for face in TopologyExplorer(shape).faces():
        return topods.Face(face)
    raise ValueError("No faces found in shape.")


# -------------------------
# Save point clouds (optional)
# -------------------------
def save_point_cloud(points, filename):
    cloud = trimesh.points.PointCloud(points)
    cloud.export(filename)


# -------------------------
# Main: Compare Faces
# -------------------------

def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = cKDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = cKDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def compare_faces(shape1, shape2, logger , precision=0.001, samples=2048, face_index=None, save_dir="output" , threshold=1.0):
    face1 = extract_first_face(shape1)
    face2 = extract_first_face(shape2)

    BRepMesh_IncrementalMesh(face1, precision).Perform()
    BRepMesh_IncrementalMesh(face2, precision).Perform()

    verts1, tris1 = extract_mesh_from_face(face1)
    verts2, tris2 = extract_mesh_from_face(face2)

    pts1 = normalize_pc(sample_surface_points_from_mesh(verts1, tris1, samples))
    pts2 = normalize_pc(sample_surface_points_from_mesh(verts2, tris2, samples))

    # Save for debugging
    if face_index is not None:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        # save_point_cloud(pts1, save_path / f"debug/pointclouds/gt/face_{face_index}_gt.ply")
        # save_point_cloud(pts2, save_path / f"debug/pointclouds/pred/face_{face_index}_pred.ply")

    loss = chamfer_dist(pts1, pts2) * (10**3)

    if loss > threshold:
        logger.warning(f"Chamfer distance for face {face_index} exceeds threshold: {loss:.4f}")
        # save_point_cloud(pts1, save_path / f"debug/pointclouds/gt/face_{face_index}_gt.ply")
        # save_point_cloud(pts2, save_path / f"debug/pointclouds/pred/face_{face_index}_pred.ply")
    return loss.item()


def normalize_pc(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    scale = max_coords - min_coords

    # Avoid division by zero
    scale[scale == 0] = 1.0

    normalized = (points - min_coords) / scale
    return normalized
