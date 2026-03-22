
import json
import os
import sys

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer


from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone



from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopoDS import TopoDS_Shape


def read_step_file(step_file, logger):
    try:
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file)
        if status != IFSelect_RetDone:
            raise IOError(f"Failed to read STEP file: {step_file}")
        step_reader.TransferRoots()
        shape = step_reader.OneShape() 
        logger.success('STEP File Loaded')    
        return shape
    except Exception as ex:
        logger.exception(f'Error File Loading the Step File (Utility Function read_step_file). MSG {ex}')
        return 0
    
def save_shape_as_stl(shape, file_path, logger ,linear_deflection=0.001, angular_deflection=0.5, ):
    """
    Save the given shape to an STL file.
    
    This function first computes a mesh for the shape and then saves it as STL.
    
    :param shape: The TopoDS_Shape to be saved.
    :param file_path: The destination file path for the STL file.
    :param linear_deflection: Controls the precision of the meshing (smaller values for higher precision).
    :param angular_deflection: Controls the angular deflection for the meshing.
    """
    try:
        # Check if the shape is already meshed
        # Mesh the shape if not already meshed
        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()
        
        stl_writer = StlAPI_Writer()
        # Optionally, set ASCII mode to False (binary STL) for a smaller file size.
        stl_writer.SetASCIIMode(True)
        stl_writer.Write(shape, file_path)
        logger.success(f'STL File Saved {file_path}')
    except Exception as e:
        logger.error(f'Error saving shape as STL: {e}')
        raise e

def save_point_cloud_as_ply(shape, file_path, logger, linear_deflection=0.001, angular_deflection=0.5):
    """
    Save the mesh vertices of a TopoDS_Shape as a point cloud in PLY format.

    :param shape: The TopoDS_Shape to extract mesh points from.
    :param file_path: The destination file path for the .ply file.
    :param linear_deflection: Meshing precision.
    :param angular_deflection: Angular meshing precision.
    """
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods
        from OCC.Core.TopLoc import TopLoc_Location

        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()

        points_set = set()

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, loc)
            if triangulation:
                for i in range(1, triangulation.NbNodes() + 1):
                    pnt = triangulation.Node(i)
                    points_set.add((pnt.X(), pnt.Y(), pnt.Z()))
            face_explorer.Next()

        # Write valid ASCII .ply
        with open(file_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_set)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for x, y, z in points_set:
                f.write(f"{x} {y} {z}\n")

        logger.success(f'STY File Saved {file_path}')
    except Exception as e:
        logger.error(f'Error saving point cloud as STY: {e}')
        raise e

def save_shape(shape, output_file, logger):
    # Initialize STEP writer
    step_writer = STEPControl_Writer()

    # Transfer the shape
    step_writer.Transfer(shape, STEPControl_AsIs)

    # Write to file
    status = step_writer.Write(output_file)
    if status == IFSelect_RetDone:
        logger.success(f"STEP file written successfully: {output_file}")
    else:
        logger.error(f"Error writing STEP file: {output_file}")

def load_json(file_path):
    """
    Load a JSON file and return its contents.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        dict or list: Parsed JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Normalize shape to [0,1] (unit) or [-1,1] (symmetric) space
def normalize_shape(shape: TopoDS_Shape, mode: str = "unit") -> TopoDS_Shape:
    """Normalize shape to [0,1] (unit) or [-1,1] (symmetric) space."""
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    scale_x = xmax - xmin
    scale_y = ymax - ymin
    scale_z = zmax - zmin
    max_scale = max(scale_x, scale_y, scale_z)

    if max_scale == 0:
        raise ValueError("Degenerate bounding box with zero volume.")

    # Move to origin
    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(gp_Vec(-xmin, -ymin, -zmin))
    translated = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()

    # Scale to unit cube
    trsf_scale = gp_Trsf()
    trsf_scale.SetScaleFactor(1.0 / max_scale)
    scaled = BRepBuilderAPI_Transform(translated, trsf_scale, True).Shape()

    if mode == "symmetric":
        # Shift center from [0,1] to [-1,1]
        trsf_shift = gp_Trsf()
        trsf_shift.SetTranslation(gp_Vec(-0.5, -0.5, -0.5))
        shifted = BRepBuilderAPI_Transform(scaled, trsf_shift, True).Shape()

        trsf_rescale = gp_Trsf()
        trsf_rescale.SetScaleFactor(2.0)
        normalized = BRepBuilderAPI_Transform(shifted, trsf_rescale, True).Shape()
    else:
        normalized = scaled

    return normalized
