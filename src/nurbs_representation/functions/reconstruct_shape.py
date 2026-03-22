# ------------------------------------------------------------
# System and Logger Setup
# ------------------------------------------------------------
# System imports
import logging
from utils.logger import setup_logger
logger = setup_logger()
import json

# OCC imports
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace
)
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import (
    TColStd_Array1OfReal,
    TColStd_Array1OfInteger
)
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColStd import TColStd_Array2OfReal
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.Geom import Geom_Circle
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.ShapeFix import ShapeFix_Wire

# New Imports
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface,BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face


from OCC.Core.Geom import (
    Geom_Circle, Geom_BSplineCurve, Geom_Line, Geom_Ellipse,
    Geom_BezierCurve
)

# local imports
from model.Bspline import BsplineParameters

#--------------------------------------------------------------------------------
#  Reconstruction For EACH FACE
#-------------------------------------------------------------------------------

def poles_list_to_array2(poles_list):
    """
    Converts a 2D list of (x, y, z) tuples into a TColgp_Array2OfPnt.
    """
    num_rows = len(poles_list)
    num_cols = len(poles_list[0]) if num_rows > 0 else 0

    # OCC arrays are 1-based
    array = TColgp_Array2OfPnt(1, num_rows, 1, num_cols)

    for i in range(num_rows):
        for j in range(num_cols):
            x, y, z = poles_list[i][j]
            array.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))  # 1-based indexing

    return array

def list_to_array1_of_real(values):
    """
    Converts a list of floats into a TColStd_Array1OfReal.
    """
    n = len(values)
    array = TColStd_Array1OfReal(1, n)  # OCC arrays are 1-based
    for i in range(n):
        array.SetValue(i + 1, values[i])
    return array

def list_to_array1_of_integer(values):
    """
    Converts a list of integers into a TColStd_Array1OfInteger.
    """
    n = len(values)
    array = TColStd_Array1OfInteger(1, n)  # OCC arrays are 1-based
    for i in range(n):
        array.SetValue(i + 1, values[i])
    return array

def reconstructBSplineSurface(extracted_params: BsplineParameters):
    """
    Reconstructs the BSpline surface using the extracted parameters.
    
    Parameters:
        extracted_params: Tuple of parameters (poles, u_knots, v_knots, u_mults, v_mults,
                          u_degree, v_degree, u_periodic, v_periodic).
    
    Returns:
        new_bspline_surface: The rebuilt Geom_BSplineSurface.
    """

    try:
        weights_list = extracted_params.weights

        num_poles_u=extracted_params.num_poles_u
        num_poles_v=extracted_params.num_poles_v

        u_periodic=extracted_params.u_periodic
        v_periodic=extracted_params.v_periodic

        if not isinstance(u_periodic, bool):
            u_periodic = bool(u_periodic)
        if not isinstance(v_periodic, bool):
            v_periodic = bool(v_periodic)

        weights_array = TColStd_Array2OfReal(1, num_poles_u, 1, num_poles_v)

        for iu in range(1, num_poles_u + 1):
            for iv in range(1, num_poles_v + 1):
                try:
                    value = weights_list[iu - 1][iv - 1]
                except IndexError:
                    # print(f"Missing weight at ({iu - 1}, {iv - 1})! Defaulting to 1.0")
                    value = 1.0  # or any fallback
                weights_array.SetValue(iu, iv, value)


        new_bspline_surface = Geom_BSplineSurface(
            poles_list_to_array2(extracted_params.poles),
            weights_array,
            list_to_array1_of_real(extracted_params.u_knots),
            list_to_array1_of_real(extracted_params.v_knots),
            list_to_array1_of_integer(extracted_params.u_mults),
            list_to_array1_of_integer(extracted_params.v_mults),
            extracted_params.u_degree,
            extracted_params.v_degree,
            u_periodic,
            v_periodic
        )
        face_maker = BRepBuilderAPI_MakeFace(new_bspline_surface, 1e-8)  # Tolerance
        if not face_maker.IsDone():
            raise RuntimeError("Failed to create face from BSpline surface")
    
        return face_maker.Face()
    except Exception as ex:
        logger.error(f'ERROR: failed to reconstruct the BSpline surface (CAD Function reconstructBSplineSurface). MSG: {ex}')
        return None


def reconstruct_from_faces(faces):
    try:
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        valid_count = 0
        for i, face in enumerate(faces):
            if face is not None:
                       
                builder.Add(compound, face)
                valid_count += 1
            else:
                logger.warning(f"Face {i} is None or invalid. Skipping.")
        return compound
    except Exception as ex:
        logger.error(f'ERROR: failed to reconstruct the compound shape (CAD Function reconstruct_from_faces). MSG: {ex}')
        raise ValueError(f"unable to construct the face due to: {ex}")



# ---------------------------------------------------------------------
#                        RECONSTRUCTING THE SHAPE
# ---------------------------------------------------------------------


from OCC.Core.TopAbs import TopAbs_REVERSED



def build_face_safely_from_wire(wire):
    from OCC.Core.TopoDS import topods

    face_builder = BRepBuilderAPI_MakeFace(wire, True) 
    if face_builder.IsDone():
        face = face_builder.Face()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Plane:
            return face_builder  
        else:
            pass

    # If not planar or failed, fallback to filling
    face_builder = BRepFill_Filling()
    exp = TopExp_Explorer(wire, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        if isinstance(edge, TopoDS_Edge) or isinstance(edge, TopoDS_Face):
            face_builder.Add(edge, GeomAbs_C0)
        # else:
        #     print(type(edge), isinstance(edge, TopoDS_Edge))
        exp.Next()

    face_builder.Build()
    if not face_builder.IsDone():
        raise ValueError("BRepFill_Filling failed")
    return face_builder



def get_edge_endpoints(edge):
    curve = BRepAdaptor_Curve(edge)
    p1 = curve.Value(curve.FirstParameter())
    p2 = curve.Value(curve.LastParameter())
    return p1, p2



from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.gp import gp_Pnt


def get_edge_endpoints(edge):
    curve = BRepAdaptor_Curve(edge)
    p1 = curve.Value(curve.FirstParameter())
    p2 = curve.Value(curve.LastParameter())
    return p1, p2


def create_closed_wire(edges, gap_tolerance=1e-5, max_closing_edges=10, index=0):
    """
    Build a TopoDS_Wire from a list of edges, force-closing small gaps.
    Adds at most `max_closing_edges` to prevent runaway repairs.
    """
    wire_builder = BRepBuilderAPI_MakeWire()
    for edge in edges:
        wire_builder.Add(edge)

    try:
        return wire_builder.Wire()
    except Exception as e:
        logger.warning(f"Wire build failed: {e}. Attempting forced closure...")

    # Retry with forced closure
    wire_builder = BRepBuilderAPI_MakeWire()
    fixed_edges = []
    prev_end = None
    closing_edge_count = 0

    for i, edge in enumerate(edges):
        start, end = get_edge_endpoints(edge)

        if prev_end:
            gap = prev_end.Distance(start)
            if gap > gap_tolerance:
                if closing_edge_count >= max_closing_edges:
                    raise ValueError("Too many closing edges required. Check input geometry.")
                if gap > 22:
                    raise ValueError(f"Gap {gap:.6f} is too large to force close. Fetching nurbs estimation")
                logger.warning(f"Gap {gap:.6f} between edge {i-1} and {i}. Adding closing edge for {index}.")
                closing_edge = BRepBuilderAPI_MakeEdge(prev_end, start).Edge()
                wire_builder.Add(closing_edge)
                fixed_edges.append(closing_edge)
                closing_edge_count += 1

        wire_builder.Add(edge)
        fixed_edges.append(edge)
        prev_end = end

    # Final closure check
    if fixed_edges:
        first_start, _ = get_edge_endpoints(fixed_edges[0])
        final_gap = prev_end.Distance(first_start)
        if final_gap > gap_tolerance:
            if closing_edge_count >= max_closing_edges:
                raise ValueError("Too many closing edges required (final loop).")
            logger.warning(f"Final gap {final_gap:.6f}. Adding final closing edge for {index}.")
            closing_edge = BRepBuilderAPI_MakeEdge(prev_end, first_start).Edge()
            wire_builder.Add(closing_edge)

    try:
        return wire_builder.Wire()
    except Exception as e:
        # logger.error(f"Failed to build wire after forced closure: {e}")
        raise ValueError("Wire creation failed even after forced closure.")




def build_face_from_loops(loop_data, index=0):
    try:
        outer_wire = None
        inner_wires = []

        for loop in loop_data:
            edges = []
            for item in loop["edges"]:
                curve_type = item["type"]

                if curve_type == "Geom_BSplineCurve":
                    poles_data = item["poles"]
                    poles = TColgp_Array1OfPnt(1, len(poles_data))
                    for i, pt in enumerate(poles_data):
                        poles.SetValue(i + 1, gp_Pnt(*pt))

                    degree = item["degree"]
                    knots = item["knots"]
                    multiplicities = item["multiplicities"]
                    is_periodic = item["is_periodic"]
                    if not isinstance(is_periodic, bool):
                        is_periodic = bool(is_periodic)
                    weights = item.get("weights", None)

                    knots_array = TColStd_Array1OfReal(1, len(knots))
                    for i, k in enumerate(knots):
                        knots_array.SetValue(i + 1, k)

                    mult_array = TColStd_Array1OfInteger(1, len(multiplicities))
                    for i, m in enumerate(multiplicities):
                        mult_array.SetValue(i + 1, m)

                    try:
                        if weights is not None:
                            weights_array=TColStd_Array1OfReal(1, len(weights))
                            for i, w in enumerate(weights):
                                weights_array.SetValue(i + 1, w)
                                try:
                                    bspline = Geom_BSplineCurve(poles, weights_array, knots_array, mult_array, degree, is_periodic)
                                except:
                                    bspline = Geom_BSplineCurve(poles, knots_array, mult_array, degree, is_periodic)
                        else:
                            bspline = Geom_BSplineCurve(poles, knots_array, mult_array, degree, is_periodic)
                    except Exception as e:
                        print(f"Knots ", len(knots), " Multiplicities ", len(multiplicities), " Degree ", degree, " Is Periodic ", is_periodic)
                    edge = BRepBuilderAPI_MakeEdge(bspline, item["first"], item["last"]).Edge()
                    # Handle orientation
                    if item.get("orientation", "").upper() == "REVERSED":
                        edge.Reverse()

                elif curve_type == "Geom_Circle":
                    center = gp_Pnt(*item["center"])
                    normal = gp_Dir(*item["normal"])
                    radius = item["radius"]
                    ax2 = gp_Ax2(center, normal)
                    circle = Geom_Circle(ax2, radius)

                    edge_builder = BRepBuilderAPI_MakeEdge(circle, item["first"], item["last"])
                    edge = edge_builder.Edge()

                    # Handle orientation
                    if item.get("orientation", "").upper() == "REVERSED":
                        edge.Reverse()
                elif curve_type == "Geom_Ellipse":
                    center = gp_Pnt(*item["center"])
                    normal = gp_Dir(*item["normal"])
                    major_radius = item["major_radius"]
                    minor_radius = item["minor_radius"]
                    ax2 = gp_Ax2(center, normal)
                    ellipse = Geom_Ellipse(ax2, major_radius, minor_radius)

                    edge_builder = BRepBuilderAPI_MakeEdge(ellipse, item["first"], item["last"])
                    edge = edge_builder.Edge()

                    # Handle orientation
                    if item.get("orientation", "").upper() == "REVERSED":
                        edge.Reverse()
                elif curve_type == "Geom_BezierCurve":
                    poles_data = item["poles"]
                    poles = TColgp_Array1OfPnt(1, len(poles_data))
                    for i, pt in enumerate(poles_data):
                        poles.SetValue(i + 1, gp_Pnt(*pt))

                    degree = item["degree"]
                    bezier = Geom_BezierCurve(poles, degree)

                    edge = BRepBuilderAPI_MakeEdge(bezier, item["first"], item["last"]).Edge()
                    # Handle orientation
                    if item.get("orientation", "").upper() == "REVERSED":
                        edge.Reverse()


                elif curve_type == "Geom_Line":
                    # point = gp_Pnt(*item["point"])
                    # direction = gp_Dir(*item["direction"])
                    # line = Geom_Line(point, direction)
                    # edge = BRepBuilderAPI_MakeEdge(line, item["first"], item["last"]).Edge()

                    # # Handle orientation
                    # if item.get("orientation", "").upper() == "REVERSED":
                    #     edge.Reverse()

                    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*item["start"]), gp_Pnt(*item["end"])).Edge()

                else:
                    logger.warning(f"Unsupported edge type: {curve_type}")

                edges.append(edge)
            

            wire=create_closed_wire(edges, index=index)
            
            # Fix the wire
            fixer = ShapeFix_Wire()
            fixer.Load(wire)
            fixer.FixClosed()
            fixed_wire = fixer.Wire()

            # Check if the wire is closed
            analyzer = BRepCheck_Analyzer(fixed_wire)
            if not analyzer.IsValid():
                logger.warning("Wire is not valid or not closed.")


            if loop["is_outer"]:
                outer_wire = fixed_wire
            else:
                fixed_wire.Reverse()  # 🌀 
                inner_wires.append(fixed_wire)

        if outer_wire is None:
            logger.error("No outer wire found in loop data.")
            raise ValueError("No outer wire found")

    
        face_builder=build_face_safely_from_wire(outer_wire)

        for inner_wire in inner_wires:
            face_builder.Add(inner_wire)
        
        return face_builder.Face()
    except IndexError:  
        logger.error("Error: Index out of range while processing loop data.")
        raise ValueError("No outer wire found in loop data")
    except Exception as e:
        logger.error(f"Error building face from loops: {e}")
        raise ValueError(f"Error building face from loops: {loop_data} - {e}")


