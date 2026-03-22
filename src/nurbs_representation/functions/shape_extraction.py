# # ----------------------------------------------------------------
# # Imports
# # ----------------------------------------------------------------

# System imports
import logging
from utils.logger import setup_logger
logger = setup_logger()
import json

# OCC imports
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_NurbsConvert,
)
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.Geom import Geom_Circle
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

# New Imports
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer


from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import (
    Geom_Circle, Geom_BSplineCurve, Geom_Line, Geom_Ellipse,
    Geom_BezierCurve, Geom_TrimmedCurve, Geom_OffsetCurve,
    Geom_Hyperbola, Geom_Parabola
)
from OCCUtils.Topology import Topo
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_INTERNAL, TopAbs_EXTERNAL
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

# local imports
from model.Bspline import BsplineParameters

# ----------------------------------------------------------------
#  Extract Faces
# ----------------------------------------------------------------

def extract_faces(nurbs_shape):
    faces = []
    # Explore all faces in the NURBS shape
    try:
        explorer = TopExp_Explorer(nurbs_shape, TopAbs_FACE)
        while explorer.More():
            # Use the static method topods.Face instead of the deprecated topods_Face
            face = topods.Face(explorer.Current())
            faces.append(face)
            explorer.Next()
        logger.success('STEP file converted into faces')  
    except Exception as ex:
        logger.error(f'ERROR: failed to split the CAD , (CAD Function extract_faces). MSG: {ex}')
    
    return faces


# ---------------------------------------------------------------------
#      Convert an entire TopoDS_Shape to NURBS (faces+edges) at once
# ---------------------------------------------------------------------
def convert_whole_shape_to_nurbs(original_shape):
    """
    Convert the entire shape to a NURBS representation using
    BRepBuilderAPI_NurbsConvert, then return the new shape.
    """
    try:
        nurbs_converter = BRepBuilderAPI_NurbsConvert(original_shape)
        nurbs_converter.Perform(original_shape)

        logger.success('Convert Shape to Nurbs Representation')
        return nurbs_converter.Shape()
        
    except Exception as ex:
        logger.error('Convert Shape to Nurbs Representation')
        return None





# ------------------------------------------------------------
# Extract Underlying Geometrical surface
# ------------------------------------------------------------

def extract_geometrical_surface(face):
    """
    Extracts the BSpline surface parameters and trimming parameters from the face.
    """
    try:
        # Extract the underlying geometric surface and convert it to a BSpline surface.
        geom_surface = BRep_Tool.Surface(face)
        bspline_surf = geomconvert.SurfaceToBSplineSurface(geom_surface)
        
        # Extract raw arrays and parameters for the BSpline surface:
        poles    = bspline_surf.Poles()         
        u_knots  = bspline_surf.UKnots()          
        v_knots  = bspline_surf.VKnots()          
        u_mults  = bspline_surf.UMultiplicities()   
        v_mults  = bspline_surf.VMultiplicities() 
        u_degree = bspline_surf.UDegree()
        v_degree = bspline_surf.VDegree()
        u_periodic = bspline_surf.IsUPeriodic()
        v_periodic = bspline_surf.IsVPeriodic()

        num_poles_u = bspline_surf.NbUPoles()
        num_poles_v = bspline_surf.NbVPoles()

        is_rational = bspline_surf.IsURational() or bspline_surf.IsVRational()
        weights = []
        if is_rational:
            for iu in range(1, num_poles_u + 1):
                wrow = []
                for iv in range(1, num_poles_v + 1):
                    wrow.append(bspline_surf.Weight(iu, iv))
                weights.append(wrow)

        # Convert control points (poles) to list of tuples.
        poles_list = []
        u_low = poles.LowerRow()
        u_up = poles.UpperRow()
        v_low = poles.LowerCol()
        v_up = poles.UpperCol()
        for i in range(u_low, u_up + 1):
            row = []
            for j in range(v_low, v_up + 1):
                pnt = poles.Value(i, j)
                row.append((pnt.X(), pnt.Y(), pnt.Z()))
            poles_list.append(row)

        # Convert knots and multiplicities to lists.
        u_knots_list = [u_knots.Value(i) for i in range(u_knots.Lower(), u_knots.Upper() + 1)]
        v_knots_list = [v_knots.Value(i) for i in range(v_knots.Lower(), v_knots.Upper() + 1)]
        u_mults_list = [u_mults.Value(i) for i in range(u_mults.Lower(), u_mults.Upper() + 1)]
        v_mults_list = [v_mults.Value(i) for i in range(v_mults.Lower(), v_mults.Upper() + 1)]

        # Return an instance of bsplineParameters containing all the extracted parameters.
        return BsplineParameters(
            poles=poles_list,
            u_knots=u_knots_list,
            v_knots=v_knots_list,
            u_mults=u_mults_list,
            v_mults=v_mults_list,
            u_degree=u_degree,
            v_degree=v_degree,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
            num_poles_u=num_poles_u,
            num_poles_v=num_poles_v,
            weights=weights,
        )
    
    except Exception as ex:
        logger.error(f'ERROR: failed to extract the geometrical surface (CAD Function extract_geometrical_surface). MSG: {ex}')
        raise ValueError(f"Unable to extract the geometrical surface due to: {ex}")



def orientation_to_str(orientation):
    if orientation == TopAbs_FORWARD:
        return "FORWARD"
    elif orientation == TopAbs_REVERSED:
        return "REVERSED"
    elif orientation == TopAbs_INTERNAL:
        return "INTERNAL"
    elif orientation == TopAbs_EXTERNAL:
        return "EXTERNAL"
    else:
        return f"UNKNOWN({orientation})"

def extract_edges_from_wire(wire):
    topo = Topo(wire)
    edge_info_list = []

    for edge in topo.edges():
        try:
            curve_handle, first, last = BRep_Tool.Curve(edge)
        except:
            curve_handle=None
        if curve_handle is None:
            continue

        geom = curve_handle
        kind = geom.DynamicType().Name()
        edge_info = {
            "type": kind,
            "first": first,
            "last": last,
            "orientation": orientation_to_str(edge.Orientation())
        }

        try:
            if kind == "Geom_Circle":
                circle = Geom_Circle.DownCast(geom)
                pos = circle.Position().Location()
                edge_info.update({
                    "center": [pos.X(), pos.Y(), pos.Z()],
                    "radius": circle.Radius(),
                    "normal": [
                        circle.Axis().Direction().X(),
                        circle.Axis().Direction().Y(),
                        circle.Axis().Direction().Z()
                    ]
                })

            elif kind == "Geom_Line":
                line = Geom_Line.DownCast(geom)
                pos = line.Position()
                pnt = pos.Location()
                dir = pos.Direction()
                # edge_info.update({
                #     "point": [pnt.X(), pnt.Y(), pnt.Z()],
                #     "direction": [dir.X(), dir.Y(), dir.Z()]
                # })

                # convert param range to XYZ
                def at(t):
                    return [
                        pnt.X() + dir.X() * t,
                        pnt.Y() + dir.Y() * t,
                        pnt.Z() + dir.Z() * t
                    ]
                p_start = at(first)
                p_end   = at(last)
                if edge_info['orientation'] == "REVERSED":
                    p_start, p_end = p_end, p_start
                edge_info.update({
                    "start": p_start,
                    "end": p_end,
                })

                del edge_info['first'], edge_info['last'], edge_info['orientation']

            elif kind == "Geom_Ellipse":
                ellipse = Geom_Ellipse.DownCast(geom)
                pos = ellipse.Position().Location()
                edge_info.update({
                    "center": [pos.X(), pos.Y(), pos.Z()],
                    "major_radius": ellipse.MajorRadius(),
                    "minor_radius": ellipse.MinorRadius(),
                    "normal": [
                        ellipse.Axis().Direction().X(),
                        ellipse.Axis().Direction().Y(),
                        ellipse.Axis().Direction().Z()
                    ]
                })

            elif kind == "Geom_BSplineCurve":
                bspline = Geom_BSplineCurve.DownCast(geom)
                poles = [[bspline.Pole(i+1).X(),
                          bspline.Pole(i+1).Y(),
                          bspline.Pole(i+1).Z()]
                         for i in range(bspline.NbPoles())]
                weights = [bspline.Weight(i+1)
                           for i in range(bspline.NbPoles())] if bspline.IsRational() else None
                knots = [bspline.Knot(i+1)
                         for i in range(bspline.NbKnots())]
                mults = [bspline.Multiplicity(i+1)
                         for i in range(bspline.NbKnots())]
                edge_info.update({
                    "poles": poles,
                    "degree": bspline.Degree(),
                    "knots": knots,
                    "multiplicities": mults,
                    "weights": weights,
                    "is_periodic": bspline.IsPeriodic()
                })

            elif kind == "Geom_BezierCurve":
                bezier = Geom_BezierCurve.DownCast(geom)
                poles = [[bezier.Pole(i+1).X(),
                          bezier.Pole(i+1).Y(),
                          bezier.Pole(i+1).Z()]
                         for i in range(bezier.NbPoles())]
                weights = [bezier.Weight(i+1)
                           for i in range(bezier.NbPoles())] if bezier.IsRational() else None
                edge_info.update({
                    "poles": poles,
                    "degree": bezier.Degree(),
                    "weights": weights,
                    "is_rational": bezier.IsRational()
                })

            elif kind == "Geom_TrimmedCurve":
                trimmed = Geom_TrimmedCurve.DownCast(geom)
                base = trimmed.BasisCurve()
                edge_info["trimmed_curve_type"] = base.DynamicType().Name()
                edge_info["trim_bounds"] = [
                    trimmed.FirstParameter(),
                    trimmed.LastParameter()
                ]

            elif kind == "Geom_OffsetCurve":
                offset = Geom_OffsetCurve.DownCast(geom)
                base = offset.BasisCurve()
                edge_info["offset_distance"] = offset.Offset()
                edge_info["offset_base_type"] = base.DynamicType().Name()

            elif kind == "Geom_Hyperbola":
                hyperbola = Geom_Hyperbola.DownCast(geom)
                center = hyperbola.Location()
                edge_info.update({
                    "center": [center.X(), center.Y(), center.Z()],
                    "major_radius": hyperbola.MajorRadius(),
                    "minor_radius": hyperbola.MinorRadius()
                })

            elif kind == "Geom_Parabola":
                parabola = Geom_Parabola.DownCast(geom)
                vertex = parabola.Location()
                edge_info.update({
                    "vertex": [vertex.X(), vertex.Y(), vertex.Z()],
                    "focal_length": parabola.Focal()
                })

            else:
                logger.warning(f"Unsupported curve type: {kind}")

        except Exception as e:
            logger.error(f"Error extracting edge info: {e}")

        edge_info_list.append(edge_info)

    return edge_info_list

def extract_loops_from_face(face):
    try:
        wire_loops = []
        wires = list(Topo(face).wires())

        for wire in wires:
            edges = extract_edges_from_wire(wire)
            bbox = Bnd_Box()
            brepbndlib.Add(wire, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            area = (xmax - xmin) * (ymax - ymin)

            wire_loops.append({
                "edges": edges,
                "bbox_area": area
            })

        # Tag outer loop (largest area)
        max_area = max(wire_loops, key=lambda w: w["bbox_area"])["bbox_area"]
        for loop in wire_loops:
            loop["is_outer"] = loop["bbox_area"] == max_area

        return wire_loops
    except Exception as e:
        logger.error(f'ERROR: failed to extract loops from face (CAD Function extract_loops_from_face). MSG: {e}')
        raise ValueError(f"Error extracting loops from face: {e}")
