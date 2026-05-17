"""
eoio.readers.footprint_utils
=============================

Utilities for normalizing and working with spatial footprint data across readers.
Provides canonical footprint structure with standardized geometry and CRS.

Functions
---------
normalize_footprint : Convert various spatial formats to canonical form
normalize_crs : Convert CRS codes to standard "EPSG:XXXXX" format
wkt_to_geometry : Convert WKT string to Shapely geometry
bbox_to_geometry : Convert bbox list to Shapely Polygon
get_bounds_from_geometry : Extract (minx, miny, maxx, maxy) from geometry
"""

from typing import Optional, Union, Tuple, Dict, Any
import logging

try:
    from shapely.geometry import Polygon, Point, shape
    from shapely.wkt import loads as wkt_loads

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)


def normalize_crs(crs_value: Optional[Union[str, int, Dict]]) -> Optional[str]:
    """
    Normalize various CRS representations to standardized "EPSG:XXXXX" format.

    Parameters
    ----------
    crs_value : str, int, dict, or None
        CRS code in various formats:
        - "EPSG:32630" → "EPSG:32630"
        - "32630" → "EPSG:32630"
        - 32630 → "EPSG:32630"
        - None → None
        - Dict with numeric value → "EPSG:XXXXX"

    Returns
    -------
    str or None
        Normalized CRS string "EPSG:XXXXX" or None if input is None/invalid
    """
    if crs_value is None:
        return None

    # Handle integer
    if isinstance(crs_value, int):
        return f"EPSG:{crs_value}"

    # Handle string
    if isinstance(crs_value, str):
        crs_value = crs_value.strip()
        if crs_value.startswith("EPSG:"):
            return crs_value
        # String of just number
        if crs_value.isdigit():
            return f"EPSG:{crs_value}"
        return crs_value

    # Handle dict (from xmltodict or similar)
    if isinstance(crs_value, dict):
        # Try common keys: 'code', 'value', 'epsg', '#text'
        for key in ["code", "value", "epsg", "#text"]:
            if key in crs_value:
                return normalize_crs(crs_value[key])

    logger.warning(f"Could not normalize CRS value: {crs_value}")
    return None


def wkt_to_geometry(wkt_string: Optional[str]) -> Optional[Union[Polygon, Point]]:
    """
    Convert WKT string to Shapely geometry object.

    Parameters
    ----------
    wkt_string : str or None
        WKT representation of geometry (e.g., "POLYGON(...)" or "POINT(...)")

    Returns
    -------
    Polygon, Point, or None
        Shapely geometry object or None if parsing fails
    """
    if not SHAPELY_AVAILABLE:
        logger.warning("Shapely not available; cannot convert WKT to geometry")
        return None

    if wkt_string is None or not isinstance(wkt_string, str):
        return None

    try:
        return wkt_loads(wkt_string)
    except Exception as e:
        logger.warning(f"Failed to parse WKT string: {wkt_string}, error: {e}")
        return None


def bbox_to_geometry(bbox: Optional[Union[list, tuple]]) -> Optional[Polygon]:
    """
    Convert bounding box [minx, miny, maxx, maxy] to Shapely Polygon.

    Parameters
    ----------
    bbox : list, tuple, or None
        Bounding box as [minx, miny, maxx, maxy] or (minx, miny, maxx, maxy)

    Returns
    -------
    Polygon or None
        Shapely Polygon representing the bounding box, or None if invalid
    """
    if not SHAPELY_AVAILABLE:
        logger.warning("Shapely not available; cannot convert bbox to geometry")
        return None

    if bbox is None:
        return None

    try:
        bbox = list(bbox)
        if len(bbox) != 4:
            logger.warning(f"Bbox must have 4 elements, got {len(bbox)}")
            return None

        minx, miny, maxx, maxy = bbox
        # Create polygon from bbox corners
        return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert bbox to geometry: {bbox}, error: {e}")
        return None


def dict_geometry_to_shapely(geom_dict: Optional[Dict[str, Any]]) -> Optional[Union[Polygon, Point]]:
    """
    Convert dict geometry (from xmltodict or GeoJSON-like) to Shapely object.

    Handles:
    - GeoJSON-like dicts with "coordinates" key
    - xmltodict Polygon dicts with vertex lists

    Parameters
    ----------
    geom_dict : dict or None
        Dictionary with geometry data

    Returns
    -------
    Polygon, Point, or None
        Shapely geometry or None if conversion fails
    """
    if not SHAPELY_AVAILABLE or geom_dict is None:
        return None

    try:
        # Try GeoJSON-like format
        if "coordinates" in geom_dict and "type" in geom_dict:
            return shape(geom_dict)

        # Try xmltodict Polygon with vertex list
        if "Vertex" in geom_dict:
            vertices = geom_dict["Vertex"]
            # Handle single vertex (will be dict) vs multiple (list of dicts)
            if isinstance(vertices, dict):
                vertices = [vertices]

            coords = []
            for v in vertices:
                if isinstance(v, dict):
                    lon = v.get("Lon", v.get("lon"))
                    lat = v.get("Lat", v.get("lat"))
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    lon, lat = v[0], v[1]
                else:
                    continue

                if lon is not None and lat is not None:
                    try:
                        coords.append((float(lon), float(lat)))
                    except (ValueError, TypeError):
                        continue

            if len(coords) >= 3:  # Min for valid polygon
                return Polygon(coords)

        logger.warning(f"Could not parse dict geometry: {geom_dict}")
        return None

    except Exception as e:
        logger.warning(f"Error converting dict geometry to Shapely: {e}")
        return None


def get_bounds_from_geometry(geometry: Optional[Union[Polygon, Point]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract bounds tuple (minx, miny, maxx, maxy) from Shapely geometry.

    Parameters
    ----------
    geometry : Polygon, Point, or None
        Shapely geometry object

    Returns
    -------
    tuple or None
        (minx, miny, maxx, maxy) or None if geometry is None
    """
    if geometry is None:
        return None

    try:
        bounds = geometry.bounds  # Shapely property
        return tuple(bounds)  # (minx, miny, maxx, maxy)
    except Exception as e:
        logger.warning(f"Failed to extract bounds from geometry: {e}")
        return None


def normalize_footprint(
    geometry_input: Optional[Union[str, Dict, list, Polygon, Point]] = None,
    crs_input: Optional[Union[str, int, Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create canonical footprint dictionary from various input formats.

    Converts various geometry and CRS representations to a standardized format.

    Parameters
    ----------
    geometry_input : str, dict, list, Polygon, Point, or None
        Geometry in various formats:
        - WKT string: "POLYGON(...)"
        - Shapely object: Polygon or Point
        - Bbox list: [minx, miny, maxx, maxy]
        - Dict (xmltodict): from XML parsing
    crs_input : str, int, dict, or None
        CRS code in various formats (see normalize_crs for details)

    Returns
    -------
    dict or None
        Canonical footprint dict with keys:
        - 'geometry': Shapely Polygon or Point (or None)
        - 'crs': str "EPSG:XXXXX" (or None)
        - 'bounds': tuple (minx, miny, maxx, maxy) (or None)

        Returns None if no valid geometry or CRS could be extracted.
    """
    if not SHAPELY_AVAILABLE:
        logger.warning("Shapely not available; cannot create footprint")
        return None

    # Convert geometry to Shapely object
    geometry = None
    if geometry_input is not None:
        if isinstance(geometry_input, (Polygon, Point)):
            geometry = geometry_input
        elif isinstance(geometry_input, str):
            geometry = wkt_to_geometry(geometry_input)
        elif isinstance(geometry_input, (list, tuple)):
            geometry = bbox_to_geometry(geometry_input)
        elif isinstance(geometry_input, dict):
            geometry = dict_geometry_to_shapely(geometry_input)
        elif hasattr(geometry_input, "wkt"):
            geometry = wkt_to_geometry(str(geometry_input.wkt))

    # Normalize CRS
    crs = normalize_crs(crs_input)

    # If neither geometry nor CRS, return None
    if geometry is None and crs is None:
        return None

    # Extract bounds from geometry
    bounds = get_bounds_from_geometry(geometry)

    # Build footprint dict
    footprint = {
        "geometry": geometry,
        "crs": crs,
        "bounds": bounds,
    }

    return footprint


if __name__ == "__main__":
    pass
