"""
Generic ROI-based subsetting for *gridded raster* Earth Observation products.

This module provides a small, product-agnostic utility for resolving a
user-specified *region of interest* (ROI) into a form suitable for raster
subsetting.

It is intentionally **not Sentinel-2 specific** and is designed to be reused by
any reader that:

* works with data on a fixed raster grid
* wants to support spatial subsetting via bounding boxes or geometries

Out of scope
------------
* Swath/orbit products with per-pixel latitude/longitude arrays
* Vector-only products

Design principles
-----------------
* Keep the public API small and explicit
* Delay importing heavy geospatial dependencies until needed
* Separate *user intent* (ROI specification) from *reader mechanics*

Typical usage
-------------
::
   roi_subset: ResolvedROISubset = ROISubsetResolver(roi, roi_crs_epsg, image_crs_epsg, image_bounds).run()
    # within image_io.py access:
    roi_subset.clip_box
    roi_subset.geometries
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from eoio.deps import lazy_shapely, lazy_pyproj
from functools import wraps

__all__ = ["ROISubsetResolver", "ResolvedROISubset", "RasterSubsetError"]

# -----------------------------------------------------------------------------------
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
PointBox = Tuple[Point, Union[int, float]]
# -----------------------------------------------------------------------------------


@dataclass
class ResolvedROISubset:
    """
    Resolved spatial subset in the data CRS.

    This is the form consumed by raster IO backends (e.g. rasterio/rioxarray).

    :param clip_box:
        Axis-aligned bounding box ``(xmin, ymin, xmax, ymax)`` in image CRS,
        suitable for windowed reads, or ``None`` if no subsetting is applied

    :param geometries:
        List of GeoJSON geometries in image CRS, suitable for masked reads
        (e.g. ``rioxarray.clip``), or ``None`` if no subsetting is applied

    :param roi:
        Normalised ROI geometry in the *input* CRS

    :param roi_crs_epsg:
        EPSG code of the ROI CRS

    :param image_crs:
        EPSG code of the image CRS
    """

    geometries: Optional[List[Dict[str, Any]]]
    clip_box: Optional[BBox]
    roi: Any
    roi_crs_epsg: int | str
    image_crs: Any
    xy_clip_box: Optional[BBox] = field(default=None)
    tie_clip_box: Optional[BBox] = field(default=None)


class RasterSubsetError(ValueError):
    """
    Raised when ROI specification or resolution fails.

    This error is raised when:

    * an unsupported ROI format is provided
    * an unsupported CRS format is provided
    * requested ROI is not within image bounds
    """


def validate_inputs(func):
    @wraps(func)
    def checker(
        self,
        roi: Any,
        roi_crs_epsg: int | str,
        image_crs_epsg: int | str,
        image_bounds: Any,
        *args,
        **kwargs,
    ):
        # -----------------------------------------------------------------
        if roi_crs_epsg is None:
            raise RasterSubsetError(
                "roi_crs_epsg is required but was None. Provide a valid CRS identifier as an EPSG code "
                "(e.g., 4326 or 'EPSG:4326') to define the coordinate reference system for the ROI."
            )

        if isinstance(roi_crs_epsg, str):
            if not roi_crs_epsg.upper().startswith("EPSG:"):
                raise RasterSubsetError(f"roi_crs_epsg string must start with 'EPSG:', got '{roi_crs_epsg}'.")
        elif isinstance(roi_crs_epsg, int):
            if len(str(roi_crs_epsg)) not in (4, 5):
                raise RasterSubsetError(f"roi_crs_epsg must be a valid integer, got {roi_crs_epsg}.")
        else:
            raise RasterSubsetError(f"roi_crs_epsg must be int or str, got {type(roi_crs_epsg).__name__}.")

        if image_crs_epsg is None:
            raise RasterSubsetError(
                "image_crs_epsg is required but was None. Provide a valid CRS identifier as an EPSG code "
                "(e.g., 4326 or 'EPSG:4326') to define the coordinate reference system for the ROI."
            )

        if isinstance(image_crs_epsg, str):
            if not image_crs_epsg.upper().startswith("EPSG:"):
                raise RasterSubsetError(f"image_crs_epsg string must start with 'EPSG:', got '{image_crs_epsg}'.")
        elif isinstance(image_crs_epsg, int):
            if len(str(image_crs_epsg)) not in (4, 5):
                raise RasterSubsetError(f"image_crs_epsg must be a valid integer, got {image_crs_epsg}.")
        else:
            raise RasterSubsetError(f"image_crs_epsg must be int or str, got {type(image_crs_epsg).__name__}.")

        # -----------------------------------------------------------------
        if roi is None:
            raise RasterSubsetError(
                "ROI is required but was None. Provide a valid spatial subset as one of: "
                "BBox=(xmin, ymin, xmax, ymax), Point=(x, y), or PointBox=((x, y), half_width_m)."
            )

        elif isinstance(roi, tuple):
            if len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi):
                # BBox
                xmin, ymin, xmax, ymax = roi
                if not (xmax > xmin and ymax > ymin):
                    raise RasterSubsetError(f"BBox must have xmax>xmin and ymax>ymin, got {roi}.")
            elif len(roi) == 2 and all(isinstance(v, (int, float)) for v in roi):
                # If it's not PointBox, reject
                raise RasterSubsetError(
                    "Raw Point (x,y) is not supported. Use PointBox=((x,y), half_width_m) or a polygon."
                )
            elif len(roi) == 2 and isinstance(roi[0], tuple) and len(roi[0]) == 2 and isinstance(roi[1], (int, float)):
                # PointBox
                half_width = roi[1]
                if half_width <= 0:
                    raise RasterSubsetError(f"PointBox half-width must be positive, got {half_width}.")

        elif hasattr(roi, "geom_type"):  # Accept: shapely geometry
            pass

        elif isinstance(roi, dict) and "type" in roi:  # Accept: GeoJSON-like dict
            pass

        elif isinstance(roi, list) and roi and isinstance(roi[0], (list, tuple)) and len(roi[0]) == 2:
            pass  # Accept: list-of-coords polygon

        else:
            raise RasterSubsetError(
                "Unsupported ROI format. Supported forms are: "
                "BBox=(xmin,ymin,xmax,ymax), GeoJSON-like dict, list of [x,y] pairs, "
                "or PointBox=((x,y), half_width_m). Raw points (x,y) are not supported."
                f"Instead got {roi}."
            )
        # -----------------------------------------------------------------

        if image_bounds is None:
            pass

        elif (
            isinstance(image_bounds, tuple)
            and len(image_bounds) == 4
            and all(isinstance(v, (int, float)) for v in image_bounds)
        ):
            xmin, ymin, xmax, ymax = image_bounds
            if not (xmax > xmin and ymax > ymin):
                raise RasterSubsetError(f"image_bounds must have xmax>xmin and ymax>ymin, got {image_bounds}.")

        elif hasattr(image_bounds, "geom_type") and hasattr(image_bounds, "bounds"):
            # Accept any shapely geometry; also verify its bounds are non-degenerate
            bx = getattr(image_bounds, "bounds", None)
            if not (isinstance(bx, tuple) and len(bx) == 4):
                raise RasterSubsetError("image_bounds shapely geometry must expose 4-tuple .bounds.")
            xmin, ymin, xmax, ymax = bx
            if not (xmax > xmin and ymax > ymin):
                raise RasterSubsetError(f"image_bounds geometry has degenerate bounds: {bx}.")
        else:
            raise RasterSubsetError(
                "Unsupported image_bounds format. Expected either a shapely geometry "
                "(having .geom_type and .bounds) or a numeric 4-tuple "
                "(xmin, ymin, xmax, ymax) like rioxarray.rio.bounds(). "
                f"Got {type(image_bounds).__name__}."
            )

        return func(self, roi, roi_crs_epsg, image_crs_epsg, image_bounds, *args, **kwargs)

    return checker


# -----------------------------------------------------------------------------------
class ROISubsetResolver:
    """
    Create resolved ROI for clipping in image_io.py (clip or clip_box)

    Example usage:
    resolvedROIsubset: ResolvedROISubset = ROISubsetResolver(roi, roi_crs_epsg, image_crs_epsg, image_bounds).run()

    :param roi:
        Region of interest. Supported forms are:

        **Primary (recommended)**

        * shapely geometry (interpreted in ``roi_crs_epsg``)
        * bounding box tuple ``(xmin, ymin, xmax, ymax)`` in ``roi_crs_epsg``

        **Convenience forms** (accepted, but less strict):

        * GeoJSON-like ``dict`` with a ``"type"`` key
        * list of ``[x, y]`` coordinate pairs defining a polygon
        * ``((x, y), half_width_m)`` defining a square box around a point
    :param roi_crs_epsg:
        EPSG code describing the CRS in which the ROI coordinates are expressed
    :param image_crs_epsg:
        EPSG code describing the CRS in which the image coordinates are expressed.
        For rioxarray raster use raster.rio.crs.to_epsg() to obtain EPSG code.
    :param image_bounds:
        Optional, if you want to test is roi is with the target image bounds.
        Supported forms are:
        * shapely geometry
        * tuple(x_min, y_min, x_max, y_max) (raster.rio.bounds)
        Must be in the crs image_crs_epsg
    :returns ResolvedROISubset:
        Dataclass with geometries or clip_box attributes
    """

    @validate_inputs
    def __init__(
        self,
        roi: Any,
        roi_crs_epsg: int | str,
        image_crs_epsg: str | int,
        image_bounds: Optional[Any] = None,
    ):
        self.roi: Any = roi
        self.roi_crs_epsg: int | str = roi_crs_epsg
        self.image_crs_epsg: str | int = image_crs_epsg
        self.image_bounds: Optional[Any] = image_bounds

    def run(self) -> ResolvedROISubset:
        """
        Run ROISubsetResolver.
        :returns: ResolvedROISubset
        """

        roi_crs_epsg = self.roi_crs_epsg
        image_crs_epsg = self.image_crs_epsg

        norm_roi = self._normalise_roi()  # transform requested ROI to shapely geometry

        resolved_roi = self._transform_geom(
            norm_roi
        )  # transform requested ROI (shapely geometry) to match image_crs_epsg

        x_min, y_min, x_max, y_max = resolved_roi.bounds

        geometries = self._geom_to_geometries(
            resolved_roi
        )  # convert resolved requested ROI (shapely geometry) into GeoJSON geometry mappings,
        if self.image_bounds is not None:
            if isinstance(self.image_bounds, tuple):
                _, box, _, _, _ = lazy_shapely()

                self.image_bounds = box(*self.image_bounds)

            if not resolved_roi.intersects(self.image_bounds):  # check if in the image bounds
                raise RasterSubsetError(
                    f"The requested ROI {resolved_roi.bounds} does not intersect with the image extent {self.image_bounds}. "
                    "Ensure that the ROI coordinates are correct and overlap the image bounds."
                )

        return ResolvedROISubset(
            clip_box=(x_min, y_min, x_max, y_max),
            geometries=geometries,
            roi=norm_roi,
            roi_crs_epsg=roi_crs_epsg,
            image_crs=image_crs_epsg,
        )

    def _normalise_roi(self) -> Any:
        """
        Normalise a user-provided ROI into a shapely geometry.

        :returns: shapely geometry or ``None``
        """

        roi = self.roi
        roi_crs_epsg = self.roi_crs_epsg

        # Call appropriate helper for roi type

        # 1. shapely geometry - return existing object
        if hasattr(roi, "geom_type"):
            return roi

        Polygon, box, _, shape, _ = lazy_shapely()

        # 2. bbox tuple - build shapely box
        if isinstance(roi, tuple) and len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi):
            return box(float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))

        # 3. GeoJSON-like dict - build shape from dict
        if isinstance(roi, dict) and "type" in roi:
            return shape(roi)

        # 4. List of coords - use helper function to build polygon
        if isinstance(roi, list) and roi and isinstance(roi[0], (list, tuple)) and len(roi[0]) == 2:
            return Polygon(roi)

        # 5. point box (i.e., centre and width) - use helper function to resolve to polygon
        if (
            isinstance(roi, tuple)
            and len(roi) == 2
            and isinstance(roi[0], (tuple, list))
            and len(roi[0]) == 2
            and isinstance(roi[1], (int, float))
        ):
            pt = (float(roi[0][0]), float(roi[0][1]))
            hw = float(roi[1])
            return self._roi_from_point_box(pt, hw, crs_epsg=roi_crs_epsg)

    def _transform_geom(self, geom: Any) -> Any:
        """
        Transform a shapely geometry between coordinate reference systems.

        :param geom: Geometry in source CRS (roi_crs_epsg)
        :returns: Geometry in destination CRS (image_crs_epsg)
        """
        roi_crs_epsg = self.roi_crs_epsg
        image_crs_epsg = self.image_crs_epsg

        pyproj = lazy_pyproj()

        src = pyproj.CRS.from_user_input(roi_crs_epsg)  # from_user_input to allow str
        dst = pyproj.CRS.from_user_input(image_crs_epsg)
        transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)

        *_, shp_transform = lazy_shapely()

        return shp_transform(transformer.transform, geom)

    @staticmethod
    def _geom_to_geometries(geom: Any) -> List[Dict[str, Any]]:
        """
        Convert a shapely geometry into GeoJSON geometry mappings.

        :param geom: shapely geometry
        :returns: list containing a single GeoJSON geometry mapping
        """

        _, _, mapping, _, _ = lazy_shapely()
        return [mapping(geom)]

    @staticmethod
    def _roi_from_point_box(point: Point, half_width: Union[int, float], *, crs_epsg: int | str):
        """
        Create a square ROI around a point with half-width in metres.

        If the CRS is geographic (degrees), a local Azimuthal Equidistant projection
        is used internally so that the box size is defined in metres.

        :param point: ``(x, y)`` coordinate of the box centre
        :param half_width: Half-width of the box in metres
        :param crs_epsg: EPSG code of the point CRS
        :returns: shapely polygon
        """

        pyproj = lazy_pyproj()
        _, box, _, _, _ = lazy_shapely()

        x, y = point
        crs = pyproj.CRS.from_user_input(crs_epsg)

        if crs.is_projected:
            return box(x - half_width, y - half_width, x + half_width, y + half_width)

        aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={y} +lon_0={x} +datum=WGS84 +units=m +no_defs")
        fwd = pyproj.Transformer.from_crs(crs, aeqd, always_xy=True)
        inv = pyproj.Transformer.from_crs(aeqd, crs, always_xy=True)

        x_m, y_m = fwd.transform(x, y)
        bb_m = (x_m - half_width, y_m - half_width, x_m + half_width, y_m + half_width)

        xmin, ymin = inv.transform(bb_m[0], bb_m[1])
        xmax, ymax = inv.transform(bb_m[2], bb_m[3])
        return box(xmin, ymin, xmax, ymax)


if __name__ == "__main__":
    pass
