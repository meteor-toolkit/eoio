"""eoio.readers.gistda_theos.data_io - data reading functionality for THEOS meas vars."""

from __future__ import annotations

import warnings
from typing import Any

import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.gistda_theos.layout import GISTDATHEOSLayout
from eoio.readers.gistda_theos.metadata import GISTDATHEOSMetadataExtractor
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.utils.crs_utils import convert_xy
from eoio.utils.rasterio_utils import suggest_raster_chunks
import numpy as np


class GISTDATHEOSDataIOError(ValueError):
    pass


#: IMAGERY.TIF stores raster bands in a fixed physical order that does NOT match
#: the meas name numbering -- verified against a real product's own
#: TIFFTAG_IMAGEDESCRIPTION ("B3 B2 B1 B4 ") and METADATA.DIM's Band_Display_Order
#: (RED_CHANNEL=3, GREEN_CHANNEL=2, BLUE_CHANNEL=1) -- band 4 is the only one whose
#: file position already matches its own name. 0-indexed position within
#: IMAGERY.TIF's "band" dimension for each meas name.
_BAND_FILE_POSITION = {"B3": 0, "B2": 1, "B1": 2, "B4": 3}


def _band_file_position(bnd: str) -> int:
    """*bnd*'s 0-indexed position in IMAGERY.TIF's own band order -- see
    :py:data:`_BAND_FILE_POSITION`. Reading by *meas*'s requested-list position
    instead (as this used to) silently swaps B1 and B3's data whenever either is
    requested, since only B2/B4 happen to already sit at the position their own
    name would suggest.

    :raises GISTDATHEOSDataIOError: *bnd* isn't one of THEOS's 4 known bands.
    """
    try:
        return _BAND_FILE_POSITION[bnd]
    except KeyError:
        raise GISTDATHEOSDataIOError(
            f"Unknown GISTDA THEOS band {bnd!r} -- expected one of {sorted(_BAND_FILE_POSITION)}"
        ) from None


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    layout: GISTDATHEOSLayout,
    meas: list[str],
    subset: ResolvedROISubset | None,
    mtd: GISTDATHEOSMetadataExtractor,
    chunks: dict[str, int] | None = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read THEOS image bands into an xarray dataset."""
    rxr = lazy_rioxarray()

    # get meas var res
    meas_var_res = get_var_geometry()

    # set up chunking
    if use_chunks and chunks is None:
        chunks = suggest_raster_chunks(str(layout.image_file()), target_mb=32.0)

    geometries = subset.geometries if subset else None

    # open full image (not yet clipped) to get its CRS and any GCPS
    img = rxr.open_rasterio(layout.image_file(), chunks=chunks)
    img, transform, gcp_crs = georeference_from_gcps(img)

    # Full-resolution 2D lon/lat grid, computed once and carried through clip/isel
    # as ordinary (y, x) coordinates -- x/y themselves stay bare pixel indices (see
    # georeference_from_gcps) since a rotated transform can't be represented by
    # independent 1-D x/1-D y coordinates the way an unrotated grid can.
    height, width = img.sizes["y"], img.sizes["x"]
    lon_full, lat_full = lonlat_grid_from_gcps(transform, width=width, height=height)
    target_crs = mtd.product_metadata["geospatial_bounds_crs"]
    if str(gcp_crs) != str(target_crs):
        lon_full, lat_full = convert_xy((lon_full, lat_full), gcp_crs, target_crs)
    img = img.assign_coords(longitude=(("y", "x"), lon_full), latitude=(("y", "x"), lat_full))

    for i, bnd in enumerate(meas):
        file_pos = _band_file_position(bnd)
        if geometries:
            with warnings.catch_warnings():
                # rioxarray tries to recompute .rio.transform() for the clipped
                # window from x/y spacing and warns that it can't for a rotated
                # transform -- harmless here, since we never rely on that
                # recalculation: the real 2D longitude/latitude coordinates
                # (assigned below from our own transform) are what every
                # downstream consumer actually reads.
                warnings.filterwarnings("ignore", message="Transform that is non-rectilinear or with rotation found")
                var_bnd = img.isel(band=file_pos).rio.clip(geometries, from_disk=True)
        else:
            # isel already removes the band axis; a squeeze() on top would also collapse an x or y
            # dimension of length 1, which a small ROI can easily produce
            var_bnd = img.isel(band=file_pos)

        if "x" in var_bnd.dims and "y" in var_bnd.dims:
            var_bnd = var_bnd.rename({"x": f"x_{meas_var_res}m", "y": f"y_{meas_var_res}m"})

        # "longitude"/"latitude" (the full-resolution pixel-index-keyed coords
        # assigned above) are dropped here, not carried into ds -- their values
        # for this var_bnd are read into the properly resolution-suffixed
        # longitude_{res}m/latitude_{res}m coords below instead.
        ds[bnd] = var_bnd.where(var_bnd != 0).drop_vars(["band", "longitude", "latitude"])

        # convert DNs to TOA radiance using scale factor (conversion to reflectance is the units processor's job)
        band_mtd = mtd.variable_product_metadata(bnd)
        band_gain = band_mtd.get("band_gain")
        band_bias = band_mtd.get("band_bias")
        if band_gain is None or band_bias is None:
            raise GISTDATHEOSDataIOError(f"No radiometric gain/bias for band {bnd} in METADATA.DIM -- cannot scale it")
        # assign to .data (not .values) so a dask-backed band stays lazy
        ds[bnd].data = ds[bnd].data / float(band_gain) + float(band_bias)

        # longitude/latitude were already clipped/sliced along with var_bnd itself
        # (ordinary xarray coordinate propagation -- they share var_bnd's (y, x)
        # dims), so no further CRS conversion or resolution is needed here.
        if i == 0:
            ds.coords[f"longitude_{meas_var_res}m"] = (
                (f"y_{meas_var_res}m", f"x_{meas_var_res}m"),
                var_bnd.coords["longitude"].values,
            )
            ds.coords[f"latitude_{meas_var_res}m"] = (
                (f"y_{meas_var_res}m", f"x_{meas_var_res}m"),
                var_bnd.coords["latitude"].values,
            )
            ds[f"latitude_{meas_var_res}m"].attrs.update(
                {
                    "units": "degrees_north",
                    "standard_name": f"latitude {meas_var_res} m",
                    "long_name": f"latitude {meas_var_res} m resolution",
                }
            )
            ds[f"longitude_{meas_var_res}m"].attrs.update(
                {
                    "units": "degrees_east",
                    "standard_name": f"longitude {meas_var_res} m",
                    "long_name": f"longitude {meas_var_res} m resolution",
                }
            )

    return ds


def get_var_geometry() -> str:
    """
    Get the variable geometry

    :return: string of geometry
    """

    return "15"  # geometries_sensor[meas_vars.index(var)]


def get_var_resolution() -> str:
    """
    Get the variable resolution

    :return: string of resolution
    """

    return "15m"


def georeference_from_gcps(img: xr.DataArray) -> tuple[xr.DataArray, Any, str]:
    """Attach a real affine transform + CRS to *img* from its own embedded ground
    control points (GISTDA THEOS ships no GDAL geotransform -- only GCPs -- so
    ``rxr.open_rasterio`` otherwise leaves ``img``'s x/y as bare pixel indices with
    no CRS at all: verified directly against a real product's IMAGERY.TIF).

    Fits one affine transform across every GCP (:py:func:`rasterio.transform.from_gcps`,
    a least-squares fit from >= 3 points) -- appropriate for a geometrically-corrected
    Level-1 scene like this, where a single overall rotation/scale (THEOS's own
    SCENE_ORIENTATION in METADATA.DIM) explains the corners, not a per-pixel
    nonlinear distortion; THEOS provides exactly the 4 scene corners, so the fit is
    over-determined by 2 equations and therefore self-consistency-checked.

    :param img: DataArray from ``rxr.open_rasterio`` -- GCPs read via rioxarray's
        own ``.rio.get_gcps()``/``.rio.crs`` (which read a GeoTIFF's embedded GDAL
        GCP tags directly), not reader-specific parsing.
    :return: ``(img, transform, gcp_crs)`` -- *img* with CRS/transform written in
        place (so :py:meth:`rioxarray.raster_array.RasterArray.clip`, and any other
        rioxarray spatial op, can use them), the fitted affine transform and its CRS
        (for :py:func:`lonlat_grid_from_gcps` to reuse without refitting).
    :raises GISTDATHEOSDataIOError: *img* has no usable ground control points.
    """
    gcps = img.rio.get_gcps()
    if not gcps:
        raise GISTDATHEOSDataIOError(
            "no ground control points found on image -- cannot georeference this THEOS product"
        )
    gcp_crs = img.rio.crs
    if gcp_crs is None:
        raise GISTDATHEOSDataIOError("ground control points present but have no associated CRS")

    from rasterio.transform import from_gcps

    transform = from_gcps(gcps)
    img = img.rio.write_transform(transform)
    img = img.rio.write_crs(gcp_crs)
    return img, transform, str(gcp_crs)


def lonlat_grid_from_gcps(transform: Any, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution ``(height, width)`` longitude/latitude grids from *transform*
    (see :py:func:`georeference_from_gcps`) -- a real 2-D ``(row, col) -> (lon, lat)``
    mapping.

    Deliberately not row/col treated as independent 1-D axes (wrong whenever the
    transform has any rotation term, which GISTDA THEOS scenes generally do -- see
    SCENE_ORIENTATION in METADATA.DIM: verified on a real product with ~10 deg of
    rotation that independent axes mislabel most pixels by ~0.2 deg / ~20 km).

    :param transform: affine transform from :py:func:`georeference_from_gcps`.
    :param width: image width in pixels.
    :param height: image height in pixels.
    :return: ``(lon, lat)``, each shape ``(height, width)``.
    """
    from rasterio.transform import xy as rio_xy

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    lon, lat = rio_xy(transform, rows.ravel(), cols.ravel())
    return np.asarray(lon).reshape(height, width), np.asarray(lat).reshape(height, width)
