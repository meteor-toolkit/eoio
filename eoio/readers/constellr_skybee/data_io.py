"""eoio.readers.constellr_skybee.data_io - data reading functionality for Skybee VNIR meas vars."""

from __future__ import annotations

import warnings

import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.constellr_skybee.layout import MEAS_VAR_RES, ConstellrSkybeeLayout
from eoio.readers.constellr_skybee.metadata import ConstellrSkybeeMetadataExtractor
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.utils.crs_utils import convert_xy
from eoio.utils.rasterio_utils import suggest_raster_chunks


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    layout: ConstellrSkybeeLayout,
    meas: list[str],
    subset: ResolvedROISubset | None,
    mtd: ConstellrSkybeeMetadataExtractor,
    chunks: dict[str, int] | None = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read Skybee image bands into an xarray dataset."""
    rxr = lazy_rioxarray()

    coord_dict = {}

    geometries = subset.geometries if subset else None

    if use_chunks and chunks is None:
        chunks = suggest_raster_chunks(str(layout.image_file()), target_mb=32.0)

    band_paths = layout.tif_band_files(meas)

    for bnd in meas:
        if bnd not in band_paths:
            warnings.warn(f"Band {bnd} not found in {layout.product_dir}, skipping.")
            continue
        # mask_and_scale applies _FillValue (0 = no data) -> NaN and scale_factor/add_offset lazily
        var_bnd = rxr.open_rasterio(band_paths[bnd], chunks=chunks, mask_and_scale=True)
        if geometries:
            # not from_disk: that would re-read the raw file, undoing the masking and scaling above
            var_bnd = var_bnd.rio.clip(geometries)
        # one band per file: drop that axis -- and only that one. A bare squeeze() would also collapse an x or y
        # dimension of length 1, which a small ROI at 60 m can easily produce.
        if "band" in var_bnd.dims:
            var_bnd = var_bnd.isel(band=0)
        # a scalar "band" coordinate left behind would collide with the "band" dimension the stack processor
        # creates (and to_datatree renames to)
        var_bnd = var_bnd.drop_vars("band", errors="ignore")
        # GDAL's per-file tags (UNITS, NAME, STATISTICS_*, AREA_OR_POINT, ...) describe the raw file, not this
        # variable: the statistics are stale once masked/scaled/clipped, and "UNITS" differs from the "units"
        # eoio sets only by case, which netCDF refuses to write ("NetCDF: String match to name in use").
        # Scaling is already applied (mask_and_scale), so nothing here is needed downstream.
        var_bnd.attrs = {}

        meas_var_res = get_var_geometry(bnd)

        if "x" in var_bnd.dims and "y" in var_bnd.dims:
            var_bnd = var_bnd.rename({"x": f"x_{meas_var_res}m", "y": f"y_{meas_var_res}m"})

        ds[bnd] = var_bnd

        if f"longitude_{meas_var_res}m" not in coord_dict:
            lon_new, lat_new = convert_xy(
                (var_bnd[f"x_{meas_var_res}m"], var_bnd[f"y_{meas_var_res}m"]),
                mtd.product_metadata["geospatial_bounds_crs"],
                "EPSG:4326",
            )
            coord_dict[f"longitude_{meas_var_res}m"] = lon_new
            coord_dict[f"latitude_{meas_var_res}m"] = lat_new

        # assign coord info
        ds[bnd] = ds[bnd].assign_coords(
            {
                f"longitude_{meas_var_res}m": (
                    (f"y_{meas_var_res}m", f"x_{meas_var_res}m"),
                    coord_dict[f"longitude_{meas_var_res}m"],
                ),
                f"latitude_{meas_var_res}m": (
                    (f"y_{meas_var_res}m", f"x_{meas_var_res}m"),
                    coord_dict[f"latitude_{meas_var_res}m"],
                ),
            }
        )
        try:
            ds[f"longitude_{meas_var_res}m"].attrs["units"]
        except KeyError:
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


def get_var_geometry(var: str) -> int:
    """Ground sampling distance (m) of band *var*, e.g. ``"B5"`` -> ``20``."""
    return MEAS_VAR_RES[var]


def get_var_resolution(var: str) -> str:
    """Resolution label of band *var*, e.g. ``"B5"`` -> ``"20m"``."""
    return f"{MEAS_VAR_RES[var]}m"
