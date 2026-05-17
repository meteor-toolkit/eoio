"""eoio.readers.planetscope.data_io - PlanetScope image IO utilities."""

from typing import Dict, List, Optional
import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.readers.planetscope.layout import PlanetScopeLayout
from eoio.readers.planetscope.metadata import PlanetScopeMetadataExtractor

from eoio.utils.crs_utils import convert_xy
from eoio.utils.rasterio_utils import suggest_raster_chunks

""" TODO-List:
- Metadata retrieved from .tif, shall they be stored as attributes? Or stored as internal variables and later be potentially populated?
"""


def read_tif_into_dataset(
    *,
    ds: xr.Dataset,
    layout: PlanetScopeLayout,
    meas: List[str],
    subset: Optional[ResolvedROISubset],
    mtd: PlanetScopeMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
    mask_zero_as_nodata: bool = True,
) -> xr.Dataset:
    """
    Read PlanetScope image into an xarray Dataset, with optional ROI subsetting,
    per-resolution lon/lat coordinate generation, and CF-ish coordinate attributes.
    """

    # lazy import
    rxr = lazy_rioxarray()

    # get meas var res
    meas_vars_res = get_var_geometry()

    # set up chunking
    if use_chunks and chunks is None:
        chunks = suggest_raster_chunks(str(layout.image_file), target_mb=32.0)

    coord_dict = {}

    geometries = subset.geometries if subset else None

    for b, bnd in enumerate(meas):
        if geometries:
            var_bnd = (
                rxr.open_rasterio(layout.image_file, chunks=chunks).isel(band=b).rio.clip(geometries, from_disk=True)
            ).squeeze()
        else:
            var_bnd = rxr.open_rasterio(layout.image_file, chunks=chunks).isel(band=b).squeeze()

        if b == 0:
            lon_new, lat_new = convert_xy(
                (var_bnd.x, var_bnd.y),
                mtd.product_metadata["geospatial_bounds_crs"],
                "EPSG:4326",
            )
            coord_dict[f"longitude_{meas_vars_res}m"] = lon_new
            coord_dict[f"latitude_{meas_vars_res}m"] = lat_new

        # get per band values for scale and offset and fill value
        img_scale_factor = var_bnd.attrs.get("scale_factor", None)
        img_offset = var_bnd.attrs.get("add_offset", None)
        img_fill_value = var_bnd.attrs.get("_FillValue", 0)  # usially 0 in PSD data

        # apply scale and offset
        var_bnd = var_bnd.astype("float32")
        if (img_scale_factor is not None) and (img_scale_factor != 1.0):
            var_bnd /= img_scale_factor
        if img_offset is not None:
            var_bnd = var_bnd.where(var_bnd != img_fill_value)  # avoid adding offset to fill value data
            var_bnd += img_offset

        var_bnd = var_bnd.rename({"x": f"x_{meas_vars_res}m", "y": f"y_{meas_vars_res}m"})

        ds[bnd] = var_bnd.where(var_bnd != img_fill_value)

        # convert DNs to TOA radiance using scale factor
        rad_coeff = mtd.variable_product_metadata(bnd).get("radiometric_scale_factor")
        ds[bnd].values = ds[bnd].values * float(rad_coeff)  # type: ignore[arg-type]

        ds[bnd] = ds[bnd].assign_coords(
            {
                f"longitude_{meas_vars_res}m": (
                    [f"y_{meas_vars_res}m", f"x_{meas_vars_res}m"],
                    coord_dict[f"longitude_{meas_vars_res}m"],
                ),
                f"latitude_{meas_vars_res}m": (
                    [f"y_{meas_vars_res}m", f"x_{meas_vars_res}m"],
                    coord_dict[f"latitude_{meas_vars_res}m"],
                ),
            }
        )
        try:
            ds[f"longitude_{meas_vars_res}m"].attrs["units"]
        except KeyError:
            ds[f"latitude_{meas_vars_res}m"].attrs.update(
                {
                    "units": "degrees_north",
                    "standard_name": f"latitude {meas_vars_res} m",
                    "long_name": f"latitude {meas_vars_res} m resolution",
                }
            )
            ds[f"longitude_{meas_vars_res}m"].attrs.update(
                {
                    "units": "degrees_east",
                    "standard_name": f"longitude {meas_vars_res} m",
                    "long_name": f"longitude {meas_vars_res} m resolution",
                }
            )

    return ds


def get_var_geometry() -> str:
    """
    Get the variable geometry

    :return: string of geometry
    """

    return "3"


def get_var_resolution() -> str:
    """
    Get the variable resolution

    :return: string of resolution
    """

    return "3m"
