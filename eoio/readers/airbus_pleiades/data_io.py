"""eoio.readers.airbus_pleiades.data_io - data reading functionality for Airbus Pleiades meas vars."""

from __future__ import annotations
from typing import Dict, List, Optional
from eoio.deps import lazy_rioxarray
from eoio.readers.airbus_pleiades.layout import PleiadesLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.readers.airbus_pleiades.metadata import PleiadesMetadataExtractor
import xarray as xr
from copy import deepcopy

from eoio.utils.crs_utils import convert_xy
from eoio.utils.rasterio_utils import suggest_raster_chunks


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    layout: PleiadesLayout,
    meas: List[str],
    subset: Optional[ResolvedROISubset],
    mtd: PleiadesMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
    mask_zero_as_nodata: bool = True,
) -> xr.Dataset:
    """
    Read in image data

    Add image meas_var data to meas_vars present in the initialised xr.Dataset,
    subsetting in accordance with the subsetting parameters provided in roi_subset.
    Resulting xr.Dataset has converted input coordinates from initial
    coordinate reference system to the World Geodetic System 1984 (WGS 84)
    """
    # lazy import
    rxr = lazy_rioxarray()

    # get meas var res
    meas_vars_res = get_var_geometry()

    # set up chunking
    if use_chunks and chunks is None:
        chunks = suggest_raster_chunks(str(layout.image_file()), target_mb=32.0)

    coord_dict = {}

    geometries = subset.geometries if subset else None

    for b, bnd in enumerate(meas):
        if geometries:
            var_bnd = (
                rxr.open_rasterio(layout.image_file(), chunks=chunks).isel(band=b).rio.clip(geometries, from_disk=True)
            ).squeeze()
        else:
            var_bnd = rxr.open_rasterio(layout.image_file(), chunks=chunks).isel(band=b).squeeze()
        if b == 0:
            lon_new, lat_new = convert_xy(
                (var_bnd.x, var_bnd.y),
                mtd.product_metadata["geospatial_bounds_crs"],
                "EPSG:4326",
            )
            coord_dict[f"longitude_{meas_vars_res}m"] = lon_new
            coord_dict[f"latitude_{meas_vars_res}m"] = lat_new

        var_bnd = var_bnd.rename({"x": f"x_{meas_vars_res}m", "y": f"y_{meas_vars_res}m"})

        # replace 0 with nans
        ds[bnd] = var_bnd.where(var_bnd != 0).drop_vars("band")

        # convert DNs to TOA radiance using scale factor
        bnd_gain = mtd.variable_product_metadata(bnd).get("band_gain")
        bnd_bias = mtd.variable_product_metadata(bnd).get("band_bias")
        ds[bnd].values = ds[bnd].values / float(bnd_gain) + float(bnd_bias)  # type: ignore[arg-type]

        # assign coord info
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

    ds = reorder_bands(ds)

    return ds


def get_var_geometry() -> str:
    """
    Get the variable geometry

    :return: string of geometry
    """

    return "2"  # geometries_sensor[meas_vars.index(var)]


def get_var_resolution() -> str:
    """
    Get the variable resolution

    :return: string of resolution
    """

    return "2m"


# @staticmethod
def reorder_bands(ds):
    # Pleiades data is provided in order of B3, B2, B1, B4 (i.e. R, G, B, NIR) want to reorder it to B1, B2, B3, B4 (i.e. B, G, R, NIR)
    ds = ds.rename_vars({"B1": "B3", "B3": "B1"})
    for k, v in deepcopy(ds["B3"].attrs).items():
        if isinstance(v, str):
            if "B1" in v and "B3" not in v:
                ds["B3"].attrs.update({k: v.replace("B1", "B3")})
    for k, v in deepcopy(ds["B1"].attrs).items():
        if isinstance(v, str):
            if "B3" in v and "B1" not in v:
                ds["B1"].attrs.update({k: v.replace("B3", "B1")})
    return ds
