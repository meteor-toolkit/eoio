"""eoio.readers.modis.data_io - data reading functionality for MODIS products"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from shapely import Polygon, contains_xy
import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.modis.layout import MODISLayout
from eoio.readers.modis.metadata.extractor import MODISMetadataExtractor
from eoio.readers.modis.metadata.var_names import L2_MEAS_VAR_BAND_IDS_1, L2_MEAS_VAR_BAND_IDS_H, L2_MEAS_VAR_BAND_IDS_Q
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.utils.rasterio_utils import suggest_raster_chunks
from eoio.readers.modis.aux_vars.geolocation import add_geolocation

L2_FILE_RES_DICT = {
    "1": L2_MEAS_VAR_BAND_IDS_1,
    "H": L2_MEAS_VAR_BAND_IDS_H,
    "Q": L2_MEAS_VAR_BAND_IDS_Q,
}


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    geolocation_ds: xr.Dataset,
    layout: MODISLayout,
    meas_vars: List[str],
    subset: Optional[ResolvedROISubset],
    mtd: MODISMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
    preferred_resolution: Optional[int] = None,
    mask_zero_as_nodata: bool = True,
) -> xr.Dataset:

    rxr = lazy_rioxarray()

    # ------------------------------------------------------------------
    # Product metadata
    # ------------------------------------------------------------------
    prod_mtd = mtd.product_metadata or {}

    # ------------------------------------------------------------------
    # Set up chunking
    # ------------------------------------------------------------------
    if use_chunks and chunks is None:
        chunks = suggest_raster_chunks(str(layout.path), target_mb=32.0)

    # ------------------------------------------------------------------
    # Read and format bands
    # ------------------------------------------------------------------
    if layout.processing_level == "L1B":
        da = rxr.open_rasterio(layout.path, chunks=chunks)
    elif layout.processing_level == "L2":
        da = rxr.open_rasterio(layout.path, chunks=chunks)[list(L2_FILE_RES_DICT.keys()).index(layout.file_res_key)]

    # add lat/lon to dataset
    ds = add_geolocation(ds=ds, geolocation_ds=geolocation_ds, resolution=[f"{preferred_resolution}m"])

    # --------------------------------------------------------------
    # Rename spatial dimensions by actual raster resolution
    # --------------------------------------------------------------
    rename_map = {}
    if "x" in da.dims:
        rename_map["x"] = f"x_grid_{preferred_resolution}m"
    if "y" in da.dims:
        rename_map["y"] = f"y_grid_{preferred_resolution}m"
    if rename_map:
        da = da.rename(rename_map)

    # --------------------------------------------------------------
    # Apply subset clipping
    # --------------------------------------------------------------
    if subset is not None:
        if subset.clip_box:
            x_min, y_min, x_max, y_max = subset.clip_box
            lon = ds[f"longitude_{preferred_resolution}m"]
            lat = ds[f"latitude_{preferred_resolution}m"]
            mask = (lat >= y_min) & (lat <= y_max) & (lon >= x_min) & (lon <= x_max)
            da = da.where(mask, drop=True)
            ds = ds.where(mask, drop=True)

    for var in meas_vars:
        var_mtd = mtd.variable_product_metadata(var) or {}
        var_da = da[var_mtd["band_id"]].isel(band=var_mtd["band_idx"])
        # --------------------------------------------------------------
        # Mask nodata before scaling
        # --------------------------------------------------------------
        try:
            encoded_nodata = da.rio.encoded_nodata
        except Exception:
            encoded_nodata = None

        if encoded_nodata is not None:
            var_da = var_da.where(var_da != encoded_nodata)

        if mask_zero_as_nodata and var.startswith("B"):
            var_da = var_da.where(var_da != 0)

        # --------------------------------------------------------------
        # Variable-specific decoding
        # --------------------------------------------------------------
        if var.startswith("B"):
            off = var_mtd.get("reflectance_offsets", 0.0)
            q = var_mtd.get("reflectance_scales", 1.0)

            var_da = var_da.where(var_da != var_da.attrs.get("_FillValue"))
            var_da = var_da.astype("float32")
            var_da -= np.float32(off)
            var_da *= np.float32(q)

        else:
            # Unknown / future layer type: leave as read
            pass

        # Strip attrs that rioxarray may attach to spatial coords
        for coord_name in var_da.coords:
            if coord_name == "x" or coord_name == "y" or coord_name.startswith("x_") or coord_name.startswith("y_"):
                var_da.coords[coord_name].attrs = {}

        ds[var] = var_da

    # Apply quality flags if present
    ds = ds.where(
        (
            da[f"{var_mtd.get('geometry_id')} Reflectance Band Quality"].sel(band=1)
            == int("0b1000000000000000000000000000000", 2)
        ).values
    )

    return ds
