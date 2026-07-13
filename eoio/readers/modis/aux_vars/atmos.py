"""eoio.readers.modis.aux_vars.atmos - atmospheric variable reading functions"""

from __future__ import annotations
from typing import List
from eoio.readers.modis.metadata.extractor import MODISMetadataExtractor
import xarray as xr


ATMOS_DICT = {
    "TCWV": "1km water_vapor",
    "AOD_Band1": "1km Atmospheric Optical Depth Band 1",
    "AOD_Band3": "1km Atmospheric Optical Depth Band 3",
    "AOD_Band8": "1km Atmospheric Optical Depth Band 8",
}


def add_atmos(
    *,
    ds: xr.Dataset,
    aux_ds: xr.Dataset,
    atmos_names: List[str],
    mtd: "MODISMetadataExtractor",
) -> xr.Dataset:
    """
    Builds xr.DataArray for requested atmospheric variables and attaches them to ds

    :param ds: xr.Dataset
    :param atmos_names: list of atmospheric variable names
    :param mtd: MODIS metadata extractor

    :return: ds with appended atmospheric variables
    """
    for atmos_name in atmos_names:
        atmos_key = ATMOS_DICT.get(atmos_name)
        if atmos_key is None:
            raise ValueError(f"Unknown atmospheric variable: {atmos_name}")
        if atmos_key not in aux_ds:
            raise ValueError(f"Atmospheric variable {atmos_key} not found in aux dataset")
        atmos_da = aux_ds[atmos_key]

        # format atmos data
        atmos_da = atmos_da.rename({"y": "y_grid_1000m", "x": "x_grid_1000m"}).isel(band=0)
        atmos_da = (atmos_da + atmos_da.attrs["add_offset"]) * atmos_da.attrs["scale_factor"]
        # attach to ds
        ds[atmos_name] = atmos_da

    return ds
