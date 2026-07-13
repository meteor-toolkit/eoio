"""eoio.readers.modis.aux_vars.angles - angle reading functions"""

from __future__ import annotations
from typing import List
from eoio.readers.modis.metadata.extractor import MODISMetadataExtractor
import xarray as xr


ANGLE_DICT = {
    "solar_zenith_angle": "SolarZenith",
    "solar_azimuth_angle": "SolarAzimuth",
    "viewing_zenith_angle": "SensorZenith",
    "viewing_azimuth_angle": "SensorAzimuth",
}


def add_angles(
    *,
    ds: xr.Dataset,
    aux_ds: xr.Dataset,
    angle_names: List[str],
    mtd: "MODISMetadataExtractor",
) -> xr.Dataset:
    """
    Builds xr.DataArray for requested angle variables and attaches them to ds

    :param ds: xr.Dataset
    :param angle_names: list of angle variable names
    :param mtd: MODIS metadata extractor

    :return: ds with appended angles
    """
    for angle_name in angle_names:
        # unpack angle name
        if angle_name not in ANGLE_DICT:
            raise ValueError(f"Unknown angle: {angle_name}")
        aux_var_name = ANGLE_DICT[angle_name]

        # read angle grid from aux dataset
        if aux_var_name not in aux_ds:
            raise ValueError(f"Angle variable {aux_var_name} not found in aux dataset")
        ang = aux_ds[aux_var_name]

        # format ang data
        ang = ang.rename({"nscans*10": "y_grid_1000m", "mframes": "x_grid_1000m"})
        # attach to ds
        ds[angle_name] = ang

    return ds


if __name__ == "__main__":
    pass
