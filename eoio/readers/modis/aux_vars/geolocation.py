"""eoio.readers.modis.aux_vars.geolocation - geolocation reading functions"""

from __future__ import annotations
from typing import List
from eoio.readers.modis.metadata.extractor import MODISMetadataExtractor
import xarray as xr

FILE_RES_LIST = ["1000m", "500m", "250m"]


def add_geolocation(
    *,
    ds: xr.Dataset,
    geolocation_ds: xr.Dataset,
    resolution: List[str],
) -> xr.Dataset:
    """
    Attaches geolocation variables to ds as a coordinate

    :param ds: xr.Dataset
    :param geolocation_ds: xr.Dataset containing geolocation variables
    :param resolution: list of resolution strings

    :return: ds with appended geolocation variables
    """
    for res in resolution:
        # unpack resolution
        if res not in FILE_RES_LIST:
            raise ValueError(f"Unknown resolution: {res}")
        lat_var_name = f"Latitude_{res}" if res != "1000m" else "Latitude"
        lon_var_name = f"Longitude_{res}" if res != "1000m" else "Longitude"

        # read lat/lon grids from geolocation dataset
        if lat_var_name not in geolocation_ds:
            raise ValueError(f"Geolocation variable {lat_var_name} not found in geolocation dataset")
        if lon_var_name not in geolocation_ds:
            raise ValueError(f"Geolocation variable {lon_var_name} not found in geolocation dataset")

        lat = geolocation_ds[lat_var_name]
        lon = geolocation_ds[lon_var_name]

        # attach to ds
        ds = ds.assign_coords(
            {
                f"latitude_{res}": ((f"y_grid_{res}", f"x_grid_{res}"), lat.values),
                f"longitude_{res}": ((f"y_grid_{res}", f"x_grid_{res}"), lon.values),
            }
        )

    return ds


if __name__ == "__main__":
    pass
