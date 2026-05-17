"""eoio.readers.sentinel2.aux_vars.angles - viewing geometry angle reading functions"""

from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from eoio.readers.sentinel2.metadata.extractor import S2MSIMetadataExtractor
from eoio.readers.sentinel2.metadata.var_names import MEAS_VAR_BAND_IDS
import numpy as np
import xarray as xr


def add_angles(
    *,
    ds: xr.Dataset,
    angle_names: List[str],
    mtd: "S2MSIMetadataExtractor",
    ave_det: bool = True,
) -> xr.Dataset:
    """
    Builds xr.DataArray for requested angle variables and attaches them to ds

    :param ds: xr.Dataset
    :param angle_names: list of angle variable names
    :param mtd: S2 metadata extractor
    :param ave_det: (default: `True`) option to average viewing angles between detectors

    :returns: Dataset with appended angles.
    """

    # unpack grid metadata
    col_step, row_step = mtd.tl_xml_reader.find_sun_angle_steps()
    geo_10m = mtd.variable_product_metadata("B02").get("geoposition")

    # read each angle, build xr.DataArray, and attached to ds
    for i, angle_name in enumerate(angle_names):
        # unpack angle name
        parts = angle_name.split("_")
        obj = parts[0]  # "solar" or "viewing"
        direction = parts[1]  # "azimuth" or "zenith"

        # read angle grid from metadata
        if obj == "viewing":
            band = parts[3]  # band name
            band_id = MEAS_VAR_BAND_IDS[band]  # band_id

            ang, detector_ids = mtd.tl_xml_reader.find_viewing_angle_grid(band_id=band_id, direction=direction)

        elif obj == "solar":
            ang = mtd.tl_xml_reader.find_sun_angle_grid(direction=direction)

        else:
            raise ValueError(f"Unknown angle: {angle_name}")

        # build dimension arrays (reuse for each angle grid)
        if i == 0:
            # Key fix: derive (nrows, ncols) robustly for 2D (solar) and 3D (viewing) arrays
            if ang.ndim == 2:
                nrows, ncols = ang.shape
            elif ang.ndim == 3:
                # expected (detector, rows, cols)
                _, nrows, ncols = ang.shape
            else:
                raise ValueError(f"Unexpected angle array shape: {ang.shape}")

            x = (geo_10m or {})["ulx"] + np.arange(ncols) * col_step  # type: ignore[index]
            y = (geo_10m or {})["uly"] - np.arange(nrows) * row_step  # type: ignore[index]
            x_dim_name = "x_" + str(col_step) + "m"
            y_dim_name = "y_" + str(row_step) + "m"

        if obj == "viewing":
            ang_da = xr.DataArray(
                ang,
                dims=("detector", y_dim_name, x_dim_name),
                coords={
                    "detector": detector_ids,
                    x_dim_name: x,
                    y_dim_name: y,
                },
            )

            if ave_det:
                ang_da = ang_da.mean(dim="detector", skipna=True)

        else:
            ang_da = xr.DataArray(
                ang,
                dims=(y_dim_name, x_dim_name),
                coords={
                    x_dim_name: x,
                    y_dim_name: y,
                },
            )

        ds[angle_name] = ang_da

    return ds


if __name__ == "__main__":
    pass
