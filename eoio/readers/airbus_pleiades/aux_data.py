"""eoio.readers.airbus_pleiades.aux_data - auxiliary data reading functions for Airbus Pleiades reader."""

from __future__ import annotations
import warnings
from typing import Any, Optional, Sequence
import numpy as np
import xarray as xr
from processor_tools.utils.dict_tools import get_value
from eoio.readers.airbus_pleiades.metadata import PleiadesMetadataExtractor


def read_aux(ds, aux, mtd: PleiadesMetadataExtractor):
    if aux is None:
        raise ValueError(
            "No auxiliary data files have been selected. Please set aux before reading using `.aux = [...]`."
        )
    if "observation_geometry" in aux:
        ds = add_angles(ds, mtd)
    if "mask" in aux:  # TODO: update naming when masks implemented
        warnings.warn(
            "Mask data is not yet implemented for Airbus Pleiades data. Please check back later for updates."
        )  # TODO add mask here when implemented
    warnings.warn(
        "Auxiliary data is not yet implemented for Airbus Pleiades data. Please check back later for updates."
    )
    return ds


def add_aux(
    *,
    ds: xr.Dataset,
    layout: Any,
    aux_names: Any,
    # subset: Any,
) -> xr.Dataset:
    """
    Currently no mask implemented for Pleiades data; placeholder function.

    Will read mask data from Airbus Pleiades metadata and add them to ds.

    Will return original ds unchanged on unexpected formats.
    """
    return ds


def add_angles(
    ds: xr.Dataset,
    mtd: PleiadesMetadataExtractor,
    # layout: Any,
    # subset: Any,
) -> xr.Dataset:
    """
    Read viewing / solar angles from Airbus Pleiades metadata and add them to ds.

    Returns original ds unchanged if angles in unexpected format.
    """

    # locate geometric metadata list (prefer ds.attrs, else parse XML)
    def _locate_angles_list() -> Optional[Sequence[Any]]:
        if "product_metadata" in ds.attrs and "located_geometric_values" in ds.attrs["product_metadata"]:
            return ds.attrs["product_metadata"]["located_geometric_values"]
        try:  # when extracting basic metadata
            prod_mtd = mtd.product_metadata
            if get_value(prod_mtd, "located_geometric_values"):
                md = get_value(prod_mtd, "located_geometric_values")

                # deep search for located_geometric_values
                def _deep_search(d):
                    if isinstance(d, dict):
                        return get_value(d, "located_geometric_values")
                    if isinstance(d, list):
                        for item in d:
                            val_res = _deep_search(item)
                            if val_res is not None:
                                return val_res
                    return None

                return _deep_search(md)
        except Exception:
            return None
        return None

    angles_list = _locate_angles_list()
    if not angles_list:
        return ds

    lon = np.array(
        [
            ds.longitude_2m[0, int(np.floor((ds.longitude_2m.shape[1] - 1) / 2))].values,
            ds.longitude_2m[
                int(np.floor((ds.longitude_2m.shape[0] - 1) / 2)),
                int(np.floor((ds.longitude_2m.shape[1] - 1) / 2)),
            ].values,
            ds.longitude_2m[-1, int(np.floor((ds.longitude_2m.shape[1] - 1) / 2))].values,
        ]
    )
    lat = np.array(
        [
            ds.latitude_2m[0, int(np.floor((ds.latitude_2m.shape[1] - 1) / 2))].values,
            ds.latitude_2m[
                int(np.floor((ds.latitude_2m.shape[0] - 1) / 2)),
                int(np.floor((ds.latitude_2m.shape[1] - 1) / 2)),
            ].values,
            ds.latitude_2m[-1, int(np.floor((ds.latitude_2m.shape[1] - 1) / 2))].values,
        ]
    )

    y_angles = [
        ds.y_2m[0].values,
        ds.y_2m[int(np.floor((ds.latitude_2m.shape[0] - 1) / 2))].values,
        ds.y_2m[-1].values,
    ]
    x_angles = [ds.x_2m[int(np.floor((ds.longitude_2m.shape[1] - 1) / 2))].values]

    for i, j in zip(
        [
            ("Solar_Incidences", "SUN_AZIMUTH"),
            ("Solar_Incidences", "SUN_ELEVATION"),
            ("Acquisition_Angles", "AZIMUTH_ANGLE"),
            ("Acquisition_Angles", "VIEWING_ANGLE"),
        ],
        [
            "solar_azimuth_angle",
            "solar_zenith_angle",
            "sensor_azimuth_angle",
            "sensor_zenith_angle",
        ],
    ):
        if j == "solar_azimuth_angle":
            angles = [float(x[i[0]][i[1]]["#text"]) for x in angles_list]
        elif j == "solar_zenith_angle":
            angles = [90 - float(x[i[0]][i[1]]["#text"]) for x in angles_list]
        elif j == "sensor_azimuth_angle":
            angles = [float(x[i[0]][i[1]]) for x in angles_list]
        elif j == "sensor_zenith_angle":
            angles = [float(x[i[0]][i[1]]["#text"]) for x in angles_list]
        else:
            angles = []

        ds.attrs.update({f"{j}": angles})  # set angles as attributes

        y_angles = [np.nan for x in y_angles]
        x_angles = [np.nan for x in x_angles]
        ds[f"{j}"] = xr.DataArray(
            np.array(angles).reshape((len(np.array(angles)), 1)),
            {"y_2m_angles": y_angles, "x_2m_angles": x_angles},
            ("y_2m_angles", "x_2m_angles"),
        )

        # add angle attrs from metadata
        ds[f"{j}"].attrs.update(mtd.get_angle_metadata()[f"{j}"])

        ds = ds.assign_coords(
            {
                "latitude_2m_angles": (
                    ["y_2m_angles", "x_2m_angles"],
                    lat.reshape(len(lat), 1),
                ),
                "longitude_2m_angles": (
                    ["y_2m_angles", "x_2m_angles"],
                    lon.reshape(len(lon), 1),
                ),
            }
        )

    return ds
