import numpy as np
import xarray as xr
from typing import Any, Dict, List

from eoio.readers.radcalnet.subset import RADCALNETSubset
from processor_tools.utils.formatters import datetime_from_yearday

# small translation dictionary so we can use our standard naming to index the radcalnet data
RADCALNET_NAMES = {
    "sza": "Zen",
    "saa": "Azi",
}


def read_file(
    path: str,
    aux_vars: list[str],
) -> xr.Dataset:
    """
    Read RadCalNet file from ascii
    """
    DATA: Dict[str, Any] = {}
    f = open(path, "r")
    lines = f.read().split("\n")
    for line in lines:
        x = [item.strip().replace(":", "") for item in line.split("\t")]
        k = x[0] if x[0] not in DATA else f"{x[0]}_unc"
        try:
            DATA[k] = [float(_x) for _x in x[1:]]
        except ValueError:
            DATA[k] = x[1:]
    f.close()

    dims = ["wavelength", "time"]
    attrs = {
        "Site": DATA.pop("Site"),
        "Lattitude": DATA.pop("Lat"),
        "Longitude": DATA.pop("Lon"),
        "Altitude": DATA.pop("Alt"),
    }
    data_vars: Dict[str, Any] = {
        "reflectance": (dims, []),
        "reflectance_uncertainty": (dims, []),
    }
    years = DATA.pop("Year")
    doy = DATA.pop("DOY(U)")
    doyl = DATA.pop("DOY(L)")
    utc = DATA.pop("UTC")
    local = DATA.pop("Local")
    time = [datetime_from_yearday(years[t], doy[t], utc[t]) for t in range(13)]
    l_time = [datetime_from_yearday(years[t], doyl[t], local[t]) for t in range(13)]
    wavelengths = []

    if "Zen" in DATA:
        data_vars["solar_zenith_angle"] = ("time", DATA.pop("Zen"))
    if "Azi" in DATA:
        data_vars["solar_azimuth_angle"] = (
            "time",
            np.array([row % 360 for row in DATA.pop("Azi")]),  # normalise to 360 degrees
        )

    data_vars["local_time"] = (
        "time",
        l_time,
    )  # store local time as a data_var for later
    for aux in aux_vars:
        if "angle" not in aux:
            data_vars[aux] = ("time", DATA.pop(aux))

    for k, val in DATA.items():
        if "unc" in k:
            if k.split("_")[0].isnumeric():
                data_vars["reflectance_uncertainty"][1].append(val)
            elif val:
                data_vars[k] = ("time", val)
        elif k:
            wavelengths.append(float(k))
            data_vars["reflectance"][1].append(val)

    coords = {"wavelength": ("wavelength", wavelengths), "time": ("time", time)}
    return xr.Dataset(data_vars, coords, attrs=attrs)


def read_dataset(*, ds: xr.Dataset, include_vars: List[str], subset: RADCALNETSubset) -> xr.Dataset:
    """
    Read RadCalNet file into a dataset
    """

    ds = ds[include_vars]

    id_time, id_wavelength = subset.series_indices, subset.wavelength_indices
    ds = ds.isel(time=id_time, wavelength=id_wavelength)

    return ds
