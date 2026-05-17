"""Angles handling for EMIT (adds solar/viewing angles)."""

from __future__ import annotations
import xarray as xr


def add_angles(ds: xr.Dataset, obs_ds: xr.Dataset, angle_names: list[str]) -> xr.Dataset:
    """
    Add observation and solar angles to the dataset from the corresponding OBS file.

    Parameters
    ----------
    ds : xr.Dataset
        The target dataset to which angles will be added.
    obs_ds : xr.Dataset
        The observation dataset (OBS) containing angle information.
    angle_names : list of str
        List of angle variable names to add.

    Returns
    -------
    xr.Dataset
        The dataset with added angle variables.

    Notes
    -----
    Mirrors the behaviour of the former read_angles() method.
    """

    ang_label_dict = {
        "viewing_zenith_angle": "To-sensor zenith",
        "viewing_azimuth_angle": "To-sensor azimuth",
        "solar_zenith_angle": "To-sun zenith",
        "solar_azimuth_angle": "To-sun azimuth",
    }

    for k in list(ang_label_dict.keys()):
        if k not in angle_names:
            ang_label_dict.pop(k)

    for k, v in ang_label_dict.items():
        vals = obs_ds.sel(bands=v).obs.values if "bands" in obs_ds.coords else obs_ds[v].values
        ds = ds.assign({k: (["y", "x"], vals)})
        ds[k].attrs.update(
            {
                "units": "degrees",
                "standard_name": k,
                "geometry_id": "60m",
                "resolution": 60.0,
            }
        )

    return ds
