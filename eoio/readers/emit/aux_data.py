"""Auxiliary data helpers for EMIT."""

from __future__ import annotations
from typing import Any, Optional
import xarray as xr

from eoio.readers.emit.angles import add_angles

MISC_AUX_VARS = [
    "Slope",
    "Aspect",
    "Path length",
    "Solar phase",
    "Cosine(i)",
    "UTC Time",
    "Earth-sun distance",
]


def add_aux(
    ds: xr.Dataset,
    src: xr.Dataset,
    layout: Any,
    subset: Optional[Any] = None,
    config: Optional[Any] = None,
) -> xr.Dataset:
    """
    Add auxiliary datasets to the main dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The main dataset to which auxiliary data will be added.
    src : xr.Dataset
        The source dataset containing auxiliary data.
    layout : Any
        Layout object containing file path information.
    subset : dict, optional
        Subset dictionary specifying region of interest.
    config : dict, optional
        Configuration dictionary with variable selection.

    Returns
    -------
    xr.Dataset
        The dataset with auxiliary data added.
    """
    if not config or not config.vars_sel["aux"]:
        return ds

    aux_names = config.vars_sel["aux"]
    aux_names = list(aux_names) if not isinstance(aux_names, str) else [aux_names]
    angle_names = [x for x in aux_names if "angle" in x]
    misc_aux = [x for x in aux_names if x in MISC_AUX_VARS]

    obs_path = str(layout.path).replace("RAD", "OBS")
    down_idx = ds["y"].values
    cross_idx = ds["x"].values
    obs_ds = src.copy()
    obs_ds = obs_ds.sel(y=down_idx, x=cross_idx)

    obs_labels = xr.open_dataset(obs_path, group="sensor_band_parameters")
    band_labels = [x.split(" (")[0] for x in obs_labels.observation_bands.values]

    obs_ds = obs_ds.assign_coords({"bands": band_labels})

    if angle_names:
        ds = add_angles(ds=ds, obs_ds=obs_ds, angle_names=angle_names)

    if misc_aux:
        ds = add_misc(ds=ds, obs_ds=obs_ds, misc_aux=misc_aux)

    if "elev" in aux_names:
        elev_ds = xr.open_dataset(obs_path, group="location")
        elev_ds = elev_ds.sel(downtrack=down_idx, crosstrack=cross_idx)
        ds = add_elev(ds=ds, elev_ds=elev_ds)

    return ds


def add_misc(ds: xr.Dataset, obs_ds: xr.Dataset, misc_aux: list) -> xr.Dataset:
    """
    Add miscellaneous auxiliary data to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The main dataset to which miscellaneous auxiliary data will be added.
    obs_ds : xr.Dataset
        The observation dataset containing auxiliary variables.
    misc_aux : list
        List of miscellaneous auxiliary variable names to add.

    Returns
    -------
    xr.Dataset
        The dataset with miscellaneous auxiliary data added.
    """
    for k in misc_aux:
        vals = obs_ds.sel(bands=k).obs.values if "bands" in obs_ds.coords else obs_ds[k].values
        ds = ds.assign({k: (["y", "x"], vals)})
        ds[k].attrs.update({"standard_name": k, "geometry_id": "60m", "resolution": 60.0})

    return ds


def add_elev(ds: xr.Dataset, elev_ds: xr.Dataset) -> xr.Dataset:
    """
    Add elevation data to the dataset from the elevation dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The main dataset to which elevation data will be added.
    elev_ds : xr.Dataset
        The elevation dataset containing elevation values.

    Returns
    -------
    xr.Dataset
        The dataset with elevation data added.

    Notes
    -----
    Mirrors the behaviour of the former read_elev() method.
    """
    ds = ds.assign({"elev": (["y", "x"], elev_ds["elev"].values)})
    ds["elev"].attrs.update(
        {
            "units": "meters",
            "standard_name": "elevation",
            "geometry_id": "60m",
            "resolution": 60.0,
        }
    )

    return ds
