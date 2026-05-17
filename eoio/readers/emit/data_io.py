"""Image I/O helpers for EMIT based on the logic from the original reader."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from eoio.readers.emit.subset import EMITSubset
import numpy as np
import xarray as xr
import warnings


def attach_coords_from_groups(ds: xr.DataArray, path: str) -> xr.DataArray:
    """
    Attach wavelength and location coordinates from groups in the EMIT file.

    Parameters
    ----------
    ds : xr.DataArray
        The data array to which coordinates will be attached.
    path : str
        Path to the EMIT file.

    Returns
    -------
    xr.DataArray
        The data array with attached coordinates.
    """
    try:
        with xr.open_dataset(path, group="sensor_band_parameters").rename_vars({"wavelengths": "wavelength"}) as wvl:
            if "bands" in wvl.dims:
                ds = ds.assign_coords(**{str(k): (v.dims, v.data) for k, v in wvl.variables.items() if k in wvl})  # type: ignore[arg-type]
    except Exception:
        wvl = None

    try:
        with xr.open_dataset(path, group="location").rename_vars({"lat": "latitude", "lon": "longitude"}) as loc:
            if loc is not None:
                coords = {}
                for name in ("latitude", "longitude"):
                    if name in loc:
                        v = loc[name]
                        coords[name] = (v.dims, v.data)
                if coords:
                    ds = ds.assign_coords(**coords)  # type: ignore[arg-type]
    except Exception:
        loc = None

    if "bands" in ds.dims and "wavelength" in ds.coords:
        ds = ds.swap_dims({"bands": "wavelength"})
    if "crosstrack" in ds.dims:
        ds = ds.rename({"crosstrack": "x"})
    if "downtrack" in ds.dims:
        ds = ds.rename({"downtrack": "y"})
    return ds


def read_dataset(
    ds: xr.Dataset,
    src: xr.Dataset,
    meas_vars: Optional[Sequence[str]] = None,
    subset: "Optional[EMITSubset]" = None,
) -> xr.Dataset:
    """
    Read EMIT radiance dataset from the L1B file and return an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The target dataset to populate.
    src : xr.Dataset
        The source dataset containing radiance data.
    meas_vars : Sequence[str], optional
        List of measurement variable names to include.
    subset : dict, optional
        Subset dictionary specifying region of interest.

    Returns
    -------
    xr.Dataset
        The populated dataset with radiance data.

    Notes
    -----
    This function implements the essential transformations present in the original monolithic reader.
    """
    # perform roi subsetting
    roi = subset.roi if subset is not None else None
    if roi is not None:
        try:
            if roi.clip_box is None:
                raise ValueError("clip_box is None")
            minx, miny, maxx, maxy = roi.clip_box
            lat = src["latitude"]
            lon = src["longitude"]
            mask = (lat >= miny) & (lat <= maxy) & (lon >= minx) & (lon <= maxx)
            if mask.any():
                idx = np.where(mask)
                if idx[0].size and idx[1].size:
                    src = src.isel(y=np.unique(idx[0]), x=np.unique(idx[1]), drop=True)
        except Exception:
            warnings.warn("The ROI subsetting failed so was not applied")
            pass

    # reorder dimensions
    reorder = []
    if "wavelength" in src.dims:
        reorder.append("wavelength")
    if "y" in src.dims:
        reorder.append("y")
    if "x" in src.dims:
        reorder.append("x")
    for d in src.dims:
        if d not in reorder:
            reorder.append(str(d))
    src = src.transpose(*reorder, ...)

    # select relevant variables
    for meas_var in meas_vars or []:
        ds[meas_var] = src[meas_var]

    # perform wavelength subsetting
    if subset is not None and subset.wavelength_indices is not None:
        ds = ds.isel(wavelength=subset.wavelength_indices)

    return ds
