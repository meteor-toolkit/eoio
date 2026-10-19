"""
eoio.readers.flex.data_io
=========================

Data I/O for FLEX L2 FLXSYN products.

Functions
---------
.. autosummary::
   :toctree: generated/

   read_grid
   read_wavelengths
   read_dataset
   register_uncertainties
   err_corr_for
"""

from __future__ import annotations
import warnings
from typing import Dict, Iterable, List, Optional

import numpy as np
import obsarray  # noqa: F401 -- registers the `ds.unc` accessor used below
import xarray as xr

from eoio.readers.flex.grid import FlexGrid
from eoio.readers.flex.subset import FLEXSubset
from eoio.readers.flex.variables import (
    ANC,
    INSTRUMENTS,
    O2_BANDS,
    SIF_PEAKS,
    SPATIAL_DIMS,
    UNCERTAINTY_SUFFIX,
    VARIABLES,
    WAVELENGTH_COORDS,
    VarSpec,
)

__all__ = ["read_grid", "read_wavelengths", "read_dataset", "register_uncertainties", "err_corr_for"]

_LABEL_COORDS = {"instrument": INSTRUMENTS, "sif_peak": SIF_PEAKS, "o2_band": O2_BANDS}


#: default dask chunk sizes along y and x for lazy reading; spectral axes are kept whole
#: (about 10 MB per chunk for the 580-channel FLORIS cubes; larger chunks raised peak memory
#: with several decoded in parallel)
DEFAULT_CHUNKS = {"y": 64, "x": 64}


def _open_group(path: str, group: str, chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
    # mask_and_scale unpacks the packed integer cubes to float32 and turns _FillValue into NaN
    if chunks is None:
        return xr.open_dataset(path, group=group, mask_and_scale=True)
    # chunks use eoio dimension names; the file has its own names for the spatial ones
    inverse = {v: k for k, v in SPATIAL_DIMS.items()}
    return xr.open_dataset(
        path, group=group, mask_and_scale=True, chunks={inverse.get(k, k): v for k, v in chunks.items()}
    )


def read_grid(path: str) -> FlexGrid:
    """
    Read the product grid from the global attributes and the ancillary group.

    :param path: path to the FLEX product.
    :returns: the product grid.
    """
    with xr.open_dataset(path) as root:
        attrs = dict(root.attrs)
    with _open_group(path, ANC) as anc:
        ny = anc.sizes["number_of_northing_pixels"]
        nx = anc.sizes["number_of_easting_pixels"]
    return FlexGrid.from_attrs(attrs, nx=nx, ny=ny)


def read_wavelengths(path: str) -> Dict[str, xr.DataArray]:
    """
    Read the wavelength coordinate for every spectral dimension, as stored in the product.

    FLORIS wavelengths are stored per across-track sample (``across_track x channel``), but only
    the first sample is populated in the products seen so far. They are collapsed to a single
    wavelength vector (mean over the populated samples); a warning is raised if populated samples
    differ, since a single vector then no longer describes the product.
    This is provisional: a spectral smile may require per-row wavelengths, to be revisited with
    real products (it is expected to be corrected in L1).

    :param path: path to the FLEX product.
    :returns: mapping from dimension name (e.g. ``"wavelength_olci"``) to a 1-D wavelength
        ``DataArray`` in nm.
    """
    out: Dict[str, xr.DataArray] = {}
    groups: Dict[str, xr.Dataset] = {}
    try:
        for dim, (group, name) in WAVELENGTH_COORDS.items():
            if group not in groups:
                groups[group] = _open_group(path, group)
            da = groups[group][name]
            values = np.asarray(da.values, dtype=float)
            if values.ndim == 2:
                values = _collapse_wavelengths(values, name)
            out[dim] = xr.DataArray(
                values,
                dims=[dim],
                attrs={
                    "units": "nm",
                    "long_name": da.attrs.get("long_name", name),
                    "standard_name": "radiation_wavelength",
                },
            )
    finally:
        for g in groups.values():
            g.close()
    return out


def _collapse_wavelengths(values: np.ndarray, name: str) -> np.ndarray:
    populated = values[~np.isnan(values).all(axis=1)]
    if populated.shape[0] == 0:
        raise ValueError(f"No populated wavelengths in {name}")
    if not np.allclose(populated, populated[0], equal_nan=True):
        warnings.warn(
            f"{name} differs between across-track samples; using the mean over {populated.shape[0]} "
            "populated samples as a single wavelength vector"
        )
    return populated.mean(axis=0)


def _spec_for(name: str) -> tuple[str, VarSpec]:
    """Base-variable name and spec for a variable or its uncertainty partner."""
    base = name[: -len(UNCERTAINTY_SUFFIX)] if name.endswith(UNCERTAINTY_SUFFIX) else name
    if base not in VARIABLES:
        raise ValueError(f"Unknown FLEX variable: {name}")
    if base != name and not VARIABLES[base].has_uncertainty:
        raise ValueError(f"FLEX variable {base} has no uncertainty partner: {name}")
    return base, VARIABLES[base]


def err_corr_for(dims: Iterable[str]) -> List[dict]:
    """
    Assumed error-correlation structure of an uncertainty component.

    PLACEHOLDER: the product documents no correlation model. Until one is confirmed, uncertainties
    are assumed random across the spatial dimensions and fully systematic along any spectral
    dimension.

    :param dims: dimensions of the uncertainty variable.
    :returns: obsarray ``err_corr`` definition.
    """
    out = [{"dim": ["y", "x"], "form": "random", "params": [], "units": []}]
    out += [{"dim": [d], "form": "systematic", "params": [], "units": []} for d in dims if d.startswith("wavelength_")]
    return out


def read_dataset(
    path: str,
    include_vars: Iterable[str],
    grid: FlexGrid,
    wavelengths: Dict[str, xr.DataArray],
    subset: Optional[FLEXSubset] = None,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    Read the requested variables (and their uncertainty partners, if named) into a dataset.

    Dimensions are renamed to ``y``/``x`` plus one named dimension per spectral/trailing axis,
    packed integers are unpacked, and coordinates (x, y, latitude, longitude, wavelengths, labels)
    are attached. Subsetting is applied before data are loaded.

    :param path: path to the FLEX product.
    :param include_vars: variable names to read, including any ``*_uncertainty`` partners.
    :param grid: product grid.
    :param wavelengths: wavelength coordinates, as returned by :py:func:`read_wavelengths`.
    :param subset: resolved subset, or ``None`` for the full product.
    :param chunks: dask chunk sizes along ``y``/``x``. If given, variables stay lazy (dask-backed,
        file left open) after subsetting; otherwise they are loaded into memory.
    :returns: dataset with the requested variables.
    """
    subset = subset or FLEXSubset()
    by_group: Dict[str, List[str]] = {}
    for name in include_vars:
        _, spec = _spec_for(name)
        by_group.setdefault(spec.group, []).append(name)

    data_vars: Dict[str, xr.DataArray] = {}
    dropped: set = set()
    for group, names in by_group.items():
        src = _open_group(path, group, chunks)
        try:
            for name in names:
                base, spec = _spec_for(name)
                if base in dropped:
                    continue
                da = _read_variable(src, name, spec, subset, lazy=chunks is not None)
                if da is None:
                    warnings.warn(f"{base}: no data left after subsetting; variable omitted")
                    dropped.add(base)
                    continue
                data_vars[name] = da
        finally:
            if chunks is None:
                src.close()
    # drop any uncertainty partner read before its base variable was found empty
    data_vars = {k: v for k, v in data_vars.items() if _spec_for(k)[0] not in dropped}

    ds = xr.Dataset(data_vars)
    return _attach_coords(ds, path, grid, wavelengths, subset)


def _read_variable(src: xr.Dataset, name: str, spec: VarSpec, subset: FLEXSubset, lazy: bool) -> Optional[xr.DataArray]:
    """Read one variable from its group; ``None`` if the subset leaves it empty."""
    da = src[name]
    dim_map = dict(SPATIAL_DIMS)
    if spec.dim is not None:
        dim_map[str(da.dims[2])] = spec.dim
    da = da.rename(dim_map)
    isel = _isel_for(spec, subset)
    if any(len(v) == 0 for v in isel.values()):
        return None
    da = da.isel(isel)
    if not lazy:
        da = da.load()
    if da.dtype == np.float64 and np.dtype(da.encoding.get("dtype", "f8")).kind in "iu":
        # packed 16-bit integers: float32 holds them exactly enough, halving memory
        da = da.astype(np.float32)
    if spec.dim is not None:
        da = da.transpose(spec.dim, "y", "x")
    da.attrs.setdefault("standard_name", name)
    return da


def _isel_for(spec: VarSpec, subset: FLEXSubset) -> Dict[str, np.ndarray]:
    isel: Dict[str, np.ndarray] = {}
    if subset.y_indices is not None:
        isel["y"] = subset.y_indices
    if subset.x_indices is not None:
        isel["x"] = subset.x_indices
    if spec.dim in subset.wavelength_indices:
        isel[spec.dim] = subset.wavelength_indices[spec.dim]
    return isel


def _attach_coords(
    ds: xr.Dataset, path: str, grid: FlexGrid, wavelengths: Dict[str, xr.DataArray], subset: FLEXSubset
) -> xr.Dataset:
    y_idx = subset.y_indices if subset.y_indices is not None else slice(None)
    x_idx = subset.x_indices if subset.x_indices is not None else slice(None)
    x = xr.DataArray(
        grid.x[x_idx],
        dims="x",
        attrs={"standard_name": "projection_x_coordinate", "long_name": "Easting", "units": "m"},
    )
    y = xr.DataArray(
        grid.y[y_idx],
        dims="y",
        attrs={"standard_name": "projection_y_coordinate", "long_name": "Northing", "units": "m"},
    )
    ds = ds.assign_coords(x=x, y=y)

    with _open_group(path, ANC) as anc:
        loc = anc[["latitude", "longitude"]].rename(SPATIAL_DIMS).isel(y=y_idx, x=x_idx).load()
    ds = ds.assign_coords(latitude=loc["latitude"], longitude=loc["longitude"])

    for dim, wvl in wavelengths.items():
        if dim in ds.dims:
            idx = subset.wavelength_indices.get(dim)
            ds = ds.assign_coords({dim: wvl.isel({dim: idx}) if idx is not None else wvl})
    for dim, labels in _LABEL_COORDS.items():
        if dim in ds.dims:
            ds = ds.assign_coords({dim: labels})
    return ds


def register_uncertainties(ds: xr.Dataset) -> xr.Dataset:
    """
    Register every ``<var>_uncertainty`` variable as an obsarray uncertainty component of ``<var>``.

    This makes them discoverable via ``ds.unc[var]`` with machine-readable error-correlation
    metadata (see :py:func:`err_corr_for` for the assumptions made).

    :param ds: dataset as returned by :py:func:`read_dataset`.
    :returns: dataset with uncertainty variables registered.
    """
    for name in [str(v) for v in ds.data_vars if str(v).endswith(UNCERTAINTY_SUFFIX)]:
        base = name[: -len(UNCERTAINTY_SUFFIX)]
        if base not in ds:
            warnings.warn(f"{name} has no matching variable {base}; left as an ordinary variable")
            continue
        unc = ds[name]
        attrs = {**unc.attrs, "err_corr": err_corr_for([str(d) for d in unc.dims])}
        ds = ds.drop_vars(name)
        ds.unc[base][name] = (unc.dims, unc.data, attrs)
    return ds


if __name__ == "__main__":
    pass
