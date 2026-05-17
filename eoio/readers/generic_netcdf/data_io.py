"""
eoio.readers.hypernets.data_io
==============================

Data I/O utilities for HYPERNETS datasets.

Functions
---------
.. autosummary::
   :toctree: generated/

   read_bands_into_dataset
"""

from __future__ import annotations
from typing import List, Optional
import xarray as xr
from eoio.readers.generic_netcdf.subset import GENERIC_NETCDFSubset


def read_dataset(
    *,
    ds: xr.Dataset,
    include_vars: Optional[List[str]] = None,
    subset: Optional[GENERIC_NETCDFSubset] = None,
) -> xr.Dataset:
    """
    Read selected bands into an xarray.Dataset and apply subsetting.

    Parameters
    ----------
    ds
        The input dataset.
    include_vars
        Variables to include in the output dataset.
    subset
        Subsetting information containing series and wavelength indices.

    Returns
    -------
    xarray.Dataset
        The subsetted dataset containing only the selected variables.
    """

    # only keep variables in include_vars, drop the rest
    if include_vars is not None:
        ds = ds[include_vars]

    if subset is not None:
        id_wavelength = subset.wavelength_indices
        if id_wavelength is not None:
            ds = ds.isel(wavelength=id_wavelength)

    reorder = []
    if "wavelength" in ds.dims:
        reorder.append("wavelength")
    if "y" in ds.dims:
        reorder.append("y")
    if "x" in ds.dims:
        reorder.append("x")
    for d in ds.dims:
        if d not in reorder:
            reorder.append(str(d))
    if reorder != list(ds.dims):
        ds = ds.transpose(*reorder, ...)

    return ds


if __name__ == "__main__":
    pass
