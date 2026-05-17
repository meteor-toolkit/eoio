"""
eoio.readers.hypernets.data_io
==============================

Data input/output utilities for HYPERNETS NetCDF datasets. Provides functions to read and subset data variables for HYPERNETS products.

Functions
---------
.. autosummary::
   :toctree: generated/

   read_dataset
"""

from __future__ import annotations
from typing import List
import xarray as xr
from eoio.readers.hypernets.subset import HYPERNETSSubset


def read_dataset(
    *,
    ds: xr.Dataset,
    include_vars: List[str],
    subset: HYPERNETSSubset,
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
    ds = ds[include_vars]

    id_series, id_wavelength = subset.series_indices, subset.wavelength_indices
    ds = ds.isel(series=id_series, wavelength=id_wavelength)
    return ds


if __name__ == "__main__":
    pass
