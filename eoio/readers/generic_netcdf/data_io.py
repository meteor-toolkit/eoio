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
from typing import Any, List, Optional
import xarray as xr


def read_dataset(
    *,
    ds: xr.Dataset,
    include_vars: Optional[List[str]] = None,
    subset: Optional[Any] = None,
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
        Resolved subsetting information. Duck-typed rather than tied to one
        dataclass: each reader owns its own subset type (e.g.
        ``GENERIC_NETCDFSubset``, ``ECMWFSubset``) and only the attributes
        present are applied, so a reader that resolves no ROI or no datetime
        range simply omits those fields.

    Returns
    -------
    xarray.Dataset
        The subsetted dataset containing only the selected variables.
    """

    # only keep variables in include_vars, drop the rest
    if include_vars is not None:
        ds = ds[include_vars]

    if subset is not None:
        if hasattr(subset, "wavelength_indices") and subset.wavelength_indices is not None:
            ds = ds.isel(wavelength=subset.wavelength_indices)
        if hasattr(subset, "datetime_indices") and subset.datetime_indices is not None:
            ds = ds.isel(datetime=subset.datetime_indices)
        roi_subset = getattr(subset, "roi_subset", None)
        if roi_subset is not None and roi_subset.geometries is not None:
            ds = ds.rio.clip(roi_subset.geometries)

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
