"""eoio.readers.airbus_pleiades.conventions - applies eoio conventions to Airbus Pleiades dataset"""

from __future__ import annotations
from typing import Any
import xarray as xr
from eoio.readers.airbus_pleiades.layout import PleiadesLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset


def apply_conventions(
    ds: xr.Dataset,
    *,
    layout: PleiadesLayout,
    roi_subset: ResolvedROISubset,
    config: Any,
) -> xr.Dataset:
    """
    Apply standard conventions to the Airbus Pleiades dataset, including naming, attributes, provenance.

    :param ds: xarray.Dataset to modify
    :param layout: Airbus Pleiades layout object
    :param roi_subset: Subset information
    :param config: Configuration parameters

    :returns: xarray.Dataset with conventions applied.
    """
    # "" (not None) -- an attr value of None can't be written to netCDF.
    eoio_subset = roi_subset.clip_box if roi_subset else ""
    eoio_attrs = {"eoio:reader": "airbus_pleiades", "eoio:subset": eoio_subset}

    ds.attrs.update(eoio_attrs)

    return ds


if __name__ == "__main__":
    pass
