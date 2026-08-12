"""eoio.readers.landsat.conventions - Apply conventions to Landsat data."""

from __future__ import annotations
from typing import Any
import xarray as xr

from eoio.readers.landsat.layout import LandsatLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset


def apply_conventions(
    ds: xr.Dataset, *, layout: LandsatLayout, roi_subset: ResolvedROISubset, config: Any
) -> xr.Dataset:
    """
    Apply standard conventions to the Landsat dataset, including naming, attributes, provenance.

    :param ds: xarray.Dataset to modify
    :param layout: Landsat layout object
    :param roi_subset: Subset information
    :param config: Configuration parameters

    :returns: xarray.Dataset with conventions applied.
    """
    roi_subset = config.subset
    if roi_subset is None:
        # "" (not None) -- an attr value of None can't be written to netCDF.
        roi_subset_attr = ""

    else:
        roi_subset_attr = roi_subset.clip_box

    eoio_attrs = {"eoio:reader": "landsat", "eoio:subset": roi_subset_attr}

    ds.attrs.update(eoio_attrs)

    return ds


if __name__ == "__main__":
    pass
