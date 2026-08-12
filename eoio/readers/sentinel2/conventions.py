"""eoio.readers.sentinel2.conventions - applies eoio conventions to Sentinel2 dataset"""

from __future__ import annotations
from typing import Any
import xarray as xr
from eoio.readers.sentinel2.layout import S2Layout


def apply_conventions(ds: xr.Dataset, *, layout: S2Layout, config: Any) -> xr.Dataset:

    roi_subset = config.subset
    if roi_subset is None:
        # "" (not None) -- an attr value of None can't be written to netCDF.
        roi_subset_attr = ""

    else:
        roi_subset_attr = roi_subset.clip_box

    eoio_attrs = {"eoio:reader": "sentinel2", "eoio:subset": roi_subset_attr}

    ds.attrs.update(eoio_attrs)

    return ds


if __name__ == "__main__":
    pass
