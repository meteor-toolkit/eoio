"""eoio.readers.constellr_skybee.conventions - applies eoio conventions to Skybee datasets."""

from __future__ import annotations

from typing import Any

import xarray as xr

from eoio.readers.constellr_skybee.layout import ConstellrSkybeeLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset


def apply_conventions(
    ds: xr.Dataset,
    *,
    layout: ConstellrSkybeeLayout,
    roi_subset: ResolvedROISubset | None,
    config: Any,
) -> xr.Dataset:
    eoio_subset = roi_subset.clip_box if roi_subset else ""
    eoio_attrs = {"eoio:reader": "constellr_skybee", "eoio:subset": eoio_subset}
    ds.attrs.update(eoio_attrs)
    return ds
