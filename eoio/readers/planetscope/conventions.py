"""eoio.readers.planetscope.conventions - applies eoio conventions to PlanetScope dataset"""

from __future__ import annotations
from typing import Any
import xarray as xr
from eoio.readers.planetscope.layout import PlanetScopeLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset


def apply_conventions(
    ds: xr.Dataset,
    *,
    layout: PlanetScopeLayout,
    roi_subset: ResolvedROISubset,
    config: Any,
) -> xr.Dataset:
    """
    Apply standard conventions to the PlanetScope dataset, including naming, attributes, provenance.

    :param ds: xarray.Dataset to modify
    :param layout: PlanetScope layout object
    :param roi_subset: Subset information
    :param config: Configuration parameters

    :returns: xarray.Dataset with conventions applied.
    """
    # "" (not None) -- an attr value of None can't be written to netCDF.
    eoio_subset = roi_subset.clip_box if roi_subset else ""

    eoio_attrs = {"eoio:reader": "planetscope", "eoio:subset": eoio_subset}

    ds.attrs.update(eoio_attrs)

    return ds


if __name__ == "__main__":
    pass
