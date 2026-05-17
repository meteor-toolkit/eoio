"""eoio.readers.sentinel2.aux_vars.masks - mask data reading functions"""

from __future__ import annotations
from typing import Any
from eoio.readers.sentinel2.layout import S2Layout
import xarray as xr


def add_masks(*, ds: xr.Dataset, var_names: list[str], layout: S2Layout, config: Any) -> xr.Dataset:

    return ds


if __name__ == "__main__":
    pass
