"""
eoio.readers.flex.metadata
==========================

Metadata extraction for FLEX datasets.

Classes
-------
.. autosummary::
   :toctree: generated/

   FLEXMetadataExtractor
"""

from __future__ import annotations
import datetime as dt
import json
import warnings
from typing import Any, Optional

import xarray as xr

from eoio.deps import lazy_pyproj, lazy_shapely
from eoio.readers.base import BaseReader
from eoio.readers.flex.grid import FlexGrid
from eoio.readers.flex.layout import FlexLayout
from eoio.readers.footprint_utils import normalize_footprint
from eoio.readers.metadata import BaseMetadataExtractor

min_basic_var_metadata_keys = [
    "long_name",
    "standard_name",
    "units",
]

optional_basic_var_metadata_keys = [
    "_FillValue",
    "unc_comps",
    *[f"err_corr_{i}_{k}" for i in range(1, 5) for k in ("dim", "form", "units", "params")],
    "pdf_shape",
    "add_offset",
    "scale_factor",
    "flag_meanings",
    "flag_values",
    "flag_masks",
]

#: global attributes that, if present in a product, take precedence over the filename start time
_START_TIME_ATTRS = ("time_coverage_start", "sensing_start_time", "start_time")


class FLEXMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for FLEX L2 products.

    :param reader: the reader the dataset was read with.
    :param ds: the dataset (carrying the product's global attributes).
    :param layout: parsed product filename.
    :param grid: product grid.
    """

    def __init__(self, reader: BaseReader, ds: xr.Dataset, layout: FlexLayout, grid: FlexGrid):
        super().__init__(reader)
        self.ds = ds.copy()
        self.layout = layout
        self.grid = grid
        self.subset = reader.config.subset

    def get_product_metadata(self) -> dict:
        """
        Extract product metadata (the product's global attributes).

        :returns: Product metadata dictionary.
        """
        return dict(self.ds.attrs)

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Extract the variable attributes not already covered by the basic metadata.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
        """
        var_md = dict(self.ds[var].attrs)
        for key in optional_basic_var_metadata_keys + min_basic_var_metadata_keys:
            var_md.pop(key, None)
        return var_md

    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Extract basic variable metadata.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
        """
        attrs = self.ds[var].attrs
        var_md = {}
        for key in min_basic_var_metadata_keys:
            if key in attrs:
                var_md[key] = attrs[key]
            else:
                warnings.warn(f"Variable {var} missing expected metadata key: {key}")
                var_md[key] = ""
        for key in optional_basic_var_metadata_keys:
            if key in attrs:
                var_md[key] = attrs[key]
        return var_md

    def _product_datetime(self) -> Optional[dt.datetime]:
        """
        Sensing start: an explicit product attribute if present, otherwise the filename.

        The products seen so far carry no acquisition time, so this is the filename start time;
        whether a start/stop range is needed is still to be confirmed.
        """
        from dateutil import parser

        for key in _START_TIME_ATTRS:
            if key in self.ds.attrs:
                try:
                    value = parser.parse(str(self.ds.attrs[key]))
                except (ValueError, OverflowError):
                    continue
                return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        return self.layout.start_time

    def _extent(self) -> Any:
        """Extent of the (possibly subsetted) data in the product CRS, as a shapely box."""
        _, box, _, _, _ = lazy_shapely()

        if "x" in self.ds.coords and "y" in self.ds.coords:
            x, y = self.ds["x"].values, self.ds["y"].values
            return box(
                float(x.min()) - self.grid.dx / 2,
                float(y.min()) - self.grid.dy / 2,
                float(x.max()) + self.grid.dx / 2,
                float(y.max()) + self.grid.dy / 2,
            )
        return box(*self.grid.bounds)

    def get_basic_metadata(self) -> dict:
        """
        Extract basic metadata, to be used in other tools in MetEOR.

        :returns: Basic metadata dictionary.
        """
        _, _, _, _, transform = lazy_shapely()

        extent = self._extent()
        to_wgs84 = lazy_pyproj().Transformer.from_crs(self.grid.crs, "EPSG:4326", always_xy=True).transform
        basic_md = {
            "collection_name": f"FLEX {self.layout.level} {self.layout.product_type}",
            "product_name": self.layout.name,
            "platform": "FLEX",
            "name": "FLEX",
            "processing_level": self.layout.processing_level,
            "spatial_resolution": int(self.grid.dx) if float(self.grid.dx).is_integer() else self.grid.dx,
            "spatial_resolution_units": "m",
            "geometry_ids": f"{self.grid.dx:g}m",
            "product_bounds": transform(to_wgs84, extent).wkt,
            "crs": self.grid.crs,
            "product_datetime": self._product_datetime(),
            "description": "FLEX synergy product combining FLORIS, OLCI and SLSTR",
            "eoio:reader": "flex",
            # json.dumps (not repr) keeps this parseable JSON; default=str covers non-serialisable
            # values such as a shapely ROI geometry.
            "eoio:subset": json.dumps(self.subset, default=str) if self.subset else "",
            "footprint": normalize_footprint(geometry_input=extent, crs_input=self.grid.crs),
        }
        if self.ds.attrs.get("History"):
            basic_md["history"] = self.ds.attrs["History"]
        return basic_md


if __name__ == "__main__":
    pass
