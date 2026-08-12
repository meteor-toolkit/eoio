"""
eoio.readers.EMIT.metadata
===============================

Metadata extraction utilities for EMIT datasets.

Functions
---------
.. autosummary::
   :toctree: generated/

   get_product_metadata
   get_variable_product_metadata
   get_basic_metadata
"""

import json
import warnings

import xarray as xr
import os

from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.footprint_utils import normalize_footprint

# from processor_tools.utils.formatters import convert_datetime

min_basic_var_metadata_keys = [
    "long_name",
    "standard_name",
    "units",
]

optional_basic_var_metadata_keys = [
    "_FillValue",
    "unc_comps",
    "err_corr_1_dim",
    "err_corr_1_form",
    "err_corr_1_units",
    "err_corr_1_params",
    "err_corr_2_dim",
    "err_corr_2_form",
    "err_corr_2_units",
    "err_corr_2_params",
    "pdf_shape",
    "add_offset",
    "scale_factor",
    "flag_meanings",
    "flag_values",
    "flag_masks",
]


class EMITMetadataExtractor(BaseMetadataExtractor):
    def __init__(self, reader, ds: xr.Dataset, layout):
        super().__init__(reader)
        self.ds = ds.copy()
        self.layout = layout
        self.subset = reader.config.subset

    def get_product_metadata(self) -> dict:
        """
        Extract product metadata from the dataset.

        Returns
        -------
        dict
            Product metadata dictionary.
        """
        md = self.ds.attrs.copy()
        return md

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the dataset.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        dict
            Variable metadata dictionary.
        """
        var_md = self.ds[var].attrs.copy()
        for key in optional_basic_var_metadata_keys:
            var_md.pop(key, None)
        for key in min_basic_var_metadata_keys:
            var_md.pop(key, None)
        return var_md

    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Extract basic variable metadata from the dataset.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        dict
            Basic variable metadata dictionary.
        """
        var_md = {}
        for key in min_basic_var_metadata_keys:
            if key in self.ds[var].attrs:
                var_md[key] = self.ds[var].attrs[key]
            else:
                warnings.warn(f"Variable {var} missing expected metadata key: {key}")
                var_md[key] = ""
        for key in optional_basic_var_metadata_keys:
            if key in self.ds[var].attrs:
                var_md[key] = self.ds[var].attrs[key]
        return var_md

    def get_basic_metadata(self) -> dict:
        """
        Extract basic metadata from the dataset, to be used in other tools in MetEOR.

        Returns
        -------
        dict
            Basic metadata dictionary.
        """
        basic_md = {
            "collection_name": os.path.split(self.layout.path)[-1][0:8],
            "product_name": self.reader.ds_src.title,  # type: ignore[attr-defined]
            "platform": "ISS",
            "name": "EMIT",
            "processing_level": int(os.path.split(self.layout.path)[-1].split("_")[1][1]),
            "spatial_resolution": 60,
            "geometry_ids": "60m",
            "product_bounds": self.subset["roi"],
            "product_date": self.reader.ds_src.time_coverage_end.split("T")[0],  # type: ignore[attr-defined]
            "description": self.reader.ds_src.summary,  # type: ignore[attr-defined]
            "eoio:reader": "emit",
            # json.dumps (not repr) keeps this parseable JSON, matching the eoio:subset
            # convention used elsewhere -- repr() produces Python-only syntax (e.g. single
            # quotes) that isn't valid JSON. default=str covers subset values that aren't
            # natively JSON-serialisable (e.g. a shapely ROI geometry).
            "eoio:subset": json.dumps(self.subset, default=str) if self.subset else "",
            "footprint": normalize_footprint(
                geometry_input=self.subset.get("roi") if self.subset else None,
                crs_input=None,  # EMIT CRS not readily available in subset/metadata
            ),
        }
        if "history" in self.ds.attrs:
            basic_md["history"] = self.ds.attrs["history"]
        return basic_md
