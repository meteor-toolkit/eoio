"""
eoio.readers.hypernets.metadata
==============================

Metadata extraction utilities for HYPERNETS datasets.

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
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.footprint_utils import normalize_footprint

min_basic_var_metadata_keys = [
    "long_name",
    "standard_name",
    "units",
]

optional_basic_var_metadata_keys = [
    "history",
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
    "err_corr_3_dim",
    "err_corr_3_form",
    "err_corr_3_units",
    "err_corr_3_params",
    "err_corr_4_dim",
    "err_corr_4_form",
    "err_corr_4_units",
    "err_corr_4_params",
    "pdf_shape",
    "add_offset",
    "scale_factor",
    "flag_meanings",
    "flag_values",
    "flag_masks",
]


class GenericNetCDFMetadataExtractor(BaseMetadataExtractor):
    def __init__(self, reader, ds: xr.Dataset):
        super().__init__(reader)
        self.ds = ds.copy()
        self.subset = reader.config.subset

    def get_product_metadata(self) -> dict:
        """
        Extract product metadata from the dataset.

        :returns: Product metadata dictionary.
        """
        md = self.ds.attrs.copy()
        return md

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the dataset.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
        """
        var_md = self.ds[var].attrs.copy()
        for key in optional_basic_var_metadata_keys:
            var_md.pop(key, None)
        for key in min_basic_var_metadata_keys:
            var_md.pop(key, None)
        return var_md

    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the dataset.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
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

        :returns: Basic metadata dictionary.
        """
        basic_md = {
            "collection_name": self.ds.attrs.get("collection_name", ""),
            "product_name": self.ds.attrs.get("product_name", ""),
            "platform": self.ds.attrs.get("platform", ""),
            "name": self.ds.attrs.get("name", ""),
            "processing_level": self.ds.attrs.get("processing_level", ""),
            "spatial_resolution": self.ds.attrs.get("spatial_resolution", ""),
            "geometry_ids": self.ds.attrs.get("geometry_ids", ""),
            "product_bounds": self.ds.attrs.get("product_bounds", ""),
            "product_date": self.ds.attrs.get("product_date", ""),
            "product_datetime": self.ds.attrs.get("product_datetime", ""),
            "description": self.ds.attrs.get("description", ""),
            "eoio:reader": "generic_netcdf",
            # json.dumps (not repr) keeps this parseable JSON, matching the eoio:subset
            # convention used elsewhere -- repr() produces Python-only syntax (e.g. single
            # quotes) that isn't valid JSON. default=str covers subset values that aren't
            # natively JSON-serialisable (e.g. a shapely ROI geometry).
            "eoio:subset": json.dumps(self.subset, default=str) if self.subset else "",
            "footprint": normalize_footprint(
                geometry_input=self.ds.attrs.get("product_bounds"),
                crs_input=self.ds.attrs.get("footprint_crs", None),
            ),
        }
        if "history" in self.ds.attrs:
            basic_md["history"] = self.ds.attrs["history"]
        return basic_md
