"""
eoio.readers.hypernets.metadata
===============================

Metadata extraction utilities for HYPERNETS NetCDF datasets. Provides classes and functions to extract and attach product and variable metadata to HYPERNETS xarray datasets.

Classes
-------
.. autosummary::
   :toctree: generated/

   HYPERNETSMetadataExtractor

Functions
---------
.. autosummary::
   :toctree: generated/

   get_product_metadata
   get_variable_metadata
   get_basic_metadata
"""

import xarray as xr
import numpy as np
from eoio.readers.metadata import BaseMetadataExtractor
from processor_tools.utils.formatters import convert_datetime
from eoio.readers.footprint_utils import normalize_footprint

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
    "flag_masks",
]


class HYPERNETSMetadataExtractor(BaseMetadataExtractor):
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

    def get_variable_metadata(self, var: str) -> dict:
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
                Warning(f"Variable {var} missing expected metadata key: {key}")
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
        lat = self.ds.attrs["site_latitude"]
        lon = self.ds.attrs["site_longitude"]
        date = convert_datetime(np.nanmean(self.ds.acquisition_time.values)).date()
        basic_md = {
            "collection_name": self.ds.attrs["site_id"],
            "product_name": self.ds.attrs["product_name"],
            "platform": "HYPERNETS",
            "name": self.ds.attrs["site_id"],
            "processing_level": self.ds.attrs["product_level"],
            "spatial_resolution": "NA",
            "geometry_ids": "insitu",
            "product_bounds": f"POINT ({lon} {lat})",
            "product_date": str(date),
            "description": "TBD",
            "eoio:reader": "hypernets",
            "eoio:subset": repr(self.subset),
            "footprint": normalize_footprint(
                geometry_input=f"POINT ({lon} {lat})",
                crs_input=4326,  # HYPERNETS uses WGS84 (EPSG:4326) coordinates
            ),
        }
        if "history" in self.ds.attrs:
            basic_md["history"] = self.ds.attrs["history"]
        return basic_md
