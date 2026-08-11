"""
eoio.readers.radcalnet.metadata
==============================

Metadata extraction utilities for RadCalNet datasets.

Functions
---------
.. autosummary::
   :toctree: generated/

   get_product_metadata
   get_variable_product_metadata
   get_basic_metadata
"""

import warnings

import xarray as xr
from eoio.readers.base import BaseReader
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.footprint_utils import normalize_footprint

ROI_DEFINITIONS = {
    "BSCN": 300,
    "BTCN": 10,
    "GSCN": 1000,
    "GONA": 500,
    "GHNA": 500,
    "LCFR": 60,
    "RVUS": 1000,
    "ERUS": 130,
}

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


class RadCalNetMetadataExtractor(BaseMetadataExtractor):
    def __init__(self, reader: BaseReader, ds: xr.Dataset):
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
        from dateutil import parser

        lat = self.ds.attrs["Lattitude"]
        lon = self.ds.attrs["Longitude"]
        name = self.ds.attrs["Site"][0:4]
        basic_md = {
            "collection_name": f"{name} {self.ds.attrs['collection']}",
            "collection": self.ds.attrs["collection"],
            "product_name": f"{name}_RadCalNet_L1",
            "platform": "RadCalNet",
            "name": self.ds.attrs["Site"],
            "processing_level": "L1",
            "spatial_resolution": ROI_DEFINITIONS[name],
            "spatial_resolution_units": "m",
            "geometry_ids": "insitu",
            "product_bounds": f"POINT ({lon} {lat})",
            "product_date": parser.parse(str(self.ds.time.values[0])).date(),
            "description": "TBD",
            "eoio:reader": "radcalnet",
            "eoio:subset": repr(self.subset),
            "footprint": normalize_footprint(
                geometry_input=f"POINT ({lon} {lat})",
                crs_input=4326,  # RadCalNet uses WGS84 (EPSG:4326) coordinates
            ),
        }
        return basic_md
