"""
eoio.readers.hypernets.reader
=============================

Sentinel-2 MSI data reader implementation for HYPERNETS.

Classes
-------
.. autosummary::
   :toctree: generated/

   HYPERNETSReader

Functions
---------
.. autosummary::
   :toctree: generated/
"""

from __future__ import annotations
import xarray as xr
from eoio.readers.generic_netcdf.subset import build_subset
from eoio.readers.generic_netcdf.data_io import read_dataset

# from eoio.readers.hypernets.aux import maybe_add_aux
from eoio.readers.generic_netcdf.metadata import GenericNetCDFMetadataExtractor
from eoio.readers.base import BaseReader


class NetCDFReader(BaseReader):
    """
    Reader for generic NetCDF data.
    """

    def __init__(self, path, vars_sel=None, subset=None, read_params=None):
        super().__init__(path, vars_sel, subset, read_params)
        self.ds_src = xr.open_dataset(self.path)
        self.meas_def = {
            "all": list(self.ds_src.variables),
        }

    def open_dataset(self) -> xr.Dataset:
        """
        Open the HYPERNETS dataset as an xarray.Dataset according to the request parameters.

        :returns: The opened and subsetted HYPERNETS dataset.
        """

        # open data
        ds = self.ds_src.copy()

        # Build the subset
        subset = build_subset(
            ds,
            subset=self.config.subset,
        )

        # list variables to include
        include_vars = self.list_include_vars()

        # Read image data if requested
        ds = read_dataset(
            ds=ds,
            include_vars=include_vars,
            subset=subset,
        )

        # attach metadata
        mtd_level = self.config.read_params.get("metadata_level", None)
        if mtd_level is True:
            mtd_level = "all"

        if mtd_level == "original":
            return ds
        else:
            meta_ex = GenericNetCDFMetadataExtractor(self, ds)
            ds = meta_ex.clear_metadata(ds)
            if mtd_level in ("all", "basic"):
                ds = meta_ex.attach_metadata(ds, level=mtd_level)
            return ds

    @staticmethod
    def get_extension() -> str:
        """
        Return file extension for extracted files for this product.

        :returns: File extension as string.
        """
        return ".nc"


if __name__ == "__main__":
    pass
