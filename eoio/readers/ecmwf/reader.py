"""
eoio.readers.ecmwf.reader
=========================

ECMWF data reader implementation.

Classes
-------
.. autosummary::
   :toctree: generated/

   ECMWFReader

Functions
---------
.. autosummary::
   :toctree: generated/
"""

from __future__ import annotations
import xarray as xr
import os.path
from eoio.readers.ecmwf.subset import build_subset
from eoio.readers.generic_netcdf.data_io import read_dataset

# from eoio.readers.hypernets.aux import maybe_add_aux
from eoio.readers.generic_netcdf.metadata import GenericNetCDFMetadataExtractor
from eoio.readers.generic_netcdf.reader import NetCDFReader


class ECMWFReader(NetCDFReader):
    """
    Reader for generic NetCDF data.
    """

    default_vars_sel = {
        "meas": None,  # e.g. []
        "mask": None,  # e.g. ["cloud"]
        "aux": None,  # e.g. ["viewing_zenith_angle"]
    }

    default_subset = {
        "roi": None,
        "roi_crs": 4326,
        "datetime": [
            "min",
            "max",
            "nearest",
            "tolerance_days",
            "tolerance_hours",
            "tolerance_minutes",
        ],
    }

    default_read = {"save_extracted": False, "metadata_level": "all"}

    meas_def = {
        "all": [],
    }

    def __init__(self, path, vars_sel=None, subset=None, read_params=None):
        if (not os.path.exists(path)) and os.path.exists(
            os.path.join(os.path.dirname(path), "data_stream-oper_stepType-instant.nc")
        ):
            path = os.path.join(os.path.dirname(path), "data_stream-oper_stepType-instant.nc")
        if (not os.path.exists(path)) and os.path.exists(os.path.join(os.path.dirname(path), "data_sfc.nc")):
            path = os.path.join(os.path.dirname(path), "data_sfc.nc")
        super().__init__(path, vars_sel, subset, read_params)

    def open_dataset(self) -> xr.Dataset:
        """
        Open the ECMWF dataset as an xarray.Dataset according to the request parameters.

        :returns: The opened and subsetted ECMWF dataset.
        """

        # open data
        ds = self.ds_src.copy()
        ds = ds.rename({"valid_time": "datetime"})

        # Build the subset
        build_subset(
            ds,
            subset=self.config.subset,
        )

        # list variables to include
        include_vars = None

        # Read image data if requested
        ds = read_dataset(
            ds=ds,
            include_vars=include_vars,
            subset=None,  # ERA5 uses ROI subsetting, not wavelength subsetting
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


if __name__ == "__main__":
    pass
