"""
eoio.readers.generic_netcdf.reader
==================================

Reader for arbitrary netCDF files.

This is the fallback for any ``.nc`` file that no product-specific reader recognises,
including files written by :func:`eoio.write`.

Attributes are always decoded with :func:`eoio.serialisation.decode_attrs`, so values eoio
encoded on write (datetimes, nested dicts, geometries) come back as Python objects. Files
eoio did not write are decoded on a best-effort basis: strings holding a JSON object or
array, or an ISO-8601 datetime, are converted and everything else is left as it is.

Classes
-------
.. autosummary::
   :toctree: generated/

   NetCDFReader
"""

from __future__ import annotations
import xarray as xr
from eoio.readers.generic_netcdf.subset import build_subset
from eoio.readers.generic_netcdf.data_io import read_dataset
from eoio.readers.generic_netcdf.metadata import GenericNetCDFMetadataExtractor
from eoio.readers.base import BaseReader
from eoio.serialisation import decode_attrs


class NetCDFReader(BaseReader):
    """
    Reader for generic netCDF files.

    Every variable in the file is offered as a measurement variable. No subsetting
    parameters are accepted, since an arbitrary file cannot be assumed to follow eoio's
    coordinate naming.

    Readers for specific netCDF products (HYPERNETS, EMIT, ECMWF) subclass this for the
    source dataset it opens, ``self.ds_src``, but override :meth:`open_dataset`, so the
    attribute decoding done there applies to this reader only.
    """

    def __init__(self, path, vars_sel=None, subset=None, read_params=None):
        self.ds_src = xr.open_dataset(path)
        self.meas_def = {
            "all": list(self.ds_src.variables),
        }
        super().__init__(path, vars_sel, subset, read_params)

    def open_dataset(self) -> xr.Dataset:
        """
        Open the netCDF file as an xarray.Dataset according to the request parameters.

        :returns: The opened dataset, with attributes decoded.
        """

        # open data, restoring attributes encoded for serialisation (returns a copy)
        ds = decode_attrs(self.ds_src)

        # Build the subset
        subset = build_subset(
            ds,
            subset=self.config.subset,
        )

        # list variables to include
        include_vars = self.list_include_vars()

        # Read data
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
