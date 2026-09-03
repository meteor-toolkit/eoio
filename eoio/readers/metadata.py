"""
eoio.readers.metadata
==============================

Metadata extraction utilities that are common across multiple datasets.

Classes
-------
.. autosummary::
   :toctree: generated/

   BaseMetadataExtractor
"""

from abc import abstractmethod, ABC
from typing import Any, List, Optional
import xarray as xr
import datetime as dt
import os
from eoio.readers.base import BaseReader

from importlib.metadata import version

__version__ = version("eoio")


class BaseMetadataExtractor(ABC):
    """
    Base class for extracting metadata from xarray datasets.
    """

    def __init__(self, reader: BaseReader):
        self.path = reader.path
        self.reader = reader

        # initialise caching attributes for metadata
        self._basic_metadata: None | dict[str, Any] = None
        self._product_metadata: None | dict[str, Any] = None
        self._variable_product_metadata: dict = {}
        self._variable_basic_metadata: dict = {}

    @property
    def basic_metadata(self) -> dict[str, Any]:
        if self._basic_metadata is None:
            self._basic_metadata = self.get_basic_metadata()
        return self._basic_metadata

    @property
    def product_metadata(self) -> dict[str, Any]:
        if self._product_metadata is None:
            self._product_metadata = self.get_product_metadata()
        return self._product_metadata

    def variable_product_metadata(self, var: str) -> dict[str, Any]:
        if var not in self._variable_product_metadata:
            self._variable_product_metadata[var] = self.get_variable_product_metadata(var)

        return self._variable_product_metadata[var]

    def variable_basic_metadata(self, var: str) -> dict[str, Any]:
        if var not in self._variable_basic_metadata:
            self._variable_basic_metadata[var] = self.get_variable_basic_metadata(var)

        return self._variable_basic_metadata[var]

    def extract_metadata(
        self, level: None | str = None, meas_list: Optional[List[str]] = None
    ) -> tuple[dict, dict, dict, dict]:
        """
        Return basic, product, and variable metadata for the dataset.

        :param level: Metadata level ('all', 'basic', etc.).
        :param meas_list: Optional list of measurement variables used to
            filter product-level spatial metadata.
        :returns: Tuple of (basic_dataset_metadata, product_dataset_metadata,
            variable_basic_metadata, variable_metadata).
        """
        basic_md = dict(self.basic_metadata)

        md = {}
        var_md = {}
        var_basic_md = {}

        if level == "all" or level == "basic":
            md = self.product_metadata
            for var in self.reader.list_include_vars():
                var_md[var] = self.variable_product_metadata(var)
                var_basic_md[var] = self.variable_basic_metadata(var)
                if (
                    not self.reader.config.read_params.get("include_uncertainties", True)
                ) and var in self.reader.list_selected_meas():
                    # remove uncertainty variables from variable metadata
                    if "unc_comps" in var_md[var]:
                        var_basic_md[var]["unc_comps"] = []

        # If a measurement list was provided, filter product-level spatial
        # lists (e.g. spatial_resolution, geometry_ids) to only include
        # values for those measurement bands. "geometry_ids" (plural) is the
        # dataset-level key -- get_basic_metadata()'s per-variable
        # counterpart is "geometry_id" (singular), read from vmd below.
        if meas_list and ("spatial_resolution" in basic_md or "geometry_ids" in basic_md):
            # Ensure it's a list of strings
            if isinstance(meas_list, str):
                meas_list = [meas_list]

            sr_vals = []
            geom_vals = []
            for var in meas_list:
                # Only include vars that are recognised measurement bands
                vmd = self.variable_product_metadata(var)
                if not vmd:
                    continue
                if "spatial_resolution" in vmd:
                    sr_vals.append(vmd.get("spatial_resolution"))
                if "geometry_id" in vmd:
                    geom_vals.append(vmd.get("geometry_id"))

            if sr_vals:
                basic_md["spatial_resolution"] = sr_vals
            if geom_vals:
                basic_md["geometry_ids"] = geom_vals

        return basic_md, md, var_basic_md, var_md

    def attach_metadata(self, ds: xr.Dataset, level: None | str = None) -> xr.Dataset:
        """
        Attach metadata to a dataset and return a new dataset.

        :param ds: Dataset to which metadata should be attached.
        :param level: Metadata level to attach ('all', 'basic', etc.).
        :returns: Dataset with metadata attached.
        """
        # Copy to avoid mutating caller-owned dataset
        out = ds.copy()
        out.attrs = dict(out.attrs)

        # Determine measurement variables requested/resolved and present in dataset
        meas_requested: list = []
        rc = getattr(self.reader, "resolved_config", None)
        if rc:
            meas_requested = rc.vars_sel.get("meas", []) or []
        if isinstance(meas_requested, str):
            meas_requested = [meas_requested]

        meas_in_ds = [v for v in meas_requested if v in out]

        # Centralize filtering in extract_metadata by passing meas_in_ds
        basic_md, md, var_basic_md, var_md = self.extract_metadata(level=level, meas_list=meas_in_ds)

        out.attrs.update(basic_md)
        for var, vmd in var_basic_md.items():
            if var in out:
                out[var].attrs = dict(out[var].attrs)
                out[var].attrs.update(var_basic_md.get(var, {}))

        if level == "all":
            if md:
                out.attrs["product_metadata"] = md

            for var, vmd in var_md.items():
                if var in out:
                    out[var].attrs = dict(out[var].attrs)
                    if vmd:
                        out[var].attrs["product_metadata"] = vmd

        out.attrs["eoio:version"] = __version__
        out.attrs["eoio:path"] = self.path
        # Defaults only -- a reader's own get_basic_metadata() (merged in above via
        # out.attrs.update(basic_md)) takes precedence over these if it already supplied a
        # value, rather than being silently overwritten.
        out.attrs.setdefault("date_created", dt.datetime.now(dt.timezone.utc).isoformat())
        out.attrs.setdefault("license", "TBD")
        out.attrs.setdefault("references", "TBD")

        dt_now = dt.datetime.now(dt.timezone.utc).isoformat()
        if "history" in out.attrs:
            out.attrs["history"] = (
                out.attrs["history"]
                + f"\n{dt_now}: {os.path.split(self.path)[-1]} read in using eoio version {__version__}"
            )
        else:
            out.attrs["history"] = f"{dt_now}: {os.path.split(self.path)[-1]} read in using eoio version {__version__}"

        return out

    def clear_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Remove metadata (i.e. all attributes) from product and data variables in place in
        dataset. Coordinate attrs (e.g. a wavelength coordinate's units) are left untouched:
        attach_metadata() only ever restores attrs for data variables (via
        ``self.reader.list_include_vars()``), so wiping coordinate attrs here would discard
        them permanently rather than clearing-then-reattaching them.

        :param ds: Dataset from which to remove metadata (attrs).
        :returns: Dataset with attrs cleared.
        """
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        return ds

    def get_product_metadata(self) -> dict:
        """
        Extract product metadata from the dataset.

        :returns: Product metadata dictionary.
        """
        return {}

    @abstractmethod
    def get_basic_metadata(self) -> dict:
        """
        Extract basic metadata from the dataset, to be used in other tools in MetEOR.

        :returns: Basic metadata dictionary.
        """
        return {
            "collection_name": "",
            "product_name": "",
            "platform": "",
            "name": "",
            "processing_level": "",
            "spatial_resolution": "",
            "geometry_ids": "",
            "product_bounds": "",
            # Full date+time where the source data supports it (e.g. an ISO 8601 datetime
            # string/object). Some products only ever have date granularity available (e.g.
            # RadCalNet's daily files spanning many timestamps, or an annual composite like
            # ESA WorldCover) -- for those, this holds just the date, not a fabricated time.
            "product_datetime": "",
            "description": "",
            "eoio:reader": "",
            "eoio:subset": "",
            "history": "",  # include if there is any history previous to eoio reading
            "footprint": "",  # canonical form: {"geometry": Shapely, "crs": "EPSG:XXXXX", "bounds": tuple}
        }

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the dataset.

        :param var: Variable name for which metadata should be extracted.
        :returns: Variable metadata dictionary.
        """
        return {}

    @abstractmethod
    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Extract basic variable metadata from the dataset.

        :param var: Variable name for which metadata should be extracted.
        :returns: Variable metadata dictionary.
        """
        return {"units": "", "long_name": "", "standard_name": ""}


if __name__ == "__main__":
    pass
