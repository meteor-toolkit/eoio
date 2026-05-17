"""eoio.readers.airbus_pleiades.reader - Airbus Pleiades reader."""

from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path
import xarray as xr

from eoio.readers.base import BaseRasterReader
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset
from eoio.readers.airbus_pleiades.metadata import PleiadesMetadataExtractor
from eoio.readers.airbus_pleiades.layout import PleiadesLayout
from eoio.readers.airbus_pleiades.data_io import read_bands_into_dataset
from eoio.readers.airbus_pleiades.conventions import apply_conventions
from eoio.readers.airbus_pleiades.aux_data import read_aux

from processor_tools.utils.dict_tools import get_value

# AUX_OPTIONS - Available auxiliary variable names
AUX_OPTIONS = []  # TODO - add

# TODO - add observation geometry angles

# MASK_OPTIONS - Available mask variable names
MASK_OPTIONS = ["CLD", "DET", "QTE", "ROI", "SLT", "SNW", "VIS"]


class AirbusPleiadesReader(BaseRasterReader):
    """
    Base File Reader for Airbus Pleiades data
    """

    default_read_params: Dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": False,
        "use_chunks": False,
        "chunks": None,
    }
    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": "True/False",
        "use_chunks": "True/False",
        "chunks": "None/{'x':X,'y':Y}",
    }

    meas_var_res = {
        "B1": 2,
        "B2": 2,
        "B3": 2,
        "B4": 2,
    }

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ):

        self.path = Path(path)

        self.layout = PleiadesLayout(self.path)

        self.meas_def = {
            "none": [],  # no bands - don't read image
            "all": list(self.meas_var_res.keys()),
            "rgb": ["B1", "B2", "B3"],
        }

        self.aux_def = {
            "all": AUX_OPTIONS,
        }
        self.mask_def = {"all": MASK_OPTIONS}

        # initalise the metadata extractor
        self.mtd = PleiadesMetadataExtractor(self)

        # meas var res
        self.res = 2

        # run BaseRasterReader initializer
        super().__init__(self.path, vars_sel, subset, read_params)

    def resolve_subset(
        self,
        subset: Optional[Dict[str, Any]],
    ) -> Optional[ResolvedROISubset]:
        """
        Resolve the subset of the Airbus Pleiades product.

        :param subset: Subset of the Airbus Pleiades product.
        :returns: Resolved ROI subset.
        """

        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        image_crs = get_value(self.mtd.product_metadata, "geospatial_bounds_crs")

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS not available.")

        # TODO add image_bounds to enable checking if ROI is within image bounds

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    # open data
    def open_dataset(self) -> xr.Dataset:
        """Open the Airbus Pleiades dataset as xarray.Dataset according to the request parameters."""

        # unpack config
        meas = self.resolved_config.vars_sel.get("meas", None)
        aux = self.resolved_config.vars_sel.get("aux", None)
        roi_subset = self.resolved_config.subset
        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level", None)

        # initialize an empty xarray Dataset to populate
        ds = xr.Dataset()

        # Read image data if requested
        if meas:
            ds = read_bands_into_dataset(
                ds=ds,
                layout=self.layout,
                meas=meas,
                subset=roi_subset,
                mtd=self.mtd,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks", None),
            )

        if mtd_level is True:
            mtd_level = "all"
        if isinstance(mtd_level, str) and mtd_level.lower() in ("all", "basic"):
            ds = self.mtd.attach_metadata(ds, level=mtd_level.lower())

        # add aux data if requested
        if aux:
            ds = read_aux(ds=ds, aux=self.resolved_config.vars_sel["aux"], mtd=self.mtd)

        ds = apply_conventions(ds, layout=self.layout, roi_subset=roi_subset, config=self.resolved_config)

        return ds

    @staticmethod
    def get_extension() -> str:
        """
        Return the extension of the extracted Airbus Pleiades product.
        """
        return ""


if __name__ == "__main__":
    pass
