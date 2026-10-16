"""eoio.readers.gistda_theos.reader - GISTDA THEOS reader for L1A VNIR data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr
from processor_tools.utils.dict_tools import get_value

from eoio.readers.base import BaseRasterReader
from eoio.readers.gistda_theos.aux_data import read_aux
from eoio.readers.gistda_theos.conventions import apply_conventions
from eoio.readers.gistda_theos.data_io import read_bands_into_dataset
from eoio.readers.gistda_theos.layout import GISTDATHEOSLayout
from eoio.readers.gistda_theos.metadata import GISTDATHEOSMetadataExtractor
from eoio.readers.subset.roi_subset import ResolvedROISubset, ROISubsetResolver

#: Implemented by add_angles() (see aux_data.py) -- solar/viewing angles from the
#: product's own Scene_Source metadata.
AUX_OPTIONS = ["observation_geometry"]
MASK_OPTIONS: list[str] = []


class GISTDATHEOSReader(BaseRasterReader):
    default_read_params: dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",
        "include_uncertainties": False,
        "use_chunks": False,
        "chunks": None,
    }

    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",
        "include_uncertainties": "True/False",
        "use_chunks": "True/False",
        "chunks": "None/{'x':X,'y':Y}",
    }

    meas_var_res = {"B1": 15, "B2": 15, "B3": 15, "B4": 15}

    def __init__(
        self,
        path: Path | str,
        vars_sel: dict[str, Any] | None = None,
        subset: dict[str, Any] | None = None,
        read_params: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.layout = GISTDATHEOSLayout(self.path)
        self.meas_def = {"none": [], "all": list(self.meas_var_res.keys()), "rgb": ["B1", "B2", "B3"]}
        self.aux_def = {"all": AUX_OPTIONS}
        self.mask_def = {"all": MASK_OPTIONS}
        self.mtd = GISTDATHEOSMetadataExtractor(self)
        super().__init__(self.path, vars_sel, subset, read_params)

    def resolve_subset(self, subset: dict[str, Any] | None) -> ResolvedROISubset | None:
        if not subset:
            return None
        roi = subset.get("roi")
        if roi is None:
            return None
        roi_crs = subset.get("roi_crs")
        image_crs = get_value(self.mtd.product_metadata, "geospatial_bounds_crs")
        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS not available.")

        ## TODO THEOS data coords are x,y , see in data_io test_img.coords['spatial_ref'].attrs['gcps']
        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        meas = self.resolved_config.vars_sel.get("meas", None)
        aux = self.resolved_config.vars_sel.get("aux", None)
        roi_subset = self.resolved_config.subset
        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level", None)

        ds = xr.Dataset()

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
        return ""


if __name__ == "__main__":
    pass
