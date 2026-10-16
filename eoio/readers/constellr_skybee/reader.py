"""eoio.readers.constellr_skybee.reader - constellr Skybee reader for L1C VNIR data (TIR also included in the product)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from eoio.readers.base import BaseRasterReader
from eoio.readers.constellr_skybee.aux_data import read_aux
from eoio.readers.constellr_skybee.conventions import apply_conventions
from eoio.readers.constellr_skybee.data_io import read_bands_into_dataset
from eoio.readers.constellr_skybee.masks import MASK_OPTIONS, add_masks
from eoio.readers.constellr_skybee.layout import MEAS_VAR_RES, ConstellrSkybeeLayout
from eoio.readers.constellr_skybee.metadata import ConstellrSkybeeMetadataExtractor
from eoio.readers.subset.roi_subset import ResolvedROISubset, ROISubsetResolver

AUX_OPTIONS = ["observation_geometry"]


class ConstellrSkybeeReader(BaseRasterReader):
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

    meas_var_res = MEAS_VAR_RES

    def __init__(
        self,
        path: Path | str,
        vars_sel: dict[str, Any] | None = None,
        subset: dict[str, Any] | None = None,
        read_params: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.layout = ConstellrSkybeeLayout(self.path)
        self.meas_def = {"none": [], "all": list(self.meas_var_res.keys()), "rgb": ["B2", "B3", "B4"]}
        self.aux_def = {"all": AUX_OPTIONS}
        self.mask_def = {"all": MASK_OPTIONS}
        self.mtd = ConstellrSkybeeMetadataExtractor(self)
        super().__init__(self.path, vars_sel, subset, read_params)

    def resolve_subset(self, subset: dict[str, Any] | None) -> ResolvedROISubset | None:
        if not subset:
            return None
        roi = subset.get("roi")
        if roi is None:
            return None
        roi_crs = subset.get("roi_crs")
        image_crs = self.mtd.product_metadata.get("geospatial_bounds_crs")
        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS not available.")

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        meas = self.resolved_config.vars_sel.get("meas", None)
        aux = self.resolved_config.vars_sel.get("aux", None)
        mask = self.resolved_config.vars_sel.get("mask", None)
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

        if aux:
            ds = read_aux(ds=ds, aux=aux, mtd=self.mtd)

        if mask:
            ds = add_masks(
                ds=ds,
                masks=self.list_selected_mask() if isinstance(mask, str) else mask,
                layout=self.layout,
                subset=roi_subset,
                config=self.resolved_config,
            )

        ds = apply_conventions(ds, layout=self.layout, roi_subset=roi_subset, config=self.resolved_config)

        return ds

    @staticmethod
    def get_extension() -> str:
        return ""


if __name__ == "__main__":
    pass
