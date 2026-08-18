"""
eoio.readers.generic_tif.reader
"""

from __future__ import annotations

import xarray as xr
from pathlib import Path

from eoio.readers.ESAWorldCover.data_io import append_data_vars
from eoio.readers.ESAWorldCover.metadata import ESAWorldCoverMetadataExtractor
from eoio.readers.base import BaseRasterReader
from typing import Any, Dict, Optional
from eoio.deps import lazy_rasterio
from eoio.readers.subset.roi_subset import (
    ROISubsetResolver,
    ResolvedROISubset,
)

AUX_OPTIONS = []
MASK_OPTIONS = []


class ESAWorldCoverReader(BaseRasterReader):
    """
    Reader for ESAWorldCover data.

    """

    default_read_params: Dict[str, Any] = {
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

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = Path(path)

        ##self.layout = get_layout(self.path)

        self.meas_def = {
            "none": [],
            "all": ["landcover_map"],
        }

        self.aux_def = {
            "all": AUX_OPTIONS,
        }
        # think if some masks are actually aux info?
        self.mask_def = {
            "all": MASK_OPTIONS,
        }

        self.meas_var_res = {"landcover_map": 10}

        super().__init__(path, vars_sel, subset, read_params)

    def open_dataset(self) -> xr.Dataset:
        """
        Open the GeoTIFF as an xarray Dataset according to request parameters.
        """
        meas = self.resolved_config.vars_sel.get("meas")

        roi_subset = self.resolved_config.subset

        ds = xr.open_dataset(self.path)
        ds = xr.Dataset()

        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level")
        # needs path and reader, not opened product

        if meas:
            # rename "band_data"
            ds = append_data_vars(
                ds=ds,
                layout=self.path,
                subset=roi_subset,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks"),
            )

        self.mtd = ESAWorldCoverMetadataExtractor(self, ds)

        if mtd_level is True:
            mtd_level = "all"

        if isinstance(mtd_level, str):
            mtd_level = mtd_level.lower()

        if mtd_level in ("basic", "all"):
            ds = self.mtd.attach_metadata(
                ds,
                level=mtd_level,
            )

        # ds = apply_conventions(ds, layout=self.layout, config=self.resolved_config)
        return ds

    def resolve_subset(
        self,
        subset: Optional[Dict[str, Any]],
    ) -> None | ResolvedROISubset:
        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        try:
            rio = lazy_rasterio()
            with rio.open(self.path) as src:
                image_crs = str(src.crs)

        except Exception:
            raise ValueError("Cannot resolve ROI subset: raster CRS not available.")

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    @staticmethod
    def get_extension() -> str:
        return ".tif"


if __name__ == "__main__":
    pass
