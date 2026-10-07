"""eoio.readers.copernicus_dem.reader - Copernicus DEM 30m and 90m data reader implementation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import xarray as xr

from eoio.readers.base import BaseRasterReader
from eoio.readers.subset.roi_subset import (
    ROISubsetResolver,
    ResolvedROISubset,
)
from processor_tools.utils.dict_tools import get_value
from eoio.deps import lazy_rasterio
from eoio.readers.copernicus_dem.layout import get_layout
from eoio.readers.copernicus_dem.data_io import append_data_vars
from eoio.readers.copernicus_dem.metadata import CopernicusDEMMetadataExtractor
from eoio.readers.copernicus_dem.conventions import apply_conventions
from eoio.readers.copernicus_dem.masks import add_masks

AUX_OPTIONS = []

MASK_OPTIONS = [
    "water_body_mask",
    "filling_mask",
    "editing_mask",
    "height_error_mask",
]

__all__ = ["CopernicusDEMReader"]


class CopernicusDEMReader(BaseRasterReader):
    """
    Reader for Copernicus DEM GLO-30/GLO-90 products.
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

        self.layout = get_layout(self.path)

        self.meas_def = {
            "none": [],
            "all": ["elevation"],
        }

        self.aux_def = {
            "all": AUX_OPTIONS,
        }
        # think if some masks are actually aux info?
        self.mask_def = {
            "all": MASK_OPTIONS,
        }

        self.mtd = CopernicusDEMMetadataExtractor(self)

        resolution_variant = self.mtd.get_product_metadata().get("resolution_variant")

        self.meas_var_res = {"elevation": 30 if resolution_variant == "10" else 90}

        super().__init__(path, vars_sel, subset, read_params)

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

        metadata_files: list[Any] = getattr(self.layout, "metadata_files", lambda: [])()

        if not metadata_files:
            rio = lazy_rasterio()

            with rio.open(self.layout.dem) as src:
                image_crs = src.crs.to_string()
        else:
            image_crs = get_value(
                self.mtd.product_metadata,
                "geospatial_bounds_crs",
            )

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS not available.")

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        """
        Read Copernicus DEM product into xarray Dataset.
        """

        meas = self.resolved_config.vars_sel.get("meas")

        roi_subset = self.resolved_config.subset

        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level")

        ds = xr.Dataset()

        if meas:
            ds = append_data_vars(
                ds=ds,
                layout=self.layout.dem,
                subset=roi_subset,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks"),
            )

        if mtd_level is True:
            mtd_level = "all"

        if isinstance(mtd_level, str):
            mtd_level = mtd_level.lower()

        if mtd_level in ("basic", "all"):
            ds = self.mtd.attach_metadata(
                ds,
                level=mtd_level,
            )

        if self.resolved_config.vars_sel["mask"]:
            ds = add_masks(
                ds=ds,
                masks=self.resolved_config.vars_sel["mask"],
                layout=self.layout,
                subset=roi_subset,
                config=self.resolved_config,
            )

        ds = apply_conventions(ds, layout=self.layout, config=self.resolved_config)

        return ds

    @staticmethod
    def get_extension() -> str:
        """
        Copernicus DEM products are GeoTIFF-based.
        """
        return ""


if __name__ == "__main__":
    pass
