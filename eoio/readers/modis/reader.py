"""eoio.readers.modis.reader - MODIS L1B data reader implementation."""

from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path
import xarray as xr
from eoio.deps import lazy_rioxarray
from eoio.readers.base import BaseRasterReader
from eoio.readers.modis.layout import MODISLayout
from eoio.readers.modis.metadata.extractor import MODISMetadataExtractor
from eoio.readers.modis.aux_vars.aux_data import get_available_aux
from eoio.readers.modis.data_io import read_bands_into_dataset
from eoio.readers.modis.aux_vars.aux_data import get_available_aux, add_aux
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset
from eoio.utils.rasterio_utils import suggest_raster_chunks


class MODISReader(BaseRasterReader):
    default_read_params: Dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": False,
        "use_chunks": False,
        "chunks": None,
        "preferred_resolution": None,  # None or int (e.g. 500) or str (e.g. '500m')
        "geolocation_dir": None,
    }
    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": "True/False",
        "use_chunks": "True/False",
        "chunks": "None/{'x':X,'y':Y}",
        "preferred_resolution": "None/int/str (e.g. 500 or '500')",
        "geolocation_dir": "None/Path/str",
    }

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.path = str(path)

        # Initialise MODIS layout
        self.layout = MODISLayout(
            path,
            preferred_resolution=read_params.get("preferred_resolution") if read_params else None,
            geolocation_dir=read_params.get("geolocation_dir") if read_params else None,
        )

        # Initialise the variable def dictionaries
        self.meas_def = {
            "all": self.layout.default_meas_vars(),
            "rgb": ["Band 3", "Band 4", "Band 1"],
        }

        self.mask_def = {"all": []}  # MASK_OPTIONS,

        self.aux_def = {"all": get_available_aux(self.layout)}

        # Initalise the metadata extractor
        self.mtd = MODISMetadataExtractor(self)

        super().__init__(self.path, vars_sel, subset, read_params)

    def resolve_subset(
        self,
        subset: Optional[Dict[str, Any]],
    ) -> Optional[ResolvedROISubset]:
        """
        Resolve the subset of the MODIS product.

        :param subset: Subset of the MODIS product.
        :return: Resolved ROI Subset.
        """

        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        image_crs = 4326

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS (horizontal_cs_code) not available.")

        # todo - add image_bounds to enable checking if ROI is within image bounds

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        """Open the MODIS dataset as xarray.Dataset according to the request parameters."""

        # unpack config
        meas_vars = self.resolved_config.vars_sel.get("meas", None)
        roi_subset = self.resolved_config.subset
        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level", None)

        # Initialize an empty xarray Dataset to populate
        ds = xr.Dataset()

        # Read image data if requested
        preferred_resolution = rp.get("preferred_resolution")
        if isinstance(preferred_resolution, str):
            preferred_resolution = None if preferred_resolution.lower() == "none" else int(preferred_resolution)

        geolocation_path = self.layout.geolocation_path()
        if "03" in str(geolocation_path):
            chunks = (
                suggest_raster_chunks(
                    self.layout.geolocation_path(),
                    group="/HDFEOS/SWATHS/MODIS_Swath_Type_GEO/Geolocation_Fields",
                    target_mb=32.0,
                )
                if rp.get("use_chunks", False) and rp.get("chunks", None) is None
                else rp.get("chunks", None)
            )
            geolocation_ds = xr.open_dataset(
                geolocation_path,
                group="/HDFEOS/SWATHS/MODIS_Swath_Type_GEO/Geolocation Fields",
                chunks=chunks,
            )
        elif "02" in self.layout.collection and self.layout.file_res_key == "1":
            if self.layout.proc_version == "7":
                chunks = (
                    suggest_raster_chunks(
                        self.layout.geolocation_path(),
                        group="/HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation_Fields",
                        target_mb=32.0,
                    )
                    if rp.get("use_chunks", False) and rp.get("chunks", None) is None
                    else rp.get("chunks", None)
                )
                geolocation_ds = xr.open_dataset(
                    geolocation_path, group="/HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation Fields", chunks=chunks
                )
            else:
                raise ValueError(
                    f"Unsupported processing version '{self.layout.proc_version}' for geolocation data in collection '{self.layout.collection}'. Only version '7' is currently supported for MOD021KM products. Please download corresponding MOD03/MYD03 geolocation files with processing version 7, or specify the geolocation directory in read_params and ensure it contains the appropriate geolocation files."
                )

        if meas_vars:
            ds = read_bands_into_dataset(
                ds=ds,
                geolocation_ds=geolocation_ds,
                layout=self.layout,
                meas_vars=meas_vars,
                subset=roi_subset,
                mtd=self.mtd,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks", None),
                preferred_resolution=preferred_resolution,
            )

        # Add mask data if requested
        if self.config.vars_sel["aux"]:
            ds = add_aux(
                ds=ds,
                geolocation_ds=geolocation_ds,
                layout=self.layout,
                subset=roi_subset,
                config=self.resolved_config,
                mtd=self.mtd,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks", None),
            )

        if mtd_level in ("all", "basic"):
            ds = self.mtd.attach_metadata(ds, level=mtd_level)

        return ds

    @classmethod
    def get_extension(cls) -> str:
        """
        Return the extension of the extracted MODIS product.
        """
        return ".hdf"


if __name__ == "__main__":
    pass
