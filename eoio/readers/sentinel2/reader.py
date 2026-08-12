"""eoio.readers.sentinel2.reader - Sentinel-2 MSI data reader implementation."""

from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path
import xarray as xr
from eoio.readers.sentinel2.layout import S2Layout
from eoio.readers.sentinel2.data_io import read_bands_into_dataset
from eoio.readers.sentinel2.aux_vars.aux_data import get_available_aux, add_aux
from eoio.readers.sentinel2.conventions import apply_conventions
from eoio.readers.sentinel2.masks import MASK_OPTIONS, add_masks
from eoio.readers.base import BaseRasterReader
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset
from eoio.readers.sentinel2.metadata.extractor import S2MSIMetadataExtractor


class S2MSIReader(BaseRasterReader):
    default_read_params: Dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": False,
        "use_chunks": False,
        "chunks": None,
        "ave_va_det": False,
        "preferred_resolution": None,  # None means native resolution is used | 10/20/60 - if specified, will attempt to read bands at this resolution
    }

    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": "True/False",
        "use_chunks": "True/False",
        "chunks": "None/{'x':X,'y':Y}",
        "preferred_resolution": "None/10/20/60",  # None means native resolution is used | 10/20/60 - if specified, will attempt to read bands at this resolution
    }

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.path = Path(path)

        # Initialise the S2 layout - i.e., the file structure of the Sentinel-2 product
        self.layout = S2Layout(str(path))

        # Initialise the variable def dictionaries
        self.meas_def = {
            "all": self.layout.default_meas_vars(),
            "rgb": ["B02", "B03", "B04"],
        }

        self.mask_def = {
            "all": MASK_OPTIONS,
            # Convenience grouping: the two layers that represent cloud, as opposed
            # to snow/ice, which is a surface condition rather than an obstruction.
            "cloud": ["opaque_clouds", "cirrus"],
        }

        self.aux_def = {
            "all": get_available_aux(self.layout),
            "observation_geometry": [],  # key exists for options listing only; resolved dynamically in list_selected_aux
        }

        # Initalise the metadata extractor
        self.mtd = S2MSIMetadataExtractor(self)

        super().__init__(self.path, vars_sel, subset, read_params)

    def resolve_subset(
        self,
        subset: Optional[Dict[str, Any]],
    ) -> Optional[ResolvedROISubset]:
        """
        Resolve the subset of the Sentinel-2 product.

        :param subset: Subset of the Sentinel-2 product.
        :returns: Resolved ROI subset.
        """

        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        image_crs = self.mtd.product_metadata.get("horizontal_cs_code")

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS (horizontal_cs_code) not available.")

        # todo - add image_bounds to enable checking if ROI is within image bounds

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def list_selected_aux(self):
        if self.config.vars_sel.get("aux") == "observation_geometry":
            available = set(self.aux_def["all"])
            meas_bands = self.list_selected_meas()
            vars = []
            for band in meas_bands:
                for angle_type in ("viewing_zenith_angle", "viewing_azimuth_angle"):
                    var = f"{angle_type}_{band}"
                    if var in available:
                        vars.append(var)
            for solar_var in ("solar_zenith_angle", "solar_azimuth_angle"):
                if solar_var in available:
                    vars.append(solar_var)
            return vars
        return super().list_selected_aux()

    def open_dataset(self) -> xr.Dataset:
        """Open the Sentinel-2 dataset as xarray.Dataset according to the request parameters."""

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

        if meas_vars:
            ds = read_bands_into_dataset(
                ds=ds,
                layout=self.layout,
                meas_vars=meas_vars,
                subset=roi_subset,
                mtd=self.mtd,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks", None),
                preferred_resolution=preferred_resolution,
            )

        # Add auxiliary data if requested
        if self.config.vars_sel["aux"]:
            ds = add_aux(ds=ds, layout=self.layout, config=self.resolved_config, mtd=self.mtd)

        # Add mask data if requested
        if self.resolved_config.vars_sel.get("mask"):
            ds = add_masks(
                ds=ds,
                masks=self.resolved_config.vars_sel["mask"],
                layout=self.layout,
                subset=roi_subset,
                config=self.resolved_config,
            )

        if mtd_level is True:
            mtd_level = "all"
        if isinstance(mtd_level, str) and mtd_level.lower() in ("all", "basic"):
            ds = self.mtd.attach_metadata(ds, level=mtd_level.lower())

        # Apply standard conventions to the dataset
        ds = apply_conventions(ds, layout=self.layout, config=self.resolved_config)

        return ds

    @classmethod
    def get_extension(cls) -> str:
        """
        Return the extension of the extracted Sentinel-2 product.
        """
        return ".SAFE"


if __name__ == "__main__":
    pass
