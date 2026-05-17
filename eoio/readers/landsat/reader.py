"""eoio.readers.landsat.reader - Landsat data reader implementation."""

from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path

import xarray as xr

from eoio.readers.base import BaseRasterReader
from eoio.readers.landsat.layout import LandsatLayout
from eoio.readers.landsat.data_io import read_bands_into_dataset
from eoio.readers.landsat.conventions import apply_conventions
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset
from eoio.readers.landsat.metadata import LSMetadataExtractor
from eoio.readers.landsat.aux_data.aux_data import LSAuxData

# AUX_OPTIONS - Available auxiliary variable names
# N.B: depends on the processing level, more were added in L2 products
AUX_OPTIONS_L2_ONLY = [
    "ATRAN",
    "CDIST",
    "DRAD",
    "EMIS",
    "EMSD",
    "TRAD",
    "URAD",
]
ANGLE_OPTIONS = [
    "viewing_zenith_angle",
    "viewing_azimuth_angle",
    "solar_zenith_angle",
    "solar_azimuth_angle",
]

# MASK_OPTIONS - Available mask variable names
MASK_OPTIONS = [
    "fill",
    "dilated_cloud",
    "cirrus_LC",
    "cirrus_HC",
    "cloud_LC",
    "cloud_MC",
    "cloud_HC",
    "cloud_shadow_LC",
    "cloud_shadow_HC",
    "snow_ice_LC",
    "snow_ice_HC",
    "clear",
    "water",
    "saturated",
    "terrain_occlusion",
]

MASK_OPTIONS_L2_ONLY = [  # L2 has more mask options
    "aerosol_fill",
    "valid_aerosol_retrieval",
    "aerosol_water",
    "interpolated_aerosol",
    "aerosol_level_climatology",
    "aerosol_level_low",
    "aerosol_level_medium",
    "aerosol_level_high",
]


class LandsatReader(BaseRasterReader):
    """
    Landsat-8/9 product reader.
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

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.path = Path(path)

        # Initialize the Landsat layout - i.e., the file structure of the Landsat product
        self.layout = LandsatLayout(str(self.path))

        # Initialise the variable def dictionaries
        self.meas_def = {
            "all": self.layout.default_meas_vars(),
            "rgb": ["B2", "B3", "B4"],  # Blue, Green, Red
        }

        self.mask_def = {
            "all": [],  # MASK_OPTIONS
        }

        self.aux_def = {
            "all": ANGLE_OPTIONS,  # + AUX_OPTIONS_L2_ONLY,
        }

        # Initalise the metadata extractor
        self.mtd = LSMetadataExtractor(self)

        # Initialise the aux data manager
        self.aux = LSAuxData(self)

        super().__init__(path, vars_sel, subset, read_params)

    def resolve_subset(self, subset: Optional[Dict[str, Any]]) -> Optional[ResolvedROISubset]:
        """
        Resolve the subset of the Landsat product.

        :param subset: Subset of the Landsat product.
        :returns: Resolved ROI subset.
        """
        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        image_crs = self.mtd.basic_metadata.get("epsg", None)

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS (EPSG) not available.")

        # todo - add image_bounds to enable checking if ROI is within image bounds

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        """
        Open the Landsat dataset as an xarray.Dataset according to the request parameters.

        :returns: xarray.Dataset with requested data and metadata.
        """
        # Determine measurement variables to read, options:
        # * None - no bands
        # * [] - no bands
        # * ['B01', ... ] - some bands
        # * "all" - all bands

        # unpack config
        meas_vars = self.resolved_config.vars_sel.get("meas", None)
        roi_subset = self.resolved_config.subset
        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level", None)

        # Initialize an empty xarray Dataset to populate
        ds = xr.Dataset()

        # Add aux data if requested - angles must be added before bands to access per pixel solar elevation for reflectance calculation
        if self.resolved_config.vars_sel["aux"]:
            ds = self.aux.attach_aux(ds, aux=self.resolved_config.vars_sel["aux"])

        # Read image data if requested
        if meas_vars:
            ds = read_bands_into_dataset(
                ds=ds,
                layout=self.layout,
                meas_vars=meas_vars,
                subset=roi_subset,
                mtd=self.mtd,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks", None),
            )

        if mtd_level is True:
            mtd_level = "all"
        if isinstance(mtd_level, str) and mtd_level.lower() in ("all", "basic"):
            ds = self.mtd.attach_metadata(ds, level=mtd_level.lower())

        # Apply standard conventions to the dataset
        ds = apply_conventions(ds, layout=self.layout, roi_subset=roi_subset, config=self.resolved_config)

        return ds

    @classmethod
    def get_extension(cls) -> str:
        """
        Return the extension of the extracted Landsat product.
        """
        return ""


if __name__ == "__main__":
    pass
