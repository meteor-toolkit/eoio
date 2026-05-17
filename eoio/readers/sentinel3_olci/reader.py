"""eoio.readers.sentinel3_olci.reader - Sentinel-3 OLCI L1 reader adapter.

This module provides a thin, modern adapter that exposes a simple BaseReader
compatible reader following the sentinel2.reader pattern but re-using the
existing, well-tested implementation in ``eoio.readers.s3_olci_l1``.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path
import xarray as xr

from eoio.readers.base import BaseRasterReader
from eoio.readers.sentinel3_olci.metadata.extractor import S3OLCIMetadataExtractor
from .layout import S3OLCILayout
from eoio.readers.sentinel3_olci.data_io import (
    read_bands_into_dataset,
    read_uncertainty_into_dataset,
    read_lat_lon_coordinates,
)
from .auxiliary import add_aux
from .masks import add_masks
from .conventions import apply_conventions
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset

AUX_OPTIONS = [
    "observation_geometry",
    "removed_pixels",
    "humidity",
    "sea_level_pressure",
    "total_columnar_water_vapour",
    "total_ozone",
    "horizontal_wind",
    "atmospheric_temperature_profile",
    "FWHM",
    "detector_index",
    "frame_offset",
    "lambda0",
    "relative_spectral_covariance",
    "solar_flux",
]

MASK_OPTIONS = [
    "saturated",
    "dubious",
    "sun-glint_risk",
    "duplicated",
    "cosmetic",
    "invalid",
    "straylight_risk",
    "bright",
    "tidal_region",
    "fresh_inland_water",
    "coastline",
    "land",
]


class OLCIL1Reader(BaseRasterReader):
    """Sentinel-3 OLCI L1 reader.

    This adapter follows the ``eoio.readers.*.reader`` style: it accepts the
    standard BaseReader constructor arguments and exposes ``open()`` which
    returns an xarray.Dataset.

    :cvar default_read_params: Default read parameters for the reader.
    :cvar all_read_params: Documentation strings for supported read params.
    """

    default_read_params: Dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": True,
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

        # Initialise the S3OLCI layout
        self.layout = S3OLCILayout(str(self.path))

        # Initalise the metadata extractor
        self.mtd = S3OLCIMetadataExtractor(self)

        # Initialise the variable def dictionaries
        self.meas_def = {
            "all": self.layout.default_meas(),
            "rgb": ["Oa02", "Oa06", "Oa08"],
        }

        self.mask_def = {
            "all": MASK_OPTIONS,
        }

        self.aux_def = {"all": AUX_OPTIONS}

        super().__init__(path, vars_sel, subset, read_params)

    def resolve_subset(self, subset: Optional[Dict[str, Any]]) -> Optional[ResolvedROISubset]:
        """
        Resolve the subset of the OLCI product.

        :param subset: Subset of the OLCI product.
        :returns: Resolved ROI subset.
        """

        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        image_crs = self.mtd.product_metadata.get("crs_code", None)

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS not available.")

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        """Open the Sentinel-3 OLCI product and return an xarray.Dataset.

        The method resolves configuration, reads coordinate and image data,
        attaches metadata and optionally includes auxiliary data and masks.

        :returns: An xarray Dataset representing the opened OLCI product.
        """

        # Unpack resolved config
        meas = self.resolved_config.vars_sel.get("meas", None)
        roi_subset = self.resolved_config.subset
        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level", None)

        # Assign meas if using a predefined set
        if isinstance(meas, str):
            if meas.lower() in self.meas_def:
                meas = self.meas_def[meas.lower()]

            else:
                raise ValueError("Unknown meas requested: " + str(meas))

        if rp.get("include_uncertainties", False):
            self.uncertainty_vars = [
                f"u_radiance_{var}" for var in self.layout.requested_uncertainty_paths(meas).keys()
            ]

        # Initialise the dataset
        ds = xr.Dataset()

        # Read lat/lon coordinates
        ds, clip_boxes = read_lat_lon_coordinates(
            ds=ds,
            layout=self.layout,
            subset=roi_subset,
            config=self.resolved_config,
            use_chunks=rp.get("use_chunks", False),
            chunks=rp.get("chunks", None),
        )

        # Update roi_subset with clip boxes
        if roi_subset is not None and clip_boxes is not None:
            roi_subset.xy_clip_box = clip_boxes.get("lat_lon", None)
            roi_subset.tie_clip_box = clip_boxes.get("tie", None)

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
            if self.layout.proc_version >= 4 and rp.get("include_uncertainties", False):
                # Read uncertainty data if proc version >=4 and user has requested uncertainties
                read_uncertainty_into_dataset(
                    ds=ds,
                    layout=self.layout,
                    meas=meas,
                    subset=roi_subset,
                    use_chunks=rp.get("use_chunks", False),
                    chunks=rp.get("chunks", None),
                )

        if mtd_level is True:
            mtd_level = "all"
        if isinstance(mtd_level, str) and mtd_level.lower() in ("all", "basic"):
            ds = self.mtd.attach_metadata(ds, level=mtd_level.lower())

        # Add aux data if requested
        if self.resolved_config.vars_sel["aux"]:
            ds = add_aux(
                ds=ds,
                layout=self.layout,
                subset=roi_subset,
                config=self.resolved_config,
            )

        # Add masks if requested
        if self.resolved_config.vars_sel["mask"]:
            ds = add_masks(
                ds=ds,
                masks=self.resolved_config.vars_sel["mask"],
                layout=self.layout,
                subset=roi_subset,
                config=self.resolved_config,
            )

        # Apply standard conventions to the dataset
        ds = apply_conventions(ds, layout=self.layout, config=self.resolved_config)

        return ds

    @classmethod
    def get_extension(cls) -> str:
        """Return the extension of the extracted Sentinel-3 OLCI product.

        :returns: File extension for extracted SEN3 products.
        """
        return ".SEN3"


__all__ = ["OLCIL1Reader"]
if __name__ == "__main__":
    pass
