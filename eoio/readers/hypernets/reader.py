"""
eoio.readers.hypernets.reader
============================

HYPERNETS NetCDF data reader classes for Level 1 Radiance, Level 1 Irradiance, and Level 2 Reflectance products.

Classes
-------
.. autosummary::
   :toctree: generated/

   HYPERNETSReader
   HYPERNETSL1RadReader
   HYPERNETSL1IrrReader
   HYPERNETSL2RefReader

Functions
---------
.. autosummary::
   :toctree: generated/
"""

from __future__ import annotations
import xarray as xr
from eoio.readers.hypernets.subset import build_subset
from eoio.readers.hypernets.data_io import read_dataset

# from eoio.readers.hypernets.aux import maybe_add_aux
from eoio.readers.hypernets.metadata import HYPERNETSMetadataExtractor
from eoio.readers.generic_netcdf.reader import NetCDFReader


class HYPERNETSReader(NetCDFReader):
    """
    Reader for HYPERNETS data.

    Attributes
    ----------
    default_vars_sel
        Default variable selection options.
    default_subset
        Default subsetting options.
    default_read_params
        Default reading parameters.
    """

    default_vars_sel = {
        "meas": "all",  # e.g. []
        "aux": "all",
        "mask": "all",  # e.g. ["vza_B1"]
    }

    default_subset = {
        "mask": None,
        "wavelength": {"min": 380, "max": 1700},
        "angle": {
            "vza": {"min": 0, "max": 90},
            "vaa": {"min": 0, "max": 360},
            "sza": {"min": 0, "max": 90},
            "saa": {"min": 0, "max": 360},
            "raa": {"min": 0, "max": 360},
        },
        "datetime": None,
        "time_of_day_utc": None,
    }

    all_subset = {
        "wavelength": ["min", "max", "nearest", "tolerance"],
        "angle": {
            "vza": ["min", "max", "nearest", "tolerance"],
            "vaa": ["min", "max", "nearest", "tolerance"],
            "sza": ["min", "max", "nearest", "tolerance"],
            "saa": ["min", "max", "nearest", "tolerance"],
            "raa": ["min", "max", "nearest", "tolerance"],
        },
        "datetime": [
            "min",
            "max",
            "nearest",
            "tolerance_days",
            "tolerance_hours",
            "tolerance_minutes",
        ],
        "time_of_day_utc": [
            "min",
            "max",
            "nearest",
            "tolerance_hours",
            "tolerance_minutes",
        ],
        "mask": [
            True,
            False,
            "name of quality flag (see https://hypernets-processor.readthedocs.io/en/latest/content/atbd/products/flags.html)",
        ],  # True/False or list of mask names
    }

    default_read_params = {
        "save_extracted": "True",
        "metadata_level": "all",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": True,
    }

    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/original/None",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": "True/False",
    }

    uncertainty_vars = []  # to be defined in child classes

    def open_dataset(self) -> xr.Dataset:
        """
        Open the HYPERNETS dataset as an xarray.Dataset according to the request parameters.

        :returns: The opened and subsetted HYPERNETS dataset.
        """

        # open data
        ds = self.ds_src.copy()

        # list variables to include
        include_vars = self.list_include_vars()

        # Build the subset
        subset = build_subset(
            ds,
            subset=self.config.subset,
        )

        # Read image data if requested
        ds = read_dataset(
            ds=ds,
            include_vars=include_vars,
            subset=subset,
        )

        mtd_level = self.config.read_params.get("metadata_level", None)
        if mtd_level is True:
            mtd_level = "all"

        if mtd_level == "original":
            return ds
        else:
            meta_ex = HYPERNETSMetadataExtractor(self, ds)
            ds = meta_ex.clear_metadata(ds)
            if mtd_level in ("all", "basic"):
                ds = meta_ex.attach_metadata(ds, level=mtd_level)
            return ds


class HYPERNETSL1RadReader(HYPERNETSReader):
    """
    Reader for HYPERNETS Level 1 Radiance products.

    Attributes
    ----------
    meas_def
        Measurement variable definitions.
    aux_def
        Auxiliary variable definitions.
    uncertainty_vars
        List of uncertainty variable names.
    """

    meas_def = {
        "all": ["radiance"],
    }

    aux_def = {
        "all": [
            "acquisition_time",
            "series_id",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "n_valid_scans",
            "n_valid_scans_SWIR",
            "bandwidth",
            "std_radiance",
            "acceleration_x_mean",
            "acceleration_y_mean",
            "acceleration_z_mean",
            "acceleration_x_std",
            "acceleration_y_std",
            "acceleration_z_std",
        ],
        "basic": [
            "acquisition_time",
            "series_id",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "bandwidth",
        ],
    }

    uncertainty_vars = [
        "u_rel_random_radiance",
        "u_rel_systematic_corr_rad_irr_radiance",
        "u_rel_systematic_indep_radiance",
        "err_corr_systematic_corr_rad_irr_radiance",
        "err_corr_systematic_indep_radiance",
    ]


class HYPERNETSL1IrrReader(HYPERNETSReader):
    """
    Reader for HYPERNETS Level 1 Irradiance products.

    Attributes
    ----------
    meas_def
        Measurement variable definitions.
    aux_def
        Auxiliary variable definitions.
    uncertainty_vars
        List of uncertainty variable names.
    """

    meas_def = {
        "all": ["irradiance"],
    }

    aux_def = {
        "all": [
            "acquisition_time",
            "series_id",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "n_valid_scans",
            "n_valid_scans_SWIR",
            "bandwidth",
            "std_radiance",
            "acceleration_x_mean",
            "acceleration_y_mean",
            "acceleration_z_mean",
            "acceleration_x_std",
            "acceleration_y_std",
            "acceleration_z_std",
        ],
        "basic": [
            "acquisition_time",
            "series_id",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "bandwidth",
        ],
    }

    uncertainty_vars = [
        "u_rel_random_irradiance",
        "u_rel_systematic_corr_rad_irr_irradiance",
        "u_rel_systematic_indep_irradiance",
        "err_corr_systematic_corr_rad_irr_irradiance",
        "err_corr_systematic_indep_irradiance",
    ]


class HYPERNETSL2RefReader(HYPERNETSReader):
    """
    Reader for HYPERNETS Level 2 Reflectance products.

    Attributes
    ----------
    meas_def
        Measurement variable definitions.
    aux_def
        Auxiliary variable definitions.
    mask_def
        Mask definitions.
    uncertainty_vars
        List of uncertainty variable names.
    """

    meas_def = {
        "all": ["reflectance"],
    }

    aux_def = {
        "all": [
            "acquisition_time",
            "series_id",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "n_valid_scans",
            "n_valid_scans_SWIR",
            "bandwidth",
            "std_reflectance",
        ],
        "basic": [
            "acquisition_time",
            "series_id",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "bandwidth",
        ],
    }

    mask_def = {
        "all": [
            "quality_flag",
        ]
    }

    uncertainty_vars = [
        "u_rel_random_reflectance",
        "u_rel_systematic_reflectance",
        "err_corr_systematic_reflectance",
    ]


if __name__ == "__main__":
    pass
