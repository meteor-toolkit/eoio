"""
eoio.readers.radcalnet.reader
=============================

In situ data reader implementation for RadCalNet.

Classes
-------
.. autosummary::
   :toctree: generated/

   RADCALNETFileTypeError
   RadCalNetReader
   RadCalNetInputReader

Functions
---------
.. autosummary::
   :toctree: generated/
"""

import os.path
import xarray as xr
import numpy as np

from eoio.readers.radcalnet.subset import build_subset
from eoio.readers.radcalnet.metadata import RadCalNetMetadataExtractor
from eoio.readers.radcalnet.data_io import read_file, read_dataset
from eoio.readers.base import BaseReader


class RADCALNETFileTypeError(ValueError):
    """Error for an incurrect file being passed to the radcalnet reader"""

    pass


class RadCalNetReader(BaseReader):
    default_vars_sel = {
        "meas": "all",
        "aux": "all",
        "mask": None,
    }

    default_subset = {
        "wavelength": {
            "min": 400,
            "max": 2500,
        },  # TODO: BTCN has a smaller default subset
        "time_of_day_utc": {  # Changed from "time_utc" to "time_of_day_UTC"
            "min": "08:00",
            "max": "14:00",
        },
        "time_of_day_local": None,
        "angle": None,
        "datetime": None,
    }

    default_read_params = {
        "metadata_level": "all",
        "include_uncertainties": "true",
    }

    all_subset = {
        "wavelength": ["min", "max", "nearest", "tolerance"],
        "angle": {
            "sza": ["min", "max", "nearest", "tolerance"],
            "saa": ["min", "max", "nearest", "tolerance"],
        },
        "datetime": [
            "min",
            "max",
            "nearest",
            "tolerance_days",
            "tolerance_hours",
            "tolerance_minutes",
        ],
        "time_of_day_utc": "min/max/nearest/tolerance_hours/tolerance_minutes",
        "time_of_day_local": "min/max/nearest/tolerance_hours/tolerance_minutes",
    }

    meas_def = {
        "all": ["reflectance"],
    }

    aux_def = {
        "all": [
            "P",
            "T",
            "WV",
            "O3",
            "AOD",
            "Ang",
            "Type",
            "esd",
            "solar_zenith_angle",
            "solar_azimuth_angle",
        ]
    }

    uncertainty_vars = [
        "reflectance_uncertainty",
        "P_unc",
        "T_unc",
        "WV_unc",
        "O3_unc",
        "AOD_unc",
        "Ang_unc",
    ]

    def open_dataset(self) -> xr.Dataset:

        # Open data
        ext = os.path.splitext(self.path)[-1]
        if ext == ".nc":
            ds = xr.open_dataset(self.path)  # check if output/input
        elif ext == self.get_extension():
            ds = read_file(str(self.path), self.list_selected_aux())
        else:
            raise RADCALNETFileTypeError

        # list variables to include
        include_vars = self.list_include_vars()

        # convert missing values to np.nan to be consistent with other eoio readers
        mask = np.where(ds.reflectance.data >= 9998.0)
        ds.reflectance.data[mask] = np.nan
        ds.reflectance_uncertainty.data[mask] = np.nan

        # Determine measurement variables to read
        subset = build_subset(
            ds=ds,
            subset=self.config.subset,
        )

        # Read image data if requested
        ds = read_dataset(
            ds=ds,
            include_vars=include_vars,
            subset=subset,
        )

        # get metadata
        if "input" in self.get_extension():
            ds.attrs["collection"] = "Bottom of Atmosphere"
        elif "output" in self.get_extension():
            ds.attrs["collection"] = "Top of Atmosphere"

        if self.config.read_params.get("metadata_level", None) == "original":
            return ds
        else:
            meta_ex = RadCalNetMetadataExtractor(self, ds)
            ds = meta_ex.clear_metadata(ds)
            if self.config.read_params.get("metadata_level", None) in ("all", "basic"):
                ds = meta_ex.attach_metadata(ds, level=self.config.read_params["metadata_level"])
            return ds

    @staticmethod
    def get_extension() -> str:
        """Return the file/folder extension of the uncompressed file/folder
        (needed for decompressing as compressed filename might not have this extension)
        """
        return ".output"


class RadCalNetInputReader(RadCalNetReader):
    aux_def = {
        "all": [
            "P",
            "T",
            "WV",
            "O3",
            "AOD",
            "Ang",
            "Type",
        ]
    }

    @staticmethod
    def get_extension() -> str:
        """Return the file/folder extension of the uncompressed file/folder
        (needed for decompressing as compressed filename might not have this extension)
        """
        return ".input"


if __name__ == "__main__":
    pass
