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

    def __init__(
        self,
        path,
        vars_sel=None,
        subset=None,
        read_params=None,
    ) -> None:
        # Capture whether the caller explicitly set time_of_day_utc *before* it's merged
        # with the class-level default_subset in super().__init__() below -- once merged,
        # there's no way to tell an explicit override apart from the class default (both
        # just end up as self.config.subset["time_of_day_utc"]). Needed because that
        # default is a fixed UTC clock window, wrong for most sites' actual local daylight
        # hours; see _site_daylight_window_utc's docstring, used in open_dataset() below
        # only when this is False.
        self._time_of_day_utc_explicitly_set = bool(subset) and "time_of_day_utc" in subset
        super().__init__(path, vars_sel=vars_sel, subset=subset, read_params=read_params)

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

        subset_cfg = self.config.subset
        if not self._time_of_day_utc_explicitly_set:
            longitude = ds.attrs.get("Longitude")
            if longitude is not None:
                subset_cfg = {
                    **subset_cfg,
                    "time_of_day_utc": self._site_daylight_window_utc(float(longitude)),
                }

        # Determine measurement variables to read
        subset = build_subset(
            ds=ds,
            subset=subset_cfg,
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
    def _site_daylight_window_utc(longitude: float) -> dict:
        """Return this site's approximate daylight recording window as a
        ``time_of_day_utc`` subset dict (``{"min": "HH:MM", "max": "HH:MM"}``).

        RadCalNet only records meaningful reflectance data during daylight, so
        each site's file only covers a window around its own local solar noon
        -- not a fixed UTC clock window, since that shifts with longitude (e.g.
        RVUS, at 115.69 degW, has its daylight window many hours later in UTC
        than GONA, at 15.12 degE -- a UTC window hardcoded from one site is
        wrong for the other). Approximated as local solar noon +/- 3 hours
        (local solar time offset from UTC = longitude / 15 deg-per-hour) --
        this matches default_subset's old hardcoded window (GONA: solar noon
        ~10:59 UTC, window ~08:00-14:00 UTC) almost exactly, and mirrors
        scrappi's RadcalnetCallHandler._daylight_window_hours, which computes
        the equivalent window for matchup-finding using the same formula.

        Not valid for sites whose window straddles a UTC day boundary (roughly
        longitude > 135 deg E or < 135 deg W): unlike
        ``RadcalnetCallHandler._daylight_window_hours``, whose (possibly
        negative or >24) hours get added via ``timedelta`` (which normalises
        the wraparound automatically), TimeOfDaySubsetResolver's own min/max
        comparison has no such handling, so a wrapped window here would select
        zero rows instead. None of RadCalNet's current sites are in that
        range.

        :param longitude: site longitude in decimal degrees.
        :return: ``time_of_day_utc`` subset dict for this site.
        """
        solar_noon_utc = 12.0 - longitude / 15.0

        def _format(hour: float) -> str:
            # Round to the nearest minute via total minutes, not independently-rounded
            # hour/minute parts -- rounding e.g. 7.999h to hour=8, minute=round(0.999*60)
            # gives the invalid "08:60" instead of correctly carrying into "09:00".
            total_minutes = round(hour * 60)
            h, m = divmod(total_minutes, 60)
            return f"{h:02d}:{m:02d}"

        return {"min": _format(solar_noon_utc - 3.0), "max": _format(solar_noon_utc + 3.0)}

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
