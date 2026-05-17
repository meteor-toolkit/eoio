"""
eoio.readers.radcalnet.subset
=============================

Resolves user-defined subsets for radcalnet in situ measurements.

Classes
-------
.. autosummary::
   :toctree: generated/

   RADCALNETSubsetError
   RADCALNETSubset
   subset_to_timestamp

Functions
---------
.. autosummary::
   :toctree: generated/

   build_subset
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from typing import Optional
import xarray as xr

from eoio.readers.subset.wavelength_subset import WavelengthSubsetResolver
from eoio.readers.subset.angle_subset import (
    ZenithAngleSubsetResolver,
    AzimuthAngleSubsetResolver,
)
from eoio.readers.subset.time_of_day_subset import TimeOfDaySubsetResolver
from eoio.readers.subset.datetime_subset import DatetimeSubsetResolver


class RADCALNETSubsetError(ValueError):
    """
    Raised when a RadCalNet subset definition is invalid.

    """


@dataclass(frozen=True)
class RADCALNETSubset:
    """
    Resolved subsetting info for RadCalNet subset reads.

    Attributes
    ----------
    series_indices
        Indices for series dimension
    wavelength_indices
        Indices for wavelength dimension
    """

    series_indices: Optional[np.ndarray]
    wavelength_indices: Optional[np.ndarray]


def build_subset(*, ds: xr.Dataset, subset: dict) -> RADCALNETSubset:
    """
    Build a RADCALNETSubset from a dataset and subset definition.

    :param ds: The input dataset.
    :param subset: Subsetting definition.
    :returns: The resolved subsetting information.
    """

    # Collect indices for subsetting
    indices = None

    ## Wavelength subsetting, filter wavelength dimension based on user input ##
    wav_subset = WavelengthSubsetResolver(ds["wavelength"], subset["wavelength"]).run()

    # --- Datetime constraint: filter series by datetime range ---
    if subset["datetime"] is not None:
        # we might need this later for reading the netcdf version of the files
        datetime_subset = DatetimeSubsetResolver(ds["time"], subset["datetime"]).run()
        indices = datetime_subset.variable_indices

    ## Time subsetting, filter times based on user input ##
    if subset["time_of_day_local"] is not None:
        time_of_day_subset = TimeOfDaySubsetResolver(ds["local_time"], subset["time_of_day_local"]).run()
        indices = (
            time_of_day_subset.variable_indices
            if indices is None
            else np.intersect1d(indices, time_of_day_subset.variable_indices)
        )

    if subset["time_of_day_utc"] is not None:
        time_of_day_subset = TimeOfDaySubsetResolver(ds["time"], subset["time_of_day_utc"]).run()
        indices = (
            time_of_day_subset.variable_indices
            if indices is None
            else np.intersect1d(indices, time_of_day_subset.variable_indices)
        )

    ## Angle subsetting, filter angles based on user input ##
    if subset["angle"] is not None:
        # only solar geo included - radcalnet def at Nadir
        angle_str = ["sza", "saa"]  # should we not pull this info from _all_options?
        angle_str_ds = [
            "solar_zenith_angle",
            "solar_azimuth_angle",
        ]
        for ia in range(len(angle_str)):
            if angle_str[ia] in subset["angle"]:
                if angle_str_ds[ia] not in ds:
                    raise RADCALNETSubsetError(
                        f"Angle variable '{angle_str_ds[ia]}' not found in dataset for subsetting."
                    )
                if angle_str[ia] == "saa":
                    time_subset = AzimuthAngleSubsetResolver(ds[angle_str_ds[ia]], subset["angle"][angle_str[ia]]).run()

                else:
                    # Validate zenith angles are within [0, 90]
                    time_subset = ZenithAngleSubsetResolver(ds[angle_str_ds[ia]], subset["angle"][angle_str[ia]]).run()

                # Apply min/max/nearest constraints for angles
                indices = (
                    time_subset.variable_indices
                    if indices is None
                    else np.intersect1d(indices, time_subset.variable_indices)
                )

    if indices is not None and len(indices) == 0:
        raise ValueError("The subset criteria resulted in an empty dataset.")

    # the user gives a subset and we select this data
    return RADCALNETSubset(
        series_indices=indices,
        wavelength_indices=np.array(wav_subset.variable_indices) if len(wav_subset.variable_indices) > 0 else None,
    )
