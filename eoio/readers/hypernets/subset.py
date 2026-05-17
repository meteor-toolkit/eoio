"""
eoio.readers.hypernets.subset
=============================

Subsetting utilities for HYPERNETS NetCDF data. Defines classes and functions to resolve user-defined subsets for HYPERNETS products, including wavelength, angle, datetime, and time-of-day subsetting.

Classes
-------
.. autosummary::
   :toctree: generated/

   HYPERNETSSubsetError
   HYPERNETSSubset

Functions
---------
.. autosummary::
   :toctree: generated/

   build_subset
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import xarray as xr

from obsarray.templater.dataset_util import DatasetUtil
from eoio.readers.subset.wavelength_subset import WavelengthSubsetResolver
from eoio.readers.subset.angle_subset import (
    ZenithAngleSubsetResolver,
    AzimuthAngleSubsetResolver,
)
from eoio.readers.subset.datetime_subset import DatetimeSubsetResolver
from eoio.readers.subset.time_of_day_subset import TimeOfDaySubsetResolver


@dataclass(frozen=True)
class HYPERNETSSubset:
    """
    Resolved subsetting info for HYPERNETS subset reads.

    Attributes
    ----------
    series_indices
        Indices for series dimension
    wavelength_indices
        Indices for wavelength dimension
    """

    series_indices: Optional[np.ndarray] = None
    wavelength_indices: Optional[np.ndarray] = None


def build_subset(ds: xr.Dataset, subset: dict) -> HYPERNETSSubset:
    """
    Build a HYPERNETSSubset from a dataset and subset definition.

    :param ds: The input dataset.
    :param subset: Subsetting definition.
    :returns: The resolved subsetting information.
    """

    wav_subset = WavelengthSubsetResolver(ds["wavelength"], subset["wavelength"]).run()

    # Collect indices for subsetting based on masks, wavelength, and angles
    indices = None

    # --- Datetime constraint: filter series by datetime range ---
    if subset["datetime"] is not None:
        datetime_subset = DatetimeSubsetResolver(ds["acquisition_time"], subset["datetime"]).run()
        indices = datetime_subset.variable_indices

    if subset["time_of_day_utc"] is not None:
        time_of_day_subset = TimeOfDaySubsetResolver(ds["acquisition_time"], subset["time_of_day_utc"]).run()
        indices = (
            time_of_day_subset.variable_indices
            if indices is None
            else np.intersect1d(indices, time_of_day_subset.variable_indices)
        )

    # --- Mask constraint: filter series by quality flags or custom masks ---
    if subset["mask"] is not None and subset["mask"] is not False:
        if subset["mask"] is True:
            mask_idx = np.where(ds.quality_flag.values == 0)[0]
        else:
            if isinstance(subset["mask"], str):
                subset["mask"] = [subset["mask"]]
            masked = DatasetUtil.get_flags_mask_or(ds["quality_flag"], subset["mask"])
            mask_idx = np.where(~masked)[0]
        indices = mask_idx if indices is None else np.intersect1d(indices, mask_idx)

    # --- Angles constraint: filter series by viewing/solar angles ---
    if subset["angle"] is not None:
        angle_str = ["vza", "vaa", "sza", "saa", "raa"]
        angle_str_ds = [
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "relative_azimuth_angle",
        ]

        for ia in range(len(angle_str)):
            if angle_str[ia] in subset["angle"]:
                # Calculate relative azimuth angle if needed
                if angle_str[ia] == "raa":
                    ds_raa = 180 - np.abs(np.abs(ds["viewing_azimuth_angle"] - ds["solar_azimuth_angle"]) - 180)
                    ds[angle_str_ds[ia]] = ds_raa

                if angle_str[ia] == "vaa" or angle_str[ia] == "saa" or angle_str[ia] == "raa":
                    series_subset = AzimuthAngleSubsetResolver(
                        ds[angle_str_ds[ia]], subset["angle"][angle_str[ia]]
                    ).run()

                else:
                    # Validate zenith angles are within [0, 90]
                    series_subset = ZenithAngleSubsetResolver(
                        ds[angle_str_ds[ia]], subset["angle"][angle_str[ia]]
                    ).run()

                # Apply min/max/nearest constraints for angles
                indices = (
                    series_subset.variable_indices
                    if indices is None
                    else np.intersect1d(indices, series_subset.variable_indices)
                )

    if indices is not None and len(indices) == 0:
        raise ValueError("The subset criteria resulted in an empty dataset.")

    # Return the resolved subset indices for series and wavelength
    return HYPERNETSSubset(
        series_indices=indices,
        wavelength_indices=np.array(wav_subset.variable_indices) if len(wav_subset.variable_indices) > 0 else None,
    )


if __name__ == "__main__":
    pass
