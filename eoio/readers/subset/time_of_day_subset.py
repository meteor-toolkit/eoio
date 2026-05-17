"""
Future notes:

        indices_wav never none, if data is empty [] the indices_way []. but not None
        ds["wavelength"].size returns scalar value
        np.arange() create evenly sample index array with step = 1
        np.arrange(3) -> [0, 1, 2]

        ds["wavelength"].size returns tuple(2, 7) # do a check it is a scalar raise NotImplemented
        np.arrange(2, 7) -> [2, 3, 4, 5, 6]

        Is ds['wavelength] always 1D? # to make sense it should be either already idx list or it should be number of the wavelenghts
Create for 2D
"""

from __future__ import annotations
from typing import Any, Dict
from functools import wraps
import numpy as np
import datetime
from datetime import timezone

from eoio.readers.subset.base_subset import BaseSubsetResolver
from processor_tools.utils.formatters import convert_datetime
import xarray as xr


class TimeOfDaySubsetError(ValueError):
    """
    Raised when time of day subset resolution fails.

    :param message:
        Description of the error.
    :returns TimeOfDaySubsetError:
        Exception indicating invalid time of day subset configuration.
    """


def validate_inputs(func):
    @wraps(func)
    def checker(
        self,
        data,
        variable_params,
        *args,
        **kwargs,
    ):
        # Validate data
        if not hasattr(data, "size"):
            raise TimeOfDaySubsetError("Data must have a 'size' attribute.")
        if data.size == 0:
            raise TimeOfDaySubsetError("Data cannot be empty.")

        if not isinstance(data.size, int):
            raise TimeOfDaySubsetError("Data size must be an integer.")

        # Validate variable_params
        if not isinstance(variable_params, dict):
            raise TimeOfDaySubsetError("Variable parameters must be provided as a dictionary.")

        # Validate variable_params
        valid_keys = {"min", "max", "nearest"}
        if not any(k in variable_params for k in valid_keys):
            raise TimeOfDaySubsetError(
                f"variable_params must contain one of {valid_keys}, got {variable_params.keys()}"
            )

        # Validate tolerance if present
        if "tolerance_hours" in variable_params and not isinstance(variable_params["tolerance_hours"], (int, float)):
            raise TimeOfDaySubsetError("Tolerance_hours must be numeric.")

        if "tolerance_minutes" in variable_params and not isinstance(
            variable_params["tolerance_minutes"], (int, float)
        ):
            raise TimeOfDaySubsetError("Tolerance_minutes must be numeric.")

        # check if no more than one tolerance is provided
        tolerance_keys = ["tolerance_hours", "tolerance_minutes"]
        provided_tolerances = [key for key in tolerance_keys if key in variable_params]

        if len(provided_tolerances) > 1:
            raise TimeOfDaySubsetError(f"Only one of {provided_tolerances} should be provided.")

        return func(self, data, variable_params, *args, **kwargs)

    return checker


__all__ = ["TimeOfDaySubsetResolver"]


# -----------------------------------------------------------------------------------
class TimeOfDaySubsetResolver(BaseSubsetResolver):
    """
    Resolves datetime indices for subsetting raster data.

    :param data:
        Datetime data (e.g., xarray DataArray).
        Data is expected to be an xarray-like object with .values, .size, and .dims.
    :param datetime_params:
        Dictionary specifying datetime selection criteria:
        * {'min': '2020-01-01'}
        * {'max': '2020-12-31'}
        * {'nearest': '2020-06-01', 'tolerance': 20}
        * {'min': '2020-01-01', 'max': '2020-12-31'}
    :returns DatetimeSubsetResolver:
        An instance ready to compute wavelength subsets.
    """

    def __init__(
        self,
        data: Any,
        variable_params: Dict,
        timezone_info: timezone = timezone.utc,
    ):
        valid_keys = {"min", "max", "nearest"}
        for key in valid_keys:
            if key in variable_params:
                variable_params[key] = convert_datetime(variable_params[key])
                if not isinstance(variable_params[key], datetime.time):
                    variable_params[key] = variable_params[key].time().replace(tzinfo=timezone_info)

        if "tolerance_hours" in variable_params:
            variable_params["tolerance"] = np.timedelta64(variable_params["tolerance_hours"], "h")

        if "tolerance_minutes" in variable_params:
            variable_params["tolerance"] = np.timedelta64(variable_params["tolerance_minutes"], "m")

        # Create a new DataArray with times converted to UTC time objects
        times = [convert_datetime(t).time().replace(tzinfo=timezone_info) for t in data.values]
        data_conv = xr.DataArray(
            times,
            dims=data.dims,
            coords={data.dims[0]: data.coords[data.dims[0]].values},
            name=data.name if hasattr(data, "name") else None,
            attrs=getattr(data, "attrs", {}),
        )
        super().__init__(data_conv, variable_params)

        # self.time_mask = (
        #     ds["time"].values >= subset_to_timestamp(ds["time"].values[0], subset["time_utc"]["min"]),
        #     ds["time"].values <= subset_to_timestamp(ds["time"].values[0], subset["time_utc"]["max"]),
        # )


if __name__ == "__main__":
    pass
