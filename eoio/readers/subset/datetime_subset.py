"""
Future notes:

"""

from __future__ import annotations
from typing import Any, Dict
from functools import wraps

import numpy as np

from eoio.readers.subset.base_subset import BaseSubsetResolver
from processor_tools.utils.formatters import convert_datetime

__all__ = ["DatetimeSubsetResolver"]


class DatetimeSubsetError(ValueError):
    """
    Raised when datetime subset resolution fails.

    :param message:
        Description of the error.
    :returns DatetimeSubsetError:
        Exception indicating invalid datetime subset configuration.
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
            raise DatetimeSubsetError("Data must have a 'size' attribute.")
        if data.size == 0:
            raise DatetimeSubsetError("Data cannot be empty.")

        if not isinstance(data.size, int):
            raise DatetimeSubsetError("Data size must be an integer.")

        # Validate variable_params
        if not isinstance(variable_params, dict):
            raise DatetimeSubsetError("Variable parameters must be provided as a dictionary.")

        # Validate variable_params
        valid_keys = {"min", "max", "nearest"}
        if not any(k in variable_params for k in valid_keys):
            raise DatetimeSubsetError(f"variable_params must contain one of {valid_keys}, got {variable_params.keys()}")

        # Validate tolerance if present
        if "tolerance_hours" in variable_params and not isinstance(variable_params["tolerance_hours"], (int, float)):
            raise DatetimeSubsetError("Tolerance_hours must be numeric.")

        if "tolerance_days" in variable_params and not isinstance(variable_params["tolerance_days"], (int, float)):
            raise DatetimeSubsetError("Tolerance_days must be numeric.")

        if "tolerance_minutes" in variable_params and not isinstance(
            variable_params["tolerance_minutes"], (int, float)
        ):
            raise DatetimeSubsetError("Tolerance_minutes must be numeric.")

        # check if no more than one tolerance is provided
        tolerance_keys = ["tolerance_days", "tolerance_hours", "tolerance_minutes"]
        provided_tolerances = [key for key in tolerance_keys if key in variable_params]

        if len(provided_tolerances) > 1:
            raise DatetimeSubsetError(f"Only one of {provided_tolerances} should be provided.")

        return func(self, data, variable_params, *args, **kwargs)

    return checker


# -----------------------------------------------------------------------------------
class DatetimeSubsetResolver(BaseSubsetResolver):
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

    @validate_inputs
    def __init__(
        self,
        data: Any,
        variable_params: Dict,
    ):
        valid_keys = {"min", "max", "nearest"}
        for key in valid_keys:
            if key in variable_params:
                variable_params[key] = convert_datetime(variable_params[key])

        if "tolerance_days" in variable_params:
            variable_params["tolerance"] = np.timedelta64(variable_params["tolerance_days"], "D")

        if "tolerance_hours" in variable_params:
            variable_params["tolerance"] = np.timedelta64(variable_params["tolerance_hours"], "h")

        if "tolerance_minutes" in variable_params:
            variable_params["tolerance"] = np.timedelta64(variable_params["tolerance_minutes"], "m")

        data_conv = data.copy()
        try:
            data_conv.values = convert_datetime(data.values)
        except ValueError:
            # this line makes data.values not timezone aware. From my reasearch it looks like a workaround for a larger bug
            # with how datetimes are handled by xarray - Ashley
            data_conv = data_conv.assign_coords(time=convert_datetime(data.values))
        super().__init__(data_conv, variable_params)


if __name__ == "__main__":
    pass
