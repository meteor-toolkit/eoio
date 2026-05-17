"""
Future notes:


"""

from __future__ import annotations
from typing import Any, Dict, Optional
from functools import wraps
import numpy as np

from eoio.readers.subset.base_subset import BaseSubsetResolver

__all__ = [
    "ZenithAngleSubsetResolver",
    "AzimuthAngleSubsetResolver",
]


class AngleSubsetError(ValueError):
    """
    Raised when wavelength subset resolution fails.

    :param message:
        Description of the error.
    :returns AngleSubsetError:
        Exception indicating invalid wavelength subset configuration.
    """


def validate_inputs_zenith(func):
    @wraps(func)
    def checker(
        self,
        data: Any,
        angle_params: Dict,
        *args,
        **kwargs,
    ):

        # Validate angle_params
        valid_keys = {"min", "max", "nearest"}
        for key in valid_keys:
            if key in angle_params:
                if angle_params[key] < 0 or angle_params[key] > 90:
                    raise AngleSubsetError("zenith angles in subset should be between 0 and 90")

        # -----------------------------------------------------------------
        # check that if max, min or nearest then value is just a single value
        return func(self, data, angle_params, *args, **kwargs)

    return checker


def validate_inputs_azimuth(func):
    @wraps(func)
    def checker(
        self,
        data: Any,
        angle_params: Dict,
        *args,
        **kwargs,
    ):

        # Validate angle_params
        valid_keys = {"min", "max", "nearest"}
        for key in valid_keys:
            if key in angle_params:
                angle_params[key] = angle_params[key] % 360
                if key == "max" and angle_params[key] == 0:
                    angle_params[key] = 360

        # Check that angles are within valid range
        for key in valid_keys:
            if key in angle_params:
                if angle_params[key] < 0 or angle_params[key] > 360:
                    raise AngleSubsetError("azimuth angles in subset should be between 0 and 360")

        # -----------------------------------------------------------------
        # check that if max, min or nearest then value is just a single value
        return func(self, data, angle_params, *args, **kwargs)

    return checker


# -----------------------------------------------------------------------------------
class ZenithAngleSubsetResolver(BaseSubsetResolver):
    """
    Resolves viewing zenith angle indices for subsetting raster data.

    :param data:
        Viewing zenith angle data (e.g., xarray DataArray).
        Data is expected to be an xarray-like object with .values, .size, and .dims.
    :param angle_params:
        Dictionary specifying angle selection criteria:
        * {'min': 0}
        * {'max': 90}
        * {'nearest': 20, 'tolerance': 2}
        * {'min': 20, 'max': 60}
    :returns ZenithAngleSubsetResolver:
        An instance ready to compute angle subsets.
    """

    @validate_inputs_zenith
    def __init__(
        self,
        data: Any,
        angle_params: Dict,
        accuracy: Optional[float] = 0.5,
    ):
        super().__init__(data, angle_params, accuracy=accuracy)


# -----------------------------------------------------------------------------------
class AzimuthAngleSubsetResolver(BaseSubsetResolver):
    """
    Resolves viewing azimuth angle indices for subsetting raster data.

    :param data:
        Viewing azimuth angle data (e.g., xarray DataArray).
        Data is expected to be an xarray-like object with .values, .size, and .dims.
    :param angle_params:
        Dictionary specifying angle selection criteria:
        * {'min': 200}
        * {'max': 600}
        * {'nearest': 450, 'tolerance': 20}
        * {'min': 200, 'max': 600}
    :returns AzimuthAngleSubsetResolver:
        An instance ready to compute angle subsets.
    """

    @validate_inputs_azimuth
    def __init__(
        self,
        data: Any,
        angle_params: Dict,
        accuracy: Optional[float] = 1,
    ):
        super().__init__(data, angle_params, accuracy=accuracy)

    def get_min_max(self):
        """
        Resolve indices for variables within a min-max range.

        :param variable:
            Tuple of (min, max) variable values.
        :param method:
            Resolution method ('min_max').
        :returns np.ndarray:
            Array of indices satisfying the min-max variable condition.
        """
        self.min_value = self.variable_params["min"]
        self.max_value = self.variable_params["max"]

        # call min
        variable_indices_min = self.get_min()
        # call max
        variable_indices_max = self.get_max()
        # resolve
        if self.min_value > self.max_value:
            resolved_variable_indices = np.sort(np.concatenate((variable_indices_min, variable_indices_max)))
        else:
            resolved_variable_indices = np.intersect1d(variable_indices_min, variable_indices_max)
        self.variable = (self.variable_params["min"], self.variable_params["max"])
        self.method = "min_max"
        return resolved_variable_indices


if __name__ == "__main__":
    pass
