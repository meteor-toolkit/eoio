from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from functools import wraps
import numpy as np
from abc import ABC
from processor_tools.utils.formatters import convert_datetime

__all__ = ["BaseSubsetResolver", "ResolvedVariableSubset", "VariableSubsetError"]


@dataclass
class ResolvedVariableSubset:
    """
    Represents a resolved variable indices for subsetting operations.

    :param variable:
        The requested variable or variable range (int, float, or tuple).
    :param variable_indices:
        List of indices corresponding to the resolved variables.
    :param variable_dimensions:
        Dimensions of the variable data (e.g., xarray dims).
    :param method:
        Method used for resolution (e.g., 'min', 'max', 'nearest', 'min_max').
    :returns ResolvedvariableSubset:
        A dataclass containing resolved variable indices and metadata.
    """

    variable: int | float | Tuple[int | float, int | float] | None
    variable_indices: List[int | float]
    variable_dimensions: Any
    method: str | None


class VariableSubsetError(ValueError):
    """
    Raised when variable subset resolution fails.

    :param message:
        Description of the error.
    :returns VariableSubsetError:
        Exception indicating invalid variable subset configuration.
    """


def validate_inputs(func):
    @wraps(func)
    def checker(
        self,
        data,
        variable_params,
        accuracy: Optional[float] = 0,
        *args,
        **kwargs,
    ):
        # Validate data
        if not hasattr(data, "size"):
            raise VariableSubsetError("Data must have a 'size' attribute.")
        if data.size == 0:
            raise VariableSubsetError("Data cannot be empty.")

        if not isinstance(data.size, int):
            raise VariableSubsetError("Multi-dimensional variable arrays are not supported yet.")

        # Validate variable_params
        valid_keys = {"min", "max", "nearest"}
        if not any(k in variable_params for k in valid_keys):
            raise VariableSubsetError(f"variable_params must contain one of {valid_keys}, got {variable_params.keys()}")

        # Validate tolerance if present
        if "tolerance" in variable_params and not isinstance(
            variable_params["tolerance"], (int, float, np.timedelta64)
        ):
            raise VariableSubsetError("Tolerance must be numeric.")

        # Validate accuracy
        if accuracy is not None and accuracy < 0:
            raise VariableSubsetError("Accuracy should not be negative.")

        # -----------------------------------------------------------------
        # check that if max, min or nearest then value is just a single value
        return func(self, data, variable_params, accuracy, *args, **kwargs)

    return checker


# -----------------------------------------------------------------------------------
class BaseSubsetResolver(ABC):
    """
    Template for resolvers that produce a subset of indices based on variable parameters.
    Subclasses implement how indices are computed for min/max/nearest and 1D indices.
    """

    @validate_inputs
    def __init__(
        self,
        data: Any,
        variable_params: Dict,
        accuracy: Optional[float] = 0,
    ):
        self.data: Any = data
        self.variable_params: Dict = variable_params
        self.method: str | None = None
        self.variable: int | float | Tuple[int | float, int | float] | None = None
        self.accuracy = accuracy

    def run(self) -> ResolvedVariableSubset:
        """
        Resolve variable indices based on provided parameters.

        :param variable:
            variable or variable range to resolve.
        :param method:
            Resolution method ('min', 'max', 'nearest', 'min_max').
        :returns ResolvedvariableSubset:
            Object containing resolved variable indices and metadata.
        """

        self.variable_indices = np.arange(self.data.size)

        if "max" in self.variable_params and "min" in self.variable_params:
            resolved_variable_indices = self.get_min_max()
        elif "max" in self.variable_params:
            resolved_variable_indices = self.get_max()
        elif "min" in self.variable_params:
            resolved_variable_indices = self.get_min()
        elif "nearest" in self.variable_params:
            resolved_variable_indices = self.get_nearest()

        return ResolvedVariableSubset(
            variable=self.variable,
            variable_indices=resolved_variable_indices,
            variable_dimensions=self.extract_dimensions(self.data),
            method=self.method,
        )

    def get_min(self):
        """
        Resolve indices for variables greater than or equal to 'min'.

        :param variable:
            Minimum variable value.
        :param method:
            Resolution method ('min').
        :returns np.ndarray:
            Array of indices satisfying the minimum variable condition.
        """

        # --------------------------------------------------------------------------------------------------------------
        self.variable = self.variable_params["min"]
        self.method = "min"

        if self.data.values.dtype == "datetime64[ns]":
            variable_indices_min = np.where(convert_datetime(self.data.values) >= self.variable)[
                0
            ]  # min variable index
        else:
            try:
                variable_indices_min = np.where(self.data.values >= self.variable - self.accuracy)[
                    0
                ]  # min variable index
            except TypeError:
                variable_indices_min = np.where(convert_datetime(self.data.values) >= self.variable)[
                    0
                ]  # min variable index

        if len(self.variable_indices) == 0:
            resolved_variable_indices = variable_indices_min
        else:
            resolved_variable_indices = np.intersect1d(self.variable_indices, variable_indices_min)
        return resolved_variable_indices

    def get_max(self):
        """
        Resolve indices for variable less than or equal to 'max'.

        :param variable:
            Maximum variable value.
        :param method:
            Resolution method ('max').
        :returns np.ndarray:
            Array of indices satisfying the maximum variable condition.
        """

        self.variable = self.variable_params["max"]
        self.method = "max"
        # --------------------------------------------------------------------------------------------------------------
        if self.data.values.dtype == "datetime64[ns]":
            variable_indices_max = np.where(convert_datetime(self.data.values) <= self.variable)[
                0
            ]  # min variable index
        else:
            try:
                variable_indices_max = np.where(self.data.values <= self.variable + self.accuracy)[
                    0
                ]  # min variable index
            except Exception:
                try:
                    variable_indices_max = np.where(self.data.values <= self.variable)[0]  # min variable index
                except Exception:
                    from datetime import timezone

                    if self.variable.tzinfo == timezone.utc:
                        data_vals = np.array(
                            [d.replace(tzinfo=timezone.utc) for d in convert_datetime(self.data.values)]
                        )
                    else:
                        data_vals = self.data.values
                    variable_indices_max = np.where(data_vals <= self.variable)[0]  # min variable index

        if len(self.variable_indices) == 0:  # this is never the case, should if it's empty
            resolved_variable_indices = variable_indices_max
        else:
            resolved_variable_indices = np.intersect1d(
                self.variable_indices, variable_indices_max
            )  # Sorted 1D array of common and unique elements.

        return resolved_variable_indices

    def get_nearest(self):
        """
        Resolve indices for the variable nearest to the target value.

        :param variable:
            Target variable value.
        :param method:
            Resolution method ('nearest').
        :returns np.ndarray:
            Array of indices closest to the target variable within tolerance.
        """

        # ---------------------------------------------------------------------------------------------------
        self.variable = self.variable_params["nearest"]
        self.method = "nearest"

        tolerance = self.variable_params["tolerance"] if "tolerance" in self.variable_params.keys() else None

        # Ensure 1D view
        values = np.asarray(self.data.values).ravel()

        # Compute absolute differences once
        if values.dtype == "datetime64[ns]":
            diffs = np.abs(convert_datetime(values) - self.variable)
        else:
            diffs = np.abs(values - self.variable)

        # Always compute the nearest index once and reuse it
        nearest_idx = int(np.argmin(diffs))  # variable_indices_nearest
        nearest_diff = diffs[nearest_idx]

        if tolerance is None:
            # No tolerance -> just the nearest
            if self.accuracy == 0:
                resolved = np.flatnonzero(diffs <= nearest_diff)
            else:
                resolved = np.flatnonzero(diffs <= nearest_diff + self.accuracy).astype(int)
        else:
            # With tolerance -> all indices within tolerance
            within = np.flatnonzero(diffs <= tolerance)

            if within.size == 0:
                # No match within tolerance -> error
                if isinstance(tolerance, np.timedelta64):
                    raise VariableSubsetError(
                        f"No variable within tolerance. Nearest diff={nearest_diff}, tolerance={tolerance}"
                    )
                else:
                    raise VariableSubsetError(
                        f"No variable within tolerance. Nearest diff={nearest_diff:.6g}, tolerance={tolerance}"
                    )
            resolved = within.astype(int)

        resolved = np.intersect1d(resolved, np.asarray(self.variable_indices, dtype=int))

        if resolved.size == 0:
            # Optional: decide behavior; here we raise for clarity
            raise VariableSubsetError("No indices remain after intersecting with allowed_indices.")

        return resolved

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

        # call min
        variable_indices_min = self.get_min()
        # call max
        variable_indices_max = self.get_max()
        # resolve
        resolved_variable_indices = np.intersect1d(variable_indices_min, variable_indices_max)
        self.variable = (self.variable_params["min"], self.variable_params["max"])
        self.method = "min_max"
        return resolved_variable_indices

    @staticmethod
    def extract_dimensions(data):
        return getattr(data, "dims", None)


if __name__ == "__main__":
    pass
