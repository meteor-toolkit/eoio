"""
eoio.processors.units.processor
-------------------------------

Top-level unit conversion processor.

This processor provides a single, stable user-facing interface (``units.convert``)
and delegates product-specific conversion logic to registered unit conversion
drivers (e.g. Sentinel-2, Landsat).

User config example
-------------------
processors=[
  {"name": "units.convert", "params": {"to": "radiance", "var_names": ["B02", "B03"]}}
]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
from processor_tools import BaseProcessor
import xarray as xr
from eoio.processors.units.drivers.base import BaseUnitConversionDriver
from eoio.processors.units.drivers.registry import UNIT_DRIVER_REGISTRY
from eoio.processors.registry import register_processor


@dataclass(frozen=True)
class UnitsConvertConfig:
    """
    Parameters for the units.convert processor.
    """

    to: str
    var_names: Optional[Sequence[str]] = None
    on_missing: str = "error"  # "error" | "skip"
    driver: Optional[str] = None  # optional override by driver name


@register_processor("units.convert")
class UnitsConvert(BaseProcessor):
    """
    Convert measurement variables in a dataset to a requested unit.

    This processor:
    - provides a stable interface across products (Sentinel-2, Landsat, etc.)
    - selects an appropriate unit conversion driver using ``driver.matches(ds, context)``
    - delegates conversion details to the selected driver

    Processor parameters
    --------------------

    The following parameters can be provided in the `params` dict:

    :param to:
        Target unit name (e.g. ``"radiance"`` or ``"reflectance"``).
        Must be supported by the selected driver.
    :param var_names:
        Optional list of variable names to convert.
        If omitted, the driver converts all measurements variables
    :param on_missing:
        Behaviour if required metadata for conversion is missing.
        Supported values are ``"error"`` (default, if omitted) or ``"skip"``.
    :param driver:
        Optional explicit driver name to use (e.g. ``"sentinel2"``).
        If provided, automatic driver matching is bypassed.

    Notes
    -----
    - This processor is intended to run after reading.
    - Product-specific metadata requirements are owned by the driver.
    - If multiple drivers match, this processor raises a clear error unless
      the user pins a driver explicitly via the ``driver`` param.
    """

    _all_options = {
        "to": "Name of units to convert the measurement variables to (e.g. 'radiance'). If multiple types of measurement variables, then a nested dictionary for each type (e.g. 'to':{'optical': 'radiance', 'thermal': 'brightness_temperature'}).",
        "var_names": "List of measurement variables as strings to convert the units for.",
        "on_missing": "Behaviour if required metadata for conversion is missing. Supported values are 'error' (default, if omitted) or 'skip'.",
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a unit conversion processor.

        :param params:
            Processor parameters (see class docstring for details)
        :param context:
            Processing context provided by eoio (reader info, metadata view, logger, etc.).
        """

        super().__init__(context=context)
        self.units_convert_config = self._parse_params(params or {})

    def _parse_params(self, params: Dict[str, Any]) -> UnitsConvertConfig:
        """
        Validate and normalise processor parameters.

        :param params:
            Raw params dict from the processor spec.
        :return:
            Parsed UnitsConvertParams.
        :raises ValueError:
            If required parameters are missing or invalid.
        """

        # resolve "to" param
        if "to" not in params:
            raise ValueError("units.convert: missing required param 'to' (e.g. 'radiance').")

        to = params["to"]

        # resolve "var_names" param
        var_names = params.get("var_names", None)
        if var_names not in [None, "all"]:
            if not isinstance(var_names, (list, tuple)) or not all(isinstance(v, str) for v in var_names):
                raise ValueError("units.convert: 'var_names' must be a list[str], 'all' or None.")

        # resolve "on_missing" param
        on_missing = str(params.get("on_missing", "error")).lower()
        if on_missing not in {"error", "skip"}:
            raise ValueError("units.convert: 'on_missing' must be 'error' or 'skip'.")

        # resolve "driver" param
        driver = params.get("driver", None)
        if driver is not None:
            driver = str(driver)

        return UnitsConvertConfig(
            to=to,
            var_names=var_names,
            on_missing=on_missing,
            driver=driver,
        )

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Run unit conversion on the dataset.

        :param ds:
            Input dataset.
        :return:
            Output dataset with converted variables (if applicable).
        :raises ValueError:
            If no driver matches, multiple drivers match, or conversion is unsupported.
        """

        if not isinstance(ds, xr.Dataset):
            raise TypeError("units.convert: input must be an xarray.Dataset.")

        context: Mapping[str, Any] = self.context or {}

        driver = self._select_driver(ds, context=context)

        # Delegate conversion to the driver.
        # Drivers may raise KeyError / ValueError for missing vars/metadata; allow those to propagate.
        ds_out = driver.convert(
            ds,
            to=self.units_convert_config.to,
            var_names=self.units_convert_config.var_names,
            context=context,
        )

        # Record processing history
        ds_out = self._record_provenance(ds_out, driver=driver)

        return ds_out

    def _select_driver(self, ds: xr.Dataset, *, context: Mapping[str, Any]) -> BaseUnitConversionDriver:
        """
        Select a unit conversion driver for the dataset.

        Selection rules:
        - If ``params['driver']`` is provided, select that driver by name from the registry.
        - Otherwise, select the unique driver whose ``matches(ds, context)`` returns True.
        - If none match, raise.
        - If multiple match, raise and ask the user to pin a driver.

        :param ds:
            Input dataset.
        :param context:
            Processing context.
        :return:
            Selected driver instance.
        :raises ValueError:
            If selection fails or is ambiguous.
        """

        # 1) Explicit driver selection
        if self.units_convert_config.driver is not None:
            try:
                driver_cls = UNIT_DRIVER_REGISTRY[self.units_convert_config.driver]
            except KeyError as e:
                available = sorted(UNIT_DRIVER_REGISTRY.keys())
                raise ValueError(
                    f"units.convert: unknown driver {self.units_convert_config.driver!r}. "
                    f"Available drivers: {available}"
                ) from e
            return driver_cls()

        # 2) Automatic matching
        matches: list[BaseUnitConversionDriver] = []
        for name, driver_cls in UNIT_DRIVER_REGISTRY.items():
            driver = driver_cls()
            try:
                if driver.matches(ds, context):
                    matches.append(driver)
            except Exception:
                # matches() should be cheap and should not raise; treat any exception as non-match
                continue

        if not matches:
            available = sorted(UNIT_DRIVER_REGISTRY.keys())
            raise ValueError(
                f"units.convert: no unit conversion driver matched this dataset. Available drivers: {available}"
            )

        if len(matches) > 1:
            names = sorted(getattr(d, "name", "unknown") for d in matches)
            raise ValueError(
                "units.convert: multiple unit conversion drivers matched this dataset. "
                f"Matched drivers: {names}. "
                "Please specify params['driver'] to pin one."
            )

        return matches[0]

    def _record_provenance(self, ds: xr.Dataset, *, driver: BaseUnitConversionDriver) -> xr.Dataset:
        """
        Record a minimal provenance entry at processor level.

        Drivers may also record provenance. This method should remain lightweight
        to avoid conflicting conventions.

        :param ds:
            Output dataset.
        :param driver:
            Driver used for conversion.
        :return:
            Dataset (same object, attrs updated).
        """

        steps: list = ds.attrs.get("eoio:processing_steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]

        steps.append(
            {
                "processor": "units.convert",
                "to": self.units_convert_config.to,
                "vars": (
                    list(self.units_convert_config.var_names)
                    if self.units_convert_config.var_names is not None
                    else None
                ),
                "driver": getattr(driver, "name", "unknown"),
            }
        )
        ds.attrs["eoio:processing_steps"] = steps
        return ds


if __name__ == "__main__":
    pass
