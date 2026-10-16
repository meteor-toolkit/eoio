"""eoio.processors.units.drivers.constellr_skybee - unit conversion for constellr Skybee products"""

from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Set, Tuple
import xarray as xr
from eoio.processors.units.drivers.base import BaseUnitConversionDriver
from eoio.processors.units.drivers.registry import register_unit_driver
from processor_tools.utils.dict_tools import get_value
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)


@register_unit_driver("constellr_skybee")
class ConstellrSkybeeUnitDriver(BaseUnitConversionDriver):
    """
    constellr Skybee unit conversion driver.

    Intended scope:
    - radiance -> reflectance, using attributes available in the product

    This driver assumes eoio readers provide enough product identification to
    allow cheap matching (e.g. ds.attrs['eoio:product'] in {"constellr_skybee"}).

    Notes
    -----
    constellr Skybee unit conversion depends on:
    - solar_zenith_angle & solar irradiance

    Where those values live should be standardised in dataset and variable attrs.
    """

    name: str = "constellr_skybee"

    def matches(self, ds: xr.Dataset, context: Mapping[str, Any]) -> bool:
        """
        Determine whether this driver can handle the given dataset.

        Matching should be cheap and unambiguous; avoid heavy inspection.

        :param ds:
            Input dataset.
        :param context:
            Processing context provided by eoio.
        :return:
            True if this looks like a constellr Skybee dataset, otherwise False.
        """

        # .get: every registered driver is asked about every dataset, including ones with no eoio:reader
        return str(ds.attrs.get("eoio:reader", "")).lower() == self.name

    def supported(self) -> Set[Tuple[str, str]]:
        """
        Return supported conversion pairs.

        :return:
            Set of supported (from_unit, to_unit) pairs.
        """
        return {
            ("radiance", "reflectance"),
        }

    def convert(
        self,
        ds: xr.Dataset,
        to: str,
        var_names: Optional[Sequence[str]],
        *,
        context: Mapping[str, Any],
    ) -> xr.Dataset:
        """
        Convert selected variables in the dataset to the requested target units, based on supported conversion pairs.
        If no variables are selected, the default is all measurement vars.
        Converts the data using conversion coefficients found in metadata (per band gain and bias, solar zenith, etc.) and updates units attributes accordingly.

        :param ds:
            Input dataset.
        :param to:
            Target unit name (e.g. "reflectance").
        :param var_names:
            Variables to convert. If None default is all measurement vars.
        :param context:
            Processing context provided by eoio, passed to all processors.
        :return:
            Dataset with converted variables.
        :raises ValueError:
            If an unsupported conversion is requested.
        :raises KeyError:
            If requested variables are not present.
        """

        vars_list = [str(var) for var in ds.data_vars if str(var).startswith("B")]
        if var_names is None or var_names == "all":
            var_names = vars_list  # default: every measurement variable
        else:
            for var in var_names:
                if var not in ds.data_vars:
                    raise KeyError(
                        f"ConstellrSkybeeUnitDriver: Selected Var not in the dataset: {var}. Vars on the dataset include: {str(vars_list)}"
                    )

        if to == "reflectance":
            # get band gain, bias and solar irradiance from meas_var attrs and apply to variable data
            for meas_var in var_names:
                # print(f"Converting {meas_var} to reflectance...")
                logger.debug(f"Converting {meas_var} to reflectance")  # for developer
                logger.info(f"Converting {meas_var} to reflectance")  # for user

                bnd_solar_irrad = ds[meas_var].attrs["product_metadata"].get("band_solar_irradiance", None)
                solar_zenith_angle: Any = None

                # get solar zenith angle from ds or global attrs
                # (one copy per grid, "solar_zenith_angle_<res>", when bands on several grids were read)
                sza_vars = [
                    str(v)
                    for v in ds.data_vars
                    if str(v) == "solar_zenith_angle" or str(v).startswith("solar_zenith_angle_")
                ]
                if sza_vars:  # if angles extracted
                    solar_zenith_angle = ds[sza_vars[0]].values
                else:  # get from ds attrs
                    located = get_value(ds.attrs, "observation_geometry")
                    if located is not None:
                        solar_zenith_angle = float(located["solar_zenith_angle"])

                if bnd_solar_irrad is None:
                    raise KeyError(
                        f"Variable '{meas_var}' attrs missing 'band_solar_irradiance' needed for conversion."
                    )
                if solar_zenith_angle is None:
                    raise KeyError(
                        "Solar zenith angle needed for conversion not found. \
                                    Update eoio.read() to select at least one of the following: \
                                    'vars_sel': {'aux': {'observation_geometry':True}} or \
                                    'read_params'={'metadata_level': 'basic'}"
                    )

                if "radiance" in ds[meas_var].attrs["product_metadata"]["measurand"].lower():
                    # calc TOA reflectance from solar irradiance values, solar zenith angle and Earth-Sun distance, following calc here: https://seadas.gsfc.nasa.gov/help-9.0.0/rad2refl/Rad2ReflAlgorithmSpecification.html
                    # d^2 scales the irradiance to the Earth-Sun distance (au) at acquisition
                    earth_sun_distance = ds[meas_var].attrs["product_metadata"].get("earth_sun_distance", None)
                    if earth_sun_distance is None:
                        warnings.warn(
                            f"{meas_var}: no earth_sun_distance in the product metadata; assuming 1 au "
                            "(up to ~3 % error in reflectance depending on the time of year)."
                        )
                    d_squared = float(earth_sun_distance) ** 2 if earth_sun_distance is not None else 1.0
                    # assign to .data (not .values) so dask-backed bands stay lazy
                    ds[meas_var].data = (np.pi * ds[meas_var].data * d_squared) / (
                        float(bnd_solar_irrad) * np.cos(np.deg2rad(solar_zenith_angle))
                    )
                else:
                    raise ValueError(
                        f"ConstellrSkybeeUnitDriver: unsupported conversion from '{ds[meas_var].attrs['product_metadata']['measurand']}'. Supported: {self.supported()}"
                    )

                semantics = {
                    "standard_name": "toa_reflectance",
                    "long_name": f"TOA reflectance in band {meas_var}",
                    "measurand": "reflectance",
                    "units": 1,
                }
                ds[meas_var].attrs.update(semantics)
                ds[meas_var].attrs["product_metadata"].update(semantics)

        else:
            raise ValueError(
                f"ConstellrSkybeeUnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}"
            )

        return ds


if __name__ == "__main__":
    pass
