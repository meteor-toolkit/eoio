"""eoio.processors.units.drivers.airbus_pleiades - unit conversion for AirbusPleiades products"""

from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Set, Tuple
import xarray as xr
from eoio.processors.units.drivers.base import BaseUnitConversionDriver
from eoio.processors.units.drivers.registry import register_unit_driver
from processor_tools.utils.dict_tools import get_value
import numpy as np
import logging

logger = logging.getLogger(__name__)


@register_unit_driver("airbus_pleiades")
class AirbusPleiadesUnitDriver(BaseUnitConversionDriver):
    """
    AirbusPleiades unit conversion driver.

    Intended scope:
    - radiance -> reflectance, using attributes available in the product
    - radiance -> DNs, using attributes available in the product

    This driver assumes eoio readers provide enough product identification to
    allow cheap matching (e.g. ds.attrs['eoio:product'] in {"airbus_pleiades"}).

    Notes
    -----
    AirbusPleiades unit conversion depends on:
    - per-band reflectance coefficients

    Where those values live should be standardised in dataset and variable attrs.
    """

    name: str = "airbus_pleiades"

    def matches(self, ds: xr.Dataset, context: Mapping[str, Any]) -> bool:
        """
        Determine whether this driver can handle the given dataset.

        Matching should be cheap and unambiguous; avoid heavy inspection.

        :param ds:
            Input dataset.
        :param context:
            Processing context provided by eoio.
        :return:
            True if this looks like a AirbusPleiades dataset, otherwise False.
        """

        return ds.attrs["eoio:reader"].lower() == self.name

    def supported(self) -> Set[Tuple[str, str]]:
        """
        Return supported conversion pairs.

        :return:
            Set of supported (from_unit, to_unit) pairs.
        """
        return {
            ("radiance", "DNs"),
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

        if var_names:
            if var_names == "all":
                var_names = [str(var) for var in ds.data_vars if str(var).startswith("B")]
            elif isinstance(var_names, list):
                vars_list = [str(var) for var in ds.data_vars if str(var).startswith("B")]
                for var in var_names:
                    if var not in ds.data_vars:
                        raise KeyError(
                            f"AirbusPleiadesUnitDriver: Selected Var not in the dataset: {var}. Vars on the dataset include: {str(vars_list)}"
                        )
        else:
            var_names = [str(var) for var in ds.data_vars if str(var).startswith("B")]

        if to == "DNs":
            # get band gain and bias from var attrs and apply to variable data
            for meas_var in var_names:
                # print(f"Converting {meas_var} to digital numbers (DNs)...")
                logger.debug(f"Converting {meas_var} to digital numbers (DNs)")  # for developer
                logger.info(f"Converting {meas_var} to digital numbers (DNs)")  # for user

                bnd_gain = ds[meas_var].attrs["product_metadata"].get("band_gain", None)
                bnd_bias = ds[meas_var].attrs["product_metadata"].get("band_bias", None)

                if bnd_gain is None:
                    raise KeyError(f"Variable '{meas_var}' attrs missing 'band_gain' needed for conversion.")
                if bnd_bias is None:
                    raise KeyError(f"Variable '{meas_var}' attrs missing 'band_bias' needed for conversion.")

                if "radiance" in ds[meas_var].attrs["product_metadata"]["measurand"].lower():
                    ds[meas_var].values = (ds[meas_var].values - float(bnd_bias)) * float(bnd_gain)
                else:
                    raise ValueError(
                        f"AirbusPleiadesUnitDriver: unsupported conversion from {ds[meas_var].attrs['product_metadata']['measurand']}. Supported: {self.supported()}"
                    )

                semantics = {
                    "standard_name": "toa_digital_number",
                    "long_name": f"TOA digital numbers in band {meas_var}",
                    "measurand": "DNs",
                    "units": 1,
                }
                ds[meas_var].attrs.update(semantics)
                ds[meas_var].attrs["product_metadata"].update(semantics)

        elif to == "reflectance":
            # get band gain, bias and solar irradiance from meas_var attrs and apply to variable data
            for meas_var in var_names:
                # print(f"Converting {meas_var} to reflectance...")
                logger.debug(f"Converting {meas_var} to reflectance")  # for developer
                logger.info(f"Converting {meas_var} to reflectance")  # for user

                bnd_solar_irrad = ds[meas_var].attrs["product_metadata"].get("band_solar_irradiance", None)
                solar_zenith_angle = None

                # get solar zenith angle from ds or global attrs
                if "solar_zenith_angle" in ds.data_vars:  # if angles extracted
                    solar_zenith_angle = np.mean(ds["solar_zenith_angle"].values)
                else:  # get from ds attrs
                    located = get_value(ds.attrs, "located_geometric_values")
                    if located is not None:
                        solar_elevation = located["SUN_ELEVATION"]["#text"]
                        solar_zenith_angle = 90 - float(solar_elevation)

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
                    # calc TOA reflectance from solar irradiance values and solar zenith angle, following calc here: https://seadas.gsfc.nasa.gov/help-9.0.0/rad2refl/Rad2ReflAlgorithmSpecification.html
                    ds[meas_var].values = (np.pi * ds[meas_var].values) / (
                        float(bnd_solar_irrad) * np.cos(np.deg2rad(solar_zenith_angle))
                    )
                else:
                    raise ValueError(
                        f"AirbusPleiadesUnitDriver: unsupported conversion from '{ds[meas_var].attrs['product_metadata']['measurand']}'. Supported: {self.supported()}"
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
                f"AirbusPleiadesUnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}"
            )

        return ds


if __name__ == "__main__":
    pass
