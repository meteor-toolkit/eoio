"""eoio.processors.units.drivers.planetscope - unit conversion for PlanetScope products"""

from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Set, Tuple
import xarray as xr
from eoio.processors.units.drivers.base import BaseUnitConversionDriver
from eoio.processors.units.drivers.registry import register_unit_driver
import logging

logger = logging.getLogger(__name__)


@register_unit_driver("planetscope")
class PlanetScopeUnitDriver(BaseUnitConversionDriver):
    """
    PlanetScope unit conversion driver.

    Intended scope:
    - radiance -> reflectance, using attributes available in the product
    - reflectance -> radiance, using attributes available in the product
    - reflectance -> DNs, using attributes available in the product
    - radiance -> DNs, using attributes available in the product

    This driver assumes eoio readers provide enough product identification to
    allow cheap matching (e.g. ds.attrs['eoio:product'] in {"planetscope"}).

    Notes
    -----
    PlanetScope unit conversion depends on:
    - per-band reflectance coefficients

    Where those values live should be standardised in dataset and variable attrs.
    """

    name: str = "planetscope"

    def matches(self, ds: xr.Dataset, context: Mapping[str, Any]) -> bool:
        """
        Determine whether this driver can handle the given dataset.

        Matching should be cheap and unambiguous; avoid heavy inspection.

        :param ds:
            Input dataset.
        :param context:
            Processing context provided by eoio.
        :return:
            True if this looks like a PlanetScope dataset, otherwise False.
        """

        return ds.attrs["eoio:reader"].lower() == self.name

    def supported(self) -> Set[Tuple[str, str]]:
        """
        Return supported conversion pairs.

        :return:
            Set of supported (from_unit, to_unit) pairs.
        """
        return {
            ("radiance", "reflectance"),
            ("radiance", "DNs"),
            ("reflectance", "DNs"),
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
        Converts the data using conversion coefficients found in metadata (reflectance coefficient, radiometric scale factor) and updates units attributes accordingly.

        :param ds:
            Input dataset.
        :param to:
            Target unit name (e.g. "radiance").
        :param var_names:
            Variables to convert. If None, default is all measurement vars.
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
                            f"PlanetScopeUnitDriver: Selected Var not in the dataset: {var}. Vars on the dataset include: {str(vars_list)}"
                        )
        else:
            var_names = [str(var) for var in ds.data_vars if str(var).startswith("B")]

        if to == "DNs":
            # get radiometric scale factor and reflectance coefficient from meas_var attrs and apply to variable data
            for meas_var in var_names:
                # print(f"Converting {meas_var} to digital numbers (DNs)...")
                logger.debug(f"Converting {meas_var} to digital numbers (DNs)")  # for developer
                logger.info(f"Converting {meas_var} to digital numbers (DNs)")  # for user

                rad_coeff = ds[meas_var].attrs["product_metadata"].get("radiometric_scale_factor", None)
                refl_coeff = ds[meas_var].attrs["product_metadata"].get("reflectance_coefficient", None)

                if rad_coeff is None:
                    raise KeyError(
                        f"Variable '{meas_var}' attrs missing 'radiometric_scale_factor' needed for conversion."
                    )

                # check what units to convert from

                if "radiance" in ds[meas_var].attrs["product_metadata"]["measurand"].lower():
                    ds[meas_var].values = ds[meas_var].values / float(rad_coeff)
                elif "reflectance" in ds[meas_var].attrs["product_metadata"]["measurand"].lower():
                    if refl_coeff is None:
                        raise KeyError(
                            f"Variable '{meas_var}' attrs missing 'reflectance_coefficient' needed for conversion."
                        )
                    else:
                        ds[meas_var].values = ds[meas_var].values / float(refl_coeff)
                else:
                    raise ValueError(
                        f"PlanetScopeUnitDriver: unsupported conversion from '{ds[meas_var].attrs['product_metadata']['measurand']}'. Supported: {self.supported()}"
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
            # get radiometric scale factor and reflectance coefficient from meas_var attrs and apply to variable data
            for meas_var in var_names:
                # print(f"Converting {meas_var} to reflectance...")
                logger.debug(f"Converting {meas_var} to reflectance")  # for developer
                logger.info(f"Converting {meas_var} to reflectance")  # for user

                refl_coeff = ds[meas_var].attrs["product_metadata"].get("reflectance_coefficient", None)
                rad_coeff = ds[meas_var].attrs["product_metadata"].get("radiometric_scale_factor", None)

                if refl_coeff is None:
                    raise KeyError(
                        f"Variable '{meas_var}' attrs missing 'reflectance_coefficient' needed for conversion."
                    )

                if "radiance" in ds[meas_var].attrs["product_metadata"]["measurand"].lower():
                    if rad_coeff is None:
                        raise KeyError(
                            f"Variable '{meas_var}' attrs missing 'radiometric_scale_factor' needed for conversion."
                        )
                    else:
                        ds[meas_var].values = ds[meas_var].values * (float(refl_coeff) / float(rad_coeff))
                else:
                    raise ValueError(
                        f"PlanetScopeUnitDriver: unsupported conversion from '{ds[meas_var].attrs['product_metadata']['measurand']}'. Supported: {self.supported()}"
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
            raise ValueError(f"PlanetScopeUnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}")

        return ds


if __name__ == "__main__":
    pass
