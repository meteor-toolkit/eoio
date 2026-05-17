"""eoio.processors.units.drivers.l89 - unit conversion for Landsat 89 L1 products"""

from __future__ import annotations
from typing import Any, Mapping, Sequence, Set, Optional, Tuple
import numpy as np
import xarray as xr
from eoio.processors.units.drivers.base import BaseUnitConversionDriver
from eoio.processors.units.drivers.registry import register_unit_driver
from processor_tools.utils.dict_tools import get_value
import warnings


import logging

logger = logging.getLogger(__name__)


@register_unit_driver("landsatc2l1")
class LANDSATC2L1UnitDriver(BaseUnitConversionDriver):
    """
    Landsat 89 L1 unit conversion driver.

    Intended scope:
    - reflectance -> radiance, using metadata available in the product
    - radiance -> brightness temperature, using metadata available in the product

    This driver assumes eoio readers provide enough product identification to
    allow cheap matching (e.g. ds.attrs['eoio:product'] in {"BrightnessTemperatureL1", "ReflectanceL1"}).

    Notes
    -----
    Landsat 89 L1 unit conversion depends on:
    - solar zenith angle (or sun elevation) at acquisition time / tile
    - reflectance/radiance scaling factors
    - brightness temperature constants, e.g. k1, k2

    Where those values live should be standardised in dataset and variable attrs.
    """

    name: str = "LANDSATC2L1"

    def matches(self, ds: xr.Dataset, context: Mapping[str, Any]) -> bool:
        """
        Determine whether this driver can handle the given dataset.

        Matching should be cheap and unambiguous; avoid heavy inspection.

        :param ds:
            Input dataset.
        :param context:
            Processing context provided by eoio.
        :return:
            True if this looks like a Landsat 89 L1 dataset, otherwise False.
        """
        coll_name_list = ds.attrs["collection_name"].lower().split("_")
        coll_name = ""
        for prod_str in coll_name_list:
            coll_name += prod_str

        return coll_name == self.name

    def supported(self) -> Set[Tuple[str, str]]:
        """
        Return supported conversion pairs.

        :return:
            Set of supported (from_unit, to_unit) pairs.
        """
        return {  # type: ignore[return-value]
            "vnir": ("reflectance", "radiance"),
            "tir": ("radiance", "brightness_temperature"),
        }

    def convert(  # type: ignore[override]
        self,
        ds: xr.Dataset,
        to: dict,
        var_names: Optional[Sequence[str]],
        *,
        context: Mapping[str, Any],
    ) -> xr.Dataset:
        """
        Convert selected variables in the dataset to the requested target units, based on supported conversion pairs.
        If no variables are selected, the default is all measurement vars.

        Converts the data using conversion coefficients found in metadata (radiance_mult, radiance_add, etc.) and updates units attributes accordingly.

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
                for var in var_names:
                    if var not in ds.data_vars:
                        raise KeyError(
                            f"L89UnitDriver: Selected Var not in the dataset: {var}. Vars on the dataset include: {str([str(var) for var in ds.data_vars if str(var).startswith('B')])}"
                        )
        else:
            var_names = [str(var) for var in ds.data_vars if str(var).startswith("B")]

        if "vnir" in to.keys() and to["vnir"] == "radiance":
            # undo reflectance scaling and apply radiometric scaling to radiance for vnir bands, and leave for tir bands alone (already in radiance)
            for meas_var in var_names:
                if any(["reflectance" in i for i in ds[meas_var].attrs.values() if isinstance(i, str)]):
                    if meas_var not in ["B10", "B11"]:
                        # print(f"Converting {meas_var} to radiance...")
                        logger.debug(f"Converting {meas_var} to radiance")  # for developer
                        logger.info(f"Converting {meas_var} to radiance")  # for user

                        var_mtd = ds[meas_var].attrs

                        # first convert back to DN: DN = (Reflectance * sin(SUN_ELEVATION) - REFLECTANCE_ADD_BAND_x)/REFLECTANCE_MULT_BAND_x

                        # try per-pixel solar angles if available & if not in ds, use sun_elevation from metadata
                        if "solar_zenith_angle" in ds:
                            # TODO: handle case where grid coords do not align - for non-30m band
                            try:
                                ds[meas_var] *= np.sin(
                                    np.deg2rad(90.0 - ds["solar_zenith_angle"].values)
                                )  # per-pixel SZA
                            except Exception:
                                warnings.warn(
                                    "Per-pixel solar zenith angle correction failed; using metadata sun elevation instead."
                                )
                                ds[meas_var] *= np.sin(
                                    np.deg2rad(ds.attrs["product_metadata"].get("sun_elevation"))
                                )  # Perhaps a warning should be raised here as SZA used is less accurate as not per pixel ?
                        else:
                            warnings.warn(
                                "Per-pixel solar zenith angle correction not available; using metadata sun elevation instead."
                            )
                            ds[meas_var] *= np.sin(np.deg2rad(ds.attrs["product_metadata"].get("sun_elevation")))

                        ds[meas_var] -= var_mtd.get("reflectance_add", 0.0)
                        ds[meas_var] /= var_mtd.get("reflectance_mult", 1.0)

                        # Radiance (float) = (RADIANCE_MULT_BAND_x * DN) + RADIANCE_ADD_BAND_x
                        ds[meas_var] *= var_mtd.get("radiance_mult", 1.0)
                        ds[meas_var] += var_mtd.get("radiance_add", 0.0)

                        # Change attributes to 'radiance'
                        ds[meas_var].attrs.update(
                            {
                                "standard_name": "toa_radiance",
                                "long_name": f"Top of atmosphere radiance in band {meas_var}",
                                "measurand": "radiance",
                                "units": "W/( m² * sr * μm)",
                            }
                        )

        if "tir" in to.keys() and to["tir"] == "brightness_temperature":
            # convert tir bands to brightness temperature
            for meas_var in var_names:
                if meas_var in ["B10", "B11"]:
                    # print(f"Converting {meas_var} to brightness temperature...")
                    logger.debug(f"Converting {meas_var} to brightness temperature")  # for developer
                    logger.info(f"Converting {meas_var} to brightness temperature")  # for user

                    if any(["radiance" in i for i in ds[meas_var].attrs.values() if isinstance(i, str)]):
                        bnd_k1 = get_value(ds[meas_var].attrs["product_metadata"], "k1")
                        bnd_k2 = get_value(ds[meas_var].attrs["product_metadata"], "k2")

                        bnd_vars = ["k1", "k2"]
                        for idx, bnd_var in enumerate([bnd_k1, bnd_k2]):
                            if bnd_var is None:
                                raise KeyError(
                                    f"Variable '{meas_var}' attrs missing {bnd_vars[idx]} constant needed for conversion."
                                )
                        else:
                            # Brightness Temperature (float) = K2/(ln((K1/Radiance)+1)
                            ds[meas_var] = bnd_k2 / (np.log(bnd_k1 / (ds[meas_var] + 1)))

                            # Change attributes from to 'brightness_temperature'
                            ds[meas_var].attrs.update(
                                {
                                    "standard_name": "toa_brightness_temperature",
                                    "long_name": f"Top of Atmosphere brightness temperature in band {meas_var}",
                                    "measurand": "brightness_temperature",
                                    "units": "K",
                                }
                            )

        if not any(["tir" in to.keys(), "vnir" in to.keys()]):
            raise ValueError(f"L89UnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}")
        elif "vnir" in to.keys() and not to["vnir"] == "radiance":
            raise ValueError(f"L89UnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}")
        elif "tir" in to.keys() and not to["tir"] == "brightness_temperature":
            raise ValueError(f"L89UnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}")

        return ds


@register_unit_driver("landsatc2l2")
class LANDSATC2L2UnitDriver(BaseUnitConversionDriver):
    """
    Landsat 89 L2 unit conversion driver.

    Intended scope:
    - radiance -> brightness temperature, using metadata available in the product

    This driver assumes eoio readers provide enough product identification to
    allow cheap matching (e.g. ds.attrs['eoio:product'] in {"BrightnessTemperatureL2", "ReflectanceL2"}).

    Notes
    -----
    Landsat 89 L2 unit conversion depends on:
    - brightness temperature constants, e.g. k1, k2

    Where those values live should be standardised in dataset and variable attrs.
    """

    name: str = "LANDSATC2L2"

    def matches(self, ds: xr.Dataset, context: Mapping[str, Any]) -> bool:
        """
        Determine whether this driver can handle the given dataset.

        Matching should be cheap and unambiguous; avoid heavy inspection.

        :param ds:
            Input dataset.
        :param context:
            Processing context provided by eoio.
        :return:
            True if this looks like a Landsat 89 L2 dataset, otherwise False.
        """
        coll_name_list = ds.attrs["collection_name"].lower().split("_")
        coll_name = ""
        for prod_str in coll_name_list:
            coll_name += prod_str

        return coll_name == self.name

    def supported(self) -> Set[Tuple[str, str]]:
        """
        Return supported conversion pairs.

        :return:
            Set of supported (from_unit, to_unit) pairs.
        """
        return {  # type: ignore[return-value]
            "vnir": ("reflectance", "radiance"),  # not needed if done in L2 reader
            "tir": ("radiance", "brightness_temperature"),
        }

    def convert(  # type: ignore[override]
        self,
        ds: xr.Dataset,
        to: dict,
        var_names: Optional[Sequence[str]],
        *,
        context: Mapping[str, Any],
    ) -> xr.Dataset:
        """
        Convert selected variables in the dataset to the requested target units, based on supported conversion pairs.
        If no variables are selected, the default is all measurement vars.

        Converts the data using conversion coefficients found in metadata (radiance_mult, radiance_add, etc.) and updates units attributes accordingly.

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
                for var in var_names:
                    if var not in ds.data_vars:
                        raise KeyError(
                            f"L89UnitDriver: Selected Var not in the dataset: {var}. Vars on the dataset include: {str([str(var) for var in ds.data_vars if str(var).startswith('B')])}"
                        )
        else:
            var_names = [str(var) for var in ds.data_vars if str(var).startswith("B")]

        if "tir" in to.keys() and to["tir"] == "brightness_temperature":
            # convert tir bands to brightness temperature
            for meas_var in var_names:
                if meas_var in ["B10", "B11"]:
                    # print(f"Converting {meas_var} to brightness temperature...")
                    logger.debug(f"Converting {meas_var} to brightness temperature")  # for developer
                    logger.info(f"Converting {meas_var} to brightness temperature")  # for user

                    if any(["radiance" in i for i in ds[meas_var].attrs.values() if isinstance(i, str)]):
                        bnd_mult = get_value(
                            ds[meas_var].attrs["product_metadata"], "mult"
                        )  # TBC what this attr will be called, based on implementation of L2 reader
                        bnd_add = get_value(
                            ds[meas_var].attrs["product_metadata"], "add"
                        )  # TBC what this attr will be called, based on implementation of L2 reader

                        bnd_vars = ["mult", "add"]
                        for idx, bnd_var in enumerate([bnd_mult, bnd_add]):
                            if bnd_var is None:
                                raise KeyError(
                                    f"Variable '{meas_var}' attrs missing {bnd_vars[idx]} constant needed for conversion."
                                )
                        else:
                            # Brightness Temperature (float) = ((REFLECTANCE_MULT_BAND_x * DN) + REFLECTANCE_ADD_BAND_x)
                            ds[meas_var] = ds[meas_var] * bnd_mult + bnd_add

                            # Change attributes from 'radiance' to 'brightness temperature'
                            ds[meas_var].attrs.update(
                                {
                                    "standard_name": "toa_brightness_temperature",
                                    "long_name": f"Top of Atmosphere brightness temperature in band {meas_var}",
                                    "measurand": "brightness_temperature",
                                    "units": "K",
                                }
                            )

        if "tir" not in to.keys():
            raise ValueError(f"L89UnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}")
        elif "tir" in to.keys() and not to["tir"] == "brightness_temperature":
            raise ValueError(f"L89UnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}")

        return ds


if __name__ == "__main__":
    pass
