"""eoio.processors.units.drivers.s2msi1c - unit conversion for Sentinel-2 MSI L1c products"""

from __future__ import annotations
from typing import Any, Mapping, Sequence, Set, Optional, Tuple, Union
import numpy as np
import xarray as xr
from eoio.processors.units.drivers.base import BaseUnitConversionDriver
from eoio.processors.units.drivers.registry import register_unit_driver
from processor_tools.utils.dict_tools import get_value
from scipy.interpolate import griddata

import math
import re
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

meas_var_res = {  # dictionary of measurement variables and their spatial resolutions in metres
    "AOT_10m": 10,
    "B02_10m": 10,
    "B03_10m": 10,
    "B04_10m": 10,
    "B08_10m": 10,
    "TCI_10m": 10,
    "WVP_10m": 10,
    "AOT_20m": 20,
    "B01_20m": 20,
    "B02_20m": 20,
    "B03_20m": 20,
    "B04_20m": 20,
    "B05_20m": 20,
    "B06_20m": 20,
    "B07_20m": 20,
    "B8A_20m": 20,
    "B11_20m": 20,
    "B12_20m": 20,
    "SCL_20m": 20,
    "TCI_20m": 20,
    "VIS_20m": 20,
    "WVP_20m": 20,
    "AOT_60m": 60,
    "B01_60m": 60,
    "B02_60m": 60,
    "B03_60m": 60,
    "B04_60m": 60,
    "B05_60m": 60,
    "B06_60m": 60,
    "B07_60m": 60,
    "B8A_60m": 60,
    "B09_60m": 60,
    "B11_60m": 60,
    "B12_60m": 60,
    "SCL_60m": 60,
    "TCI_60m": 60,
    "WVP_60m": 60,
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
    "TCI": 10,
}


@register_unit_driver("s2msi1c")
class S2MSI1CUnitDriver(BaseUnitConversionDriver):
    """
    Sentinel-2 MSI L1c unit conversion driver.

    Intended scope:
    - reflectance -> radiance, using metadata available in the product

    This driver assumes eoio readers provide enough product identification to
    allow cheap matching (e.g. ds.attrs['eoio:product'] in {"sentinel2_l1c", "sentinel2_l2a"}).

    Notes
    -----
    Sentinel-2 unit conversion depends on:
    - per-band solar irradiance (E_sun) and band definitions
    - solar zenith angle (or sun elevation) at acquisition time / tile
    - any scaling/quantification already applied by the reader

    Where those values live should be standardised in dataset and variable attrs.
    """

    name: str = "s2msi1c"

    def matches(self, ds: xr.Dataset, context: Mapping[str, Any]) -> bool:
        """
        Determine whether this driver can handle the given dataset.

        Matching should be cheap and unambiguous; avoid heavy inspection.

        :param ds:
            Input dataset.
        :param context:
            Processing context provided by eoio.
        :return:
            True if this looks like a Sentinel-2 MSI dataset, otherwise False.
        """

        return ds.attrs["collection_name"].lower() == self.name

    def supported(self) -> Set[Tuple[str, str]]:
        """
        Return supported conversion pairs.

        :return:
            Set of supported (from_unit, to_unit) pairs.
        """
        return {
            ("reflectance", "radiance"),
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
        Converts the data using conversion coefficients found in metadata (E_sun, solar zenith, etc.) and updates units attributes accordingly.

        Using algorithm from senbox.org/s2tbx
        <https://htmlpreview.github.io/?https://github.com/senbox-org/s2tbx/blob/bca0c210fc15f0f95128dceaa512947388bc4f97/s2tbx-reflectance-to-radiance-ui/src/main/resources/org/esa/s2tbx/reflectance2radiance/docs/ReflectanceToRadianceAlgorithmSpecification.html>

        radiance = pixelValue * cosine(radians(incidenceAngle)) * solarIrradiance * U / pi

        :param ds:
            Input dataset.
        :param to:
            Target unit name (e.g. "radiance").
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
                            f"S2MSI1CUnitDriver: Selected Var not in the dataset: {var}. Vars on the dataset include: {str(vars_list)}"
                        )
        else:
            var_names = [str(var) for var in ds.data_vars if str(var).startswith("B")]

        if to == "radiance":
            for resolution in [10, 20, 60]:
                if str(resolution) in str(list(ds.dims)):
                    try:
                        output_data = self.interp_var_s2(ds, resolution, var="solar_zenith_angle")
                    except KeyError as e:
                        print(repr(e), ". eoio will return the data as reflectance.")
                        continue

                    for meas_var in ds:
                        if not re.findall(r"\AB", meas_var):  # type: ignore[call-overload]
                            pass
                        elif not any(["reflectance" in i for i in ds[meas_var].attrs.values() if isinstance(i, str)]):
                            pass
                        else:
                            if resolution == meas_var_res[str(meas_var)]:
                                # print(f"Converting {meas_var} to radiance...")
                                logger.debug(f"Converting {meas_var} to radiance")  # for developer
                                logger.info(f"Converting {meas_var} to radiance")  # for user

                                var_mtd = ds[meas_var].attrs

                                meas_var_data_array = deepcopy(ds[meas_var])
                                ds[meas_var] = self.rfl_to_rad(
                                    meas_var_data_array,
                                    np.radians(output_data),
                                    get_value(
                                        meas_var_data_array.attrs["product_metadata"],
                                        "solar_irradiance",
                                    ),
                                    get_value(
                                        ds.attrs["product_metadata"],
                                        "reflectance_conversion_u",
                                    ),
                                )

                                # reassign metadata
                                ds[meas_var].attrs = var_mtd

                                # Change attributes from to 'radiance'
                                ds[meas_var].attrs.update(
                                    {
                                        "standard_name": "toa_radiance",
                                        "long_name": f"TOA spectral radiance in band {meas_var}",
                                        "measurand": "radiance",
                                        "units": "W/( m² * sr * μm)",
                                    }
                                )
        else:
            raise ValueError(f"S2MSI1CUnitDriver: unsupported conversion to '{to}'. Supported: {self.supported()}")

        return ds

    @staticmethod
    def rfl_to_rad(
        rfl: Union[float, np.ndarray, xr.DataArray],
        sza: Union[float, np.ndarray],
        e_sol: Union[float, np.ndarray],
        u: Union[float, int],
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Convert reflectance to radiance

        :param rfl: reflectance(s)
        :param sza: solar zenith angle (radians)
        :param e_sol: solar irradiance
        :param u: Earth-Sun distance correction factor
        :return: radiance (W/(m²*sr*μm))
        """
        # Calculate radiance
        return (rfl * np.cos(sza) * e_sol * u) / math.pi

    @staticmethod
    def interp_var_s2(ds, target_resolution, var="solar_zenith_angle"):
        """
        Interpolate var data from source to target resolution

        :param ds: S2 ds read in by eoio
        :param target_resolution: resolution to interpolate to
        :return: interpolated data on target resolution grid
        """
        # get var data from dataset

        if var in ds:
            data = ds[var].data
        else:
            raise KeyError(
                "Solar zenith angle needed for conversion not found. Update eoio.read() to select the relevant aux_data: 'vars_sel': {'aux': ['solar_zenith_angle']}"
            )
        if str(target_resolution) not in str(list(ds.dims)):
            raise KeyError(
                f"Resolution {target_resolution} not included in dataset. Existing dimensions are {list(ds.dims)}"
            )

        # get source x and y
        x_source = ds[ds[var].dims[[idx for idx, s in enumerate(ds[var].dims) if "x" in s][0]]].data
        y_source = ds[ds[var].dims[[idx for idx, s in enumerate(ds[var].dims) if "y" in s][0]]].data

        # get target x and y
        x_output = ds[f"x_{target_resolution:.0f}m"].data
        y_output = ds[f"y_{target_resolution:.0f}m"].data

        # create source and target grids in format needed for griddata
        x_source_grid, y_source_grid = np.meshgrid(x_source, y_source)
        x_output_grid, y_output_grid = np.meshgrid(x_output, y_output)

        # interpolate values using source and desired x y output
        output_data = griddata(
            (x_source_grid.flatten(), y_source_grid.flatten()),
            data.flatten(),
            (x_output_grid, y_output_grid),
        )

        return output_data


if __name__ == "__main__":
    pass
