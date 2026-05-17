"""eoio.readers.sentinel2.aux_vars.meteo - meteo data reading functions"""

from __future__ import annotations
from typing import Any
from eoio.deps import lazy_cfgrib
from eoio.readers.sentinel2.layout import S2Layout
from eoio.readers.sentinel2.metadata.var_names import (
    AUX_ECMWF_VARS_NEW,
    AUX_ECMWF_VARS_OLD,
    AUX_CAMS_VARS,
)
import xarray as xr
import os
from contextlib import contextmanager


def add_meteo(*, ds: xr.Dataset, var_names: list[str], layout: S2Layout, config: Any) -> xr.Dataset:
    """
    Add requested meteorological / atmospheric auxiliary variables from Sentinel-2 AUX_DATA
    GRIB products (ECMWF/CAMS) into ds (no regridding performed).

    :param ds:
        Input xarray Dataset to which auxiliary variables will be added.
        This dataset is not modified in-place; a new dataset is returned.
    :param var_names:
        List of auxiliary variable names to extract and attach. These must
        correspond to variables available in the Sentinel-2 ECMWF or CAMS
        auxiliary GRIB products (e.g. ``tcwv``, ``tco3``, ``msl``, ``u10``,
        ``v10``, ``r``).
    :param layout:
        ``S2Layout`` instance describing the Sentinel-2 SAFE product layout.
        Used to locate the granule directory and associated AUX_DATA products.
    :param config:
        Reader or processor configuration object. Currently unused, but
        included for API consistency and future extensibility.
    :returns:
        A new xarray Dataset containing the original data and the requested
        auxiliary variables.
    :raises KeyError:
        If one or more requested variables are not present in the available
        auxiliary GRIB products.
    """

    lazy_cfgrib()

    granule = layout.granule_dir()

    # Decide which requested vars require which aux source
    pv = layout.proc_version
    ecmwf_vars = AUX_ECMWF_VARS_NEW if pv[0] >= 5 else AUX_ECMWF_VARS_OLD

    need_ecmwf = any(v in ecmwf_vars for v in var_names)
    need_cams = any(v in AUX_CAMS_VARS for v in var_names)

    if not (need_ecmwf or need_cams):
        return ds

    parts: list[xr.Dataset] = []

    if need_ecmwf or need_cams:
        # ecCodes prints error messages to stderr for some GRIB messages
        # (e.g. "unable to represent the step in h"). Suppress low-level
        # C stderr while opening with cfgrib to avoid noisy output.
        @contextmanager
        def _suppress_stderr_fd():
            devnull = os.open(os.devnull, os.O_RDWR)
            try:
                old_stderr = os.dup(2)
                os.dup2(devnull, 2)
                os.close(devnull)
                yield
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)

    if need_ecmwf:
        ecmwf_path = layout.aux_path("AUX_ECMWFT", granule=granule)
        with _suppress_stderr_fd():
            parts.append(xr.open_dataset(ecmwf_path, engine="cfgrib"))

    if need_cams:
        cams_path = layout.aux_path("AUX_CAMSFO", granule=granule)
        with _suppress_stderr_fd():
            parts.append(xr.open_dataset(cams_path, engine="cfgrib"))

    aux_ds = xr.merge(parts, compat="no_conflicts", combine_attrs="drop_conflicts").rename(
        {"latitude": "latitude_aux", "longitude": "longitude_aux"}
    )

    # Make missing vars an explicit error (or skip if you prefer)
    missing = [v for v in var_names if v not in aux_ds]
    if missing:
        raise KeyError(f"Requested aux vars not found in AUX GRIB(s): {missing}. Available: {list(aux_ds.data_vars)}")

    return ds.assign({v: aux_ds[v] for v in var_names})


if __name__ == "__main__":
    pass
