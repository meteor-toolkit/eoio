"""eoio.readers.sentinel2.aux_vars.aux_data - auxiliary data reading functions for Sentinel-2 reader"""

from __future__ import annotations
from eoio.readers.base import ReaderConfig
import xarray as xr
from eoio.readers.sentinel2.layout import S2Layout
from eoio.readers.sentinel2.metadata.extractor import S2MSIMetadataExtractor
from eoio.readers.sentinel2.aux_vars.angles import add_angles
from eoio.readers.sentinel2.aux_vars.meteo import add_meteo
from eoio.readers.sentinel2.metadata.var_names import (
    ANGLE_VARS,
    AUX_CAMS_VARS,
    AUX_ECMWF_VARS_NEW,
    AUX_ECMWF_VARS_OLD,
)
from eoio.utils.aux_read import warn_on_aux_failure


def get_available_aux(layout: S2Layout) -> list[str]:
    """
    Returns a list of available aux data

    :param layout: S2Layout object
    :returns: List of available auxiliary data.
    """

    pv = layout.proc_version

    aux_vars = AUX_ECMWF_VARS_NEW + AUX_CAMS_VARS + ANGLE_VARS if pv[0] >= 5 else AUX_ECMWF_VARS_OLD + ANGLE_VARS

    return aux_vars


def add_aux(
    *,
    ds: xr.Dataset,
    layout: S2Layout,
    config: ReaderConfig,
    mtd: S2MSIMetadataExtractor,
) -> xr.Dataset:
    """
    Add auxiliary data (e.g. AUX_ECMWFT, AUX_CAMSFO) to the dataset.

    This is intentionally conservative:
    - reads full aux fields (no spatial subsetting yet)
    - merges into dataset
    - stores aux metadata under ds.attrs["product_metadata"]

    Angle and meteo aux data are read independently: if one fails (e.g. a missing or
    corrupt CAMS/ECMWF GRIB file), a warning is raised and that aux data is left out
    of the returned dataset, but the rest of the read (bands, the other aux category)
    is unaffected.
    """

    aux_names = config.vars_sel["aux"]

    # divide vars into types for processing
    angle_names = [x for x in ANGLE_VARS if x in aux_names]
    remaining_vars = [x for x in aux_names if x not in angle_names]

    if angle_names:
        with warn_on_aux_failure("Sentinel-2 angles"):
            ds = add_angles(
                ds=ds,
                angle_names=angle_names,
                mtd=mtd,
                ave_det=config.read_params["ave_va_det"],
            )

    meteo_names = [x for x in AUX_ECMWF_VARS_NEW + AUX_CAMS_VARS + AUX_ECMWF_VARS_OLD if x in remaining_vars]
    if meteo_names:
        with warn_on_aux_failure("Sentinel-2 meteo"):
            ds = add_meteo(ds=ds, var_names=meteo_names, layout=layout, config=config)

    return ds


if __name__ == "__main__":
    pass
