"""eoio.readers.modis.aux_vars.aux_data - auxiliary data reading functions for MODIS reader"""

from __future__ import annotations
from typing import Dict, Optional
import warnings
from eoio.readers.base import ReaderConfig
import xarray as xr
from eoio.deps import lazy_rioxarray
from eoio.readers.modis.aux_vars.angles import add_angles
from eoio.readers.modis.aux_vars.atmos import add_atmos
from eoio.readers.modis.aux_vars.geolocation import add_geolocation
from eoio.readers.modis.layout import MODISLayout
from eoio.readers.modis.metadata.extractor import MODISMetadataExtractor
from eoio.readers.modis.metadata.var_names import (
    ANGLE_VARS,
    ATMOS_VARS,
)
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.utils.rasterio_utils import suggest_raster_chunks


def get_available_aux(layout: MODISLayout) -> list[str]:
    """
    Returns a list of available aux data

    :param layout: MODISLayout object
    :return: list of available aux data
    """

    geolocation_path = layout.geolocation_path()
    if not geolocation_path.exists():
        aux_vars = []
    else:
        aux_vars = ANGLE_VARS + ATMOS_VARS

    return aux_vars


def add_aux(
    *,
    ds: xr.Dataset,
    geolocation_ds: xr.Dataset,
    layout: MODISLayout,
    subset: ResolvedROISubset,
    config: ReaderConfig,
    mtd: MODISMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """
    Add auxiliary data (e.g. AUX_ECMWFT, AUX_CAMSFO) to the dataset.

    This is intentionally conservative:
    - reads full aux fields (no spatial subsetting yet)
    - merges into dataset
    - stores aux metadata under ds.attrs["product_metadata"]
    """

    # ------------------------------------------------------------------
    # Set up chunking
    # ------------------------------------------------------------------
    if use_chunks and chunks is None:
        chunks = suggest_raster_chunks(str(layout.path), target_mb=32.0)

    aux_names = config.vars_sel["aux"]

    # divide vars into types for processing
    angle_names = [x for x in ANGLE_VARS if x in aux_names]
    atmos_names = [x for x in ATMOS_VARS if x in aux_names]
    remaining_vars = [x for x in aux_names if x not in angle_names and x not in atmos_names]

    ang_ds = xr.open_dataset(
        layout.geolocation_path(), group="/HDFEOS/SWATHS/MODIS_Swath_Type_GEO/Data Fields", chunks=chunks
    )
    if layout.processing_level == "L2":
        rxr = lazy_rioxarray()
        atmos_ds = rxr.open_rasterio(layout.path, chunks=chunks)[0]
    else:
        atmos_ds = None
        atmos_names = None
        warnings.warn(
            f"Atmospheric variables {atmos_names} requested but not available for processing level {layout.processing_level}"
        )

    if angle_names:
        ds = add_angles(
            ds=ds,
            aux_ds=ang_ds,
            angle_names=angle_names,
            mtd=mtd,
        )

    if atmos_names:
        ds = add_atmos(
            ds=ds,
            aux_ds=atmos_ds,
            atmos_names=atmos_names,
            mtd=mtd,
        )

    if angle_names or atmos_names:
        ds = add_geolocation(ds=ds, geolocation_ds=geolocation_ds, resolution=["1000m"])
        if subset is not None:
            if subset.clip_box:
                x_min, y_min, x_max, y_max = subset.clip_box
                lon = ds["longitude_1000m"]
                lat = ds["latitude_1000m"]
                mask = (lat >= y_min) & (lat <= y_max) & (lon >= x_min) & (lon <= x_max)
                ds = xr.Dataset(
                    {
                        name: da.where(mask, drop=True) if set(mask.dims).issubset(da.dims) else da
                        for name, da in ds.data_vars.items()
                    }
                )

    return ds


if __name__ == "__main__":
    pass
