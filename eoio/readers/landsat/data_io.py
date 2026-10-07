"""eoio.readers.landsat.data_io - data reading functionality for Landsat meas vars"""

from __future__ import annotations
from typing import Dict, List, Optional
import warnings

import numpy as np
import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.landsat.layout import LandsatLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.readers.landsat.metadata import LSMetadataExtractor
from eoio.utils.rasterio_utils import suggest_raster_chunks


def _nearest_resample_2d(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbour resample a 2D array onto *target_shape*, independently per axis.

    Used to align the angle files' own 30m pixel grid onto a band read at a different
    resolution (namely Landsat's panchromatic B8, at 15m -- exactly double the angle
    grid's linear resolution) before the per-pixel solar zenith angle correction below.
    Not assumed to be an exact integer ratio: an ROI subset is clipped independently per
    variable (see read_angles_into_dataset's/this function's own subset.clip_box/
    geometries handling), so two rasters at different native resolutions covering the
    same nominal footprint can come out a pixel or two off from a clean integer multiple
    -- this still produces a sensible (if very slightly re-aligned) result in that case,
    rather than only ever handling the exact-multiple case.

    :param arr: source 2D array (angle grid).
    :param target_shape: ``(rows, cols)`` to resample onto (the band's own shape).
    :return: *arr* resampled onto *target_shape* via nearest-neighbour indexing.
    """
    src_rows, src_cols = arr.shape
    tgt_rows, tgt_cols = target_shape
    row_idx = np.minimum((np.arange(tgt_rows) * src_rows / tgt_rows).astype(int), src_rows - 1)
    col_idx = np.minimum((np.arange(tgt_cols) * src_cols / tgt_cols).astype(int), src_cols - 1)
    return arr[np.ix_(row_idx, col_idx)]


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    layout: LandsatLayout,
    meas_vars: List[str],
    subset: None | ResolvedROISubset,
    mtd: LSMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
    mask_zero_as_nodata: bool = True,
) -> xr.Dataset:
    """
    Reads Landsat bands into the dataset.

    Minimal example:
    - Opens each TIF file via rioxarray.
    - Applies clip_box/geometries if present.
    - Stores each band as ds[varname].

    :param ds: xarray.Dataset to populate.
    :param layout: LandsatLayout object defining the product structure.
    :param meas_vars: List of measurement variables to read, e.g. ["B2", "B3"].
    :param meas_var_res: Dictionary mapping measurement variables to their resolutions.
    :param subset: Subset information for ROI and geometries.
    :param chunks:
        Optional mapping of dimension names to chunk sizes passed directly to
        ``rioxarray.open_rasterio``. If provided, this overrides any
        automatically derived chunk sizes.
    :param use_chunks:
        If ``True`` and ``chunks`` is not provided, enable Dask-backed lazy
        reading using block-aligned chunk sizes inferred from the raster
        metadata.
    :param mask_zero_as_nodata:
        If ``True``, values equal to zero are masked as missing data after
        scaling.
    :return: Populated xarray.Dataset.
    """
    # lazy import of rioxarray
    rxr = lazy_rioxarray()

    # get band paths from layout
    band_paths = layout.tif_band_files(meas_vars)

    # set up chunking if requested
    if use_chunks and chunks is None:
        first_path = next(iter(band_paths.values()))
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    # loop through bands, open variable and attach to ds
    for band, path in band_paths.items():
        # unpack required variable metadata
        band_mtd = mtd.get_variable_product_metadata(band)

        # read band data
        da = rxr.open_rasterio(path, chunks=chunks)

        if "band" in da.dims:
            da = da.isel(band=0, drop=True)

        da = da.astype("float32")

        # set no data as nan
        try:
            encoded_nodata = da.rio.encoded_nodata
        except Exception:
            encoded_nodata = None

        if encoded_nodata is not None:
            da = da.where(da != encoded_nodata)

        if mask_zero_as_nodata:
            da = da.where(da != 0)

        # apply subset clipping if provided
        if subset is not None:
            if subset.clip_box:
                x_min, y_min, x_max, y_max = subset.clip_box
                da = da.rio.clip_box(x_min, y_min, x_max, y_max)

            if subset.geometries:
                da = da.rio.clip(subset.geometries, from_disk=True)

        # Apply radiometric scaling to reflectance for visible bands, and radiance for thermal bands
        if band in ["B10", "B11"]:
            # Radiance (float) = (RADIANCE_MULT_BAND_x * DN) + RADIANCE_ADD_BAND_x
            da *= band_mtd.get("radiance_mult", 1.0)
            da += band_mtd.get("radiance_add", 0.0)
        else:
            # Reflectance (float) = ((REFLECTANCE_MULT_BAND_x * DN) + REFLECTANCE_ADD_BAND_x)/sin(SUN_ELEVATION)
            da *= band_mtd.get("reflectance_mult", 1.0)
            da += band_mtd.get("reflectance_add", 0.0)
            # try per-pixel solar angles if available & if not in ds, use sun_elevation from metadata
            if "solar_zenith_angle" in ds:
                sza_values = ds["solar_zenith_angle"].values
                if sza_values.shape != da.shape:
                    # The angle files are always read at their own native (30m) grid --
                    # see LSMetadataExtractor.get_angle_metadata()'s hardcoded
                    # geometry_id -- which only matches most Landsat bands' own
                    # resolution. The panchromatic B8 (15m) is the one exception, so its
                    # per-pixel correction needs the angle grid resampled onto its own
                    # (finer) grid first, rather than dividing two differently-shaped
                    # arrays outright.
                    sza_values = _nearest_resample_2d(sza_values, da.shape)
                try:
                    da /= np.sin(np.deg2rad(90.0 - sza_values))  # per-pixel SZA
                except Exception:
                    warnings.warn(
                        "Per-pixel solar zenith angle correction failed; using metadata sun elevation instead."
                    )
                    da /= np.sin(
                        np.deg2rad(mtd.get_product_metadata().get("sun_elevation"))  # type: ignore[arg-type]
                    )  # Perhaps a warning should be raised here as SZA used is less accurate as not per pixel ?
            else:
                warnings.warn(
                    "Per-pixel solar zenith angle correction not available; using metadata sun elevation instead."
                )
                da /= np.sin(np.deg2rad(mtd.get_product_metadata().get("sun_elevation")))  # type: ignore[arg-type]

            # Landsat Collection 2 reflectance products use a negative scaled value (e.g.
            # -0.1) as a fill sentinel for pixels outside the actual acquired footprint,
            # rather than the encoded_nodata/zero-DN handling above already catching it --
            # physically, reflectance can't be negative, so any negative value left after
            # scaling is fill data, not a real (if implausible) measurement.
            da = da.where(da >= 0)

        # name dimensions according to geometry
        geom = band_mtd.get("geometry_id")
        if geom:
            rename_map = {}
            if "x" in da.dims:
                rename_map["x"] = f"x_{geom}"
            if "y" in da.dims:
                rename_map["y"] = f"y_{geom}"
            if rename_map:
                da = da.rename(rename_map)

        # Strip attributes attached by rioxarray clipping
        for coord_name in da.coords:
            if coord_name == "x" or coord_name == "y" or coord_name.startswith("x_") or coord_name.startswith("y_"):
                da.coords[coord_name].attrs = {}

        ds[band] = da

    return ds


if __name__ == "__main__":
    pass
