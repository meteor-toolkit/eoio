"""Mask helpers for Sentinel-3 SLSTR reader."""

import xarray as xr
from eoio.readers.sentinel3_slstr.layout import S3SLSTRLayout
from eoio.readers.sentinel3_slstr.utils import GRID_RES_MAP
from eoio.readers.sentinel3_slstr.data_io import lazy_rioxarray
from typing import Optional, Dict, Any, Tuple

from eoio.utils.rasterio_utils import suggest_raster_chunks, _first_path


def add_masks(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    grids: list[str],
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]],
    config: Any,
    use_chunks: bool,
    chunks: Optional[Dict[str, int]],
):
    """Read quality masks and add them to an xarray Dataset.

    :param ds: Dataset to which quality mask variables will be added.
    :param layout: Layout helper for locating the relevant mask file.
    :param grids: List of grid identifiers to read masks for (e.g. ["an", "ao"]).
    :param clip_boxes: Bounding boxes for subsetting the mask data, or ``None``.
    :param config: Resolved reader configuration containing parameters used by the reader.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :param chunks: Optional chunking specification passed to the raster reader.
    :returns: Dataset with mask variables merged in.
    """

    rxr = lazy_rioxarray()

    meas = config.vars_sel.get("meas", None)
    mask_names = config.vars_sel.get("mask", None)
    aux_names = config.vars_sel.get("aux", None)
    if mask_names is None:
        raise ValueError("No masks have been selected.")

    # Map requested masks to variable names
    mask_vars = []
    for flag in mask_names:
        for grid in grids:
            if grid != "tx":
                if flag in ["cloud", "confidence", "bayes", "pointing"]:
                    mask_vars.append(f"{flag}_{grid}")
                    if "orphan" in aux_names:
                        mask_vars.append(f"{flag}_orphan_{grid}")
        if flag == "exception":
            for m in meas:
                var = m.split("_")[0]
                grid = m.split("_")[-1]
                mask_vars.append(f"{var}_exception_{grid}")
                if "orphan" in aux_names:
                    mask_vars.append(f"{var}_exception_orphan_{grid}")

    # Open each mask variable and merge into dataset
    for flag in mask_vars:
        flag_grid = flag.split("_")[-1]

        if flag[:-3] in ["cloud", "confidence", "bayes", "pointing"]:
            if use_chunks and chunks is None:
                first_path = _first_path(layout.flags_path(flag_grid))
                if first_path:
                    chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

            flags_nc = rxr.open_rasterio(layout.flags_path(flag_grid), chunks=chunks)
            flags_ds = flags_nc[0][flag]
            flags_ds = flags_ds.squeeze().drop_vars(["band"])

        elif flag[:-3] in [
            "cloud_orphan",
            "confidence_orphan",
            "bayes_orphan",
            "pointing_orphan",
        ]:
            if use_chunks and chunks is None:
                first_path = _first_path(layout.flags_path(flag_grid))
                if first_path:
                    chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

            flags_nc = rxr.open_rasterio(layout.flags_path(flag_grid), chunks=chunks)
            flags_ds = flags_nc[1][flag]
            flags_ds = flags_ds.squeeze().drop_vars(["band"])

        elif flag[-12:-3] == "exception":
            mask_var = layout.mask_to_var(flag)
            if mask_var is None:
                continue
            flags_nc = rxr.open_rasterio(layout.meas_paths()[mask_var])
            flags_ds = flags_nc[0][f"{flag[:2]}_exception_{flag_grid}"]
            flags_ds = flags_ds.squeeze().drop_vars(["band"])

        elif flag[-19:-3] == "exception_orphan":
            mask_var = layout.mask_to_var(flag)
            if mask_var is None:
                continue
            flags_nc = rxr.open_rasterio(layout.meas_paths()[mask_var])
            flags_ds = flags_nc[1][f"{flag[:2]}_exception_orphan_{flag_grid}"]
            flags_ds = flags_ds.squeeze().drop_vars(["band"])

        else:
            raise ValueError(f"{flag} flag type not implemented")

        try:
            flag_meanings = flags_ds.flag_meanings.split()[::-1]
            flag_masks = [str(int(i)) for i in list(flags_ds.flag_masks)[::-1]]
        except AttributeError:
            flag_meanings = [None]  # type: ignore[list-item]
            flag_masks = [None]  # type: ignore[list-item]
            raise Warning(
                f"Flag variable {flag} is missing flag_meanings or flag_masks attributes; these will be set to None."
            )

        flag_meanings, flag_masks = list(zip(*[i for i in zip(flag_meanings, flag_masks)]))  # type: ignore[assignment]

        if "orphan" not in flag.split("_"):
            if clip_boxes:
                flags_ds.rio.write_crs(4326, inplace=True)
                x_min, y_min, x_max, y_max = clip_boxes[flag_grid]
                flags_ds = flags_ds.rio.clip_box(x_min, y_min, x_max, y_max)

        if "orphan" in flag.split("_"):
            flags_x = f"x_orphan_{GRID_RES_MAP[flag_grid]}m_{flag_grid}"
            flags_y = f"y_orphan_{GRID_RES_MAP[flag_grid]}m_{flag_grid}"
        else:
            flags_x = f"x_{GRID_RES_MAP[flag_grid]}m_{flag_grid}"
            flags_y = f"y_{GRID_RES_MAP[flag_grid]}m_{flag_grid}"

        flags_ds = flags_ds.rename({"x": flags_x, "y": flags_y})

        ds[flag] = ((flags_y, flags_x), flags_ds.data)
        ds[flag].attrs = {
            "flag_meanings": " ".join(flag_meanings),
            "flag_masks": ",".join(flag_masks),
        }

    return ds
