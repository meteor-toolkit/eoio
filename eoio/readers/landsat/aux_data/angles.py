"""eoio.readers.landsat.data_io - data reading functionality for Landsat meas vars"""

from __future__ import annotations
from typing import Dict, List, Optional

import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.landsat.layout import LandsatLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.readers.landsat.metadata import LSMetadataExtractor
from eoio.utils.rasterio_utils import suggest_raster_chunks


def read_angles_into_dataset(
    *,
    ds: xr.Dataset,
    layout: LandsatLayout,
    angle_vars: List[str],
    subset: None | ResolvedROISubset,
    mtd: LSMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:

    # lazy import of rioxarray
    rxr = lazy_rioxarray()

    # get angle file paths from layout
    angle_paths = layout.get_angle_files()

    # keep only those angles requested
    angle_vars = list(set(angle_vars))
    angle_paths = {k: v for k, v in angle_paths.items() if k in angle_vars}
    angle_mtd_all = mtd.get_angle_metadata()
    if not angle_paths:
        return ds

    if use_chunks and chunks is None:
        first_path = next(iter(angle_paths.values()))
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    for angle, path in angle_paths.items():
        angle_mtd = angle_mtd_all.get(angle, {})

        da = rxr.open_rasterio(path, chunks=chunks)

        if "band" in da.dims:
            da = da.isel(band=0, drop=True)

        da = da.astype("float32")
        da /= 100.0  # scale factor for angles to degrees

        try:
            encoded_nodata = da.rio.encoded_nodata
        except Exception:
            encoded_nodata = None

        if encoded_nodata is not None:
            da = da.where(da != encoded_nodata)

        if subset is not None:
            if subset.clip_box:
                x_min, y_min, x_max, y_max = subset.clip_box
                da = da.rio.clip_box(x_min, y_min, x_max, y_max)

            if subset.geometries:
                da = da.rio.clip(subset.geometries, from_disk=True)

        geom = angle_mtd.get("geometry_id")
        if geom:
            rename_map = {}
            if "x" in da.dims:
                rename_map["x"] = f"x_{geom}"
            if "y" in da.dims:
                rename_map["y"] = f"y_{geom}"
            if rename_map:
                da = da.rename(rename_map)

        # rioxarray may attach CF-ish attrs to projected x/y coords on some paths
        # (notably clipping). Strip them here; eoio metadata is handled later.
        for coord_name in da.coords:
            if coord_name == "x" or coord_name == "y" or coord_name.startswith("x_") or coord_name.startswith("y_"):
                da.coords[coord_name].attrs = {}

        ds[angle] = da

    return ds


if __name__ == "__main__":
    pass
