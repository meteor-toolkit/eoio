"""I/O helpers for Sentinel-3 SLSTR reader.

This module provides standalone functions that operate on the existing
`SLSTRL1Reader` instance. They were factored out of the large reader
implementation to mirror the `sentinel3_olci` package layout.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any
import xarray as xr
from eoio.deps import lazy_rioxarray
from eoio.utils.crs_utils import get_nearest_lon_lat_coords
from eoio.readers.sentinel3_slstr.layout import S3SLSTRLayout
from eoio.readers.sentinel3_slstr.metadata.extractor import S3SLSTRMetadataExtractor
from eoio.utils.rasterio_utils import suggest_raster_chunks
from eoio.readers.sentinel3_slstr.utils import GRID_RES_MAP


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    meas: List[str],
    subset: Dict[str, Any],
    mtd: S3SLSTRMetadataExtractor,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read radiance bands into an xarray ``Dataset``.

    :param ds: Base dataset to which band variables will be added. The
        dataset is mutated and returned.
    :param layout: Layout helper used to locate band files on disk.
    :param meas: Sequence of band tokens to read (e.g. ``['S1_an']``).
    :param subset: Optional resolved ROI subset object; when present and
        containing a ``clip_box`` attribute the band arrays will be clipped
        to that box.
    :param mtd: Metadata extractor used to retrieve band-specific
        metadata.
    :param chunks: Optional chunking specification passed to the raster
        reader.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: The dataset with the requested bands attached.
    """
    rxr = lazy_rioxarray()

    band_paths = layout.meas_paths(meas)

    # set up chunking
    if use_chunks and chunks is None:
        first_path = next(iter(band_paths.values()))
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)
    for band, path in band_paths.items():
        if not path:
            continue

        grid = layout.get_grid(band)
        if grid is None:
            continue

        meas_ds = rxr.open_rasterio(path, chunks=chunks)
        ds_bnd = meas_ds[0]

        # Extract band dataarray if necessary
        if isinstance(ds_bnd, xr.Dataset):
            da = ds_bnd[band]

        da_attrs = da.attrs.copy()

        if clip_boxes and clip_boxes[grid] is not None:
            x_min, y_min, x_max, y_max = clip_boxes[grid]
            da.rio.write_crs(4326, inplace=True)
            da = da.rio.clip_box(x_min, y_min, x_max, y_max)

        # Drop "band"
        da = da.squeeze().drop_vars(["band"])

        # Apply scaling and valid range masking
        da = da.rename(
            {
                "x": f"x_{GRID_RES_MAP[grid]}m_{grid}",
                "y": f"y_{GRID_RES_MAP[grid]}m_{grid}",
            }
        )
        da = da.where(da != -32768)
        da.attrs.update(da_attrs)

        da = da * da.scale_factor + da.add_offset

        # Assign to dataset
        ds[band] = da

        # Update coords
        ds[band] = ds[band].assign_coords(
            {
                f"lon_{GRID_RES_MAP[grid]}m_{grid}": ds.coords[f"lon_{GRID_RES_MAP[grid]}m_{grid}"],
                f"lat_{GRID_RES_MAP[grid]}m_{grid}": ds.coords[f"lat_{GRID_RES_MAP[grid]}m_{grid}"],
            }
        )

        # Update attrs
        ds[band].attrs.update({k: v for (k, v) in da.attrs.items() if k not in ["_FillValue", "valid_min"]})

    return ds


def read_lat_lon_coordinates(
    *,
    ds: xr.Dataset,
    layout,
    grids,
    subset=None,
    config=None,
    chunks=None,
    use_chunks: bool = False,
):
    """Read SLSTR geodetic coordinate files for requested grids and attach
    latitude/longitude coordinates to ``ds``.

    :param ds: Base dataset to which coordinates will be added. The dataset is mutated and returned.
    :param layout: Layout object for handling file paths and grid information.
    :param grids: List of grids for which to read coordinates.
    :param subset: Optional subset configuration for ROI clipping.
    :param config: Configuration object for handling variable selections and other settings.
    :param chunks: Chunking configuration for reading files.
    :param use_chunks: Whether to use chunking when reading files.

    :returns: Tuple of (updated dataset, clip_boxes) where ``clip_boxes`` is a
        dict with optional keys 'lat_lon' and 'tie' suitable for downstream
        readers.
    """
    clip_boxes = {}
    rxr = lazy_rioxarray()

    # set up chunking
    if use_chunks and chunks is None:
        first_path = layout.geodetic_path(grids[0])
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    for grid in grids:
        gp = layout.geodetic_path(grid)

        # open via rioxarray
        if grid == "tx":
            ds_coords = rxr.open_rasterio(gp, chunks=chunks)
        else:
            ds_coords = rxr.open_rasterio(gp, chunks=chunks)[0]

        # Drop "band"
        ds_coords = ds_coords.squeeze().drop_vars(["band"])

        if grid == "tx":
            lats = ds_coords[f"latitude_{grid}"]
            lons = ds_coords[f"longitude_{grid}"]
        else:
            lats = ds_coords[f"latitude_{grid}"] * 10**-6
            lons = ds_coords[f"longitude_{grid}"] * 10**-6

        # Optionally apply ROI clipping
        if subset is not None and getattr(subset, "geometries", None) is not None:
            coords = get_nearest_lon_lat_coords(lons.data, lats.data, list(subset.geometries[0]["coordinates"][0]))

            xs, ys = [i[0] for i in coords], [i[1] for i in coords]
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
            clip_boxes[grid] = (x_min, y_min, x_max, y_max)

            # apply clip
            try:
                lons = lons.rio.write_crs(4326, inplace=False).rio.clip_box(x_min, y_min, x_max, y_max)
                lats = lats.rio.write_crs(4326, inplace=False).rio.clip_box(x_min, y_min, x_max, y_max)
            except Exception:
                pass

        # rename coordinate dims to include resolution and grid

        res_m = layout.get_grid_res(grid)
        xname = f"x_{res_m}m_{grid}" if res_m is not None else f"x_{grid}"
        yname = f"y_{res_m}m_{grid}" if res_m is not None else f"y_{grid}"

        lats = lats.rename({"x": xname, "y": yname})
        lons = lons.rename({"x": xname, "y": yname})

        # assign to dataset
        ds = ds.assign_coords({f"lon_{res_m}m_{grid}": lons, f"lat_{res_m}m_{grid}": lats})

        # set attrs
        ds[f"lat_{res_m}m_{grid}"].attrs.update(
            {
                "units": "degrees_north",
                "standard_name": f"latitude {grid}",
                "long_name": f"latitude {grid}",
            }
        )
        ds[f"lon_{res_m}m_{grid}"].attrs.update(
            {
                "units": "degrees_east",
                "standard_name": f"longitude {grid}",
                "long_name": f"longitude {grid}",
            }
        )

    return ds, clip_boxes if clip_boxes else None
