"""eoio.readers.sentinel2.masks - Sentinel-2 L1C classification mask reading

L1C products from processing baseline 04.00 onwards ship a per-granule
classification raster, ``QI_DATA/MSK_CLASSI_B00.jp2``: a 60 m, 3-band JP2 in
which each band is a binary (0/1) mask. The bands are, in order, opaque
clouds, cirrus, and snow/ice -- exposed here as the named variables in
:py:data:`MASK_OPTIONS`.

Earlier baselines carry ``QA60.jp2`` / ``MSK_CLOUDS_B00.gml`` instead; those
are not supported, and such products simply yield no mask variables (with a
warning) rather than raising.
"""

import warnings
from typing import Any, List, Optional

import numpy as np
import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.sentinel2.layout import S2Layout
from eoio.readers.subset.roi_subset import ResolvedROISubset

__all__ = ["MASK_OPTIONS", "MSK_CLASSI_BANDS", "add_masks", "read_masks"]

# Band index (1-based, as stored in the JP2) of each classification layer
# within MSK_CLASSI_B00.jp2. Order is fixed by the Sentinel-2 product format
# specification.
MSK_CLASSI_BANDS = {
    "opaque_clouds": 1,
    "cirrus": 2,
    "snow_ice": 3,
}

MASK_OPTIONS: List[str] = list(MSK_CLASSI_BANDS)

# CF-style attrs for each mask variable.
MASK_ATTRS = {
    "opaque_clouds": {
        "long_name": "Opaque cloud mask",
        "standard_name": "cloud_binary_mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_cloud cloud",
    },
    "cirrus": {
        "long_name": "Cirrus cloud mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_cirrus cirrus",
    },
    "snow_ice": {
        "long_name": "Snow and ice mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_snow_ice snow_ice",
    },
}


def add_masks(
    *,
    ds: xr.Dataset,
    masks: List[str],
    layout: S2Layout,
    subset: Optional[ResolvedROISubset],
    config: Any,
) -> xr.Dataset:
    """Read the requested L1C classification masks and add them to *ds*.

    Mirrors :py:func:`eoio.readers.sentinel3_olci.masks.add_masks`: validates
    the selection, then delegates the IO to :py:func:`read_masks`.

    :param ds: dataset to add mask variables to.
    :param masks: mask variable names to read (see :py:data:`MASK_OPTIONS`).
    :param layout: layout helper used to locate the mask file.
    :param subset: resolved ROI subset to clip the mask to, or ``None``.
    :param config: resolved reader configuration.
    :return: *ds* with the mask variables merged in.
    :raises ValueError: if an unknown mask name is requested.
    """
    unknown = [m for m in masks if m not in MSK_CLASSI_BANDS]
    if unknown:
        raise ValueError(f"Unknown Sentinel-2 mask(s): {unknown}. Available: {MASK_OPTIONS}")

    return read_masks(
        ds=ds,
        masks=masks,
        layout=layout,
        subset=subset,
        use_chunks=config.read_params.get("use_chunks", False),
        chunks=config.read_params.get("chunks", None),
    )


def read_masks(
    *,
    ds: xr.Dataset,
    masks: List[str],
    layout: S2Layout,
    subset: Optional[ResolvedROISubset] = None,
    chunks: Optional[dict] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read classification mask bands from ``MSK_CLASSI_B00.jp2`` into *ds*.

    The mask is a 60 m raster, which is coarser than the 10/20 m measurement
    grids, so its variables are placed on their own ``x_60m``/``y_60m``
    dimensions following this reader's existing per-resolution grid naming
    (see :py:func:`~eoio.readers.sentinel2.data_io.read_bands_into_dataset`)
    rather than being resampled onto a measurement grid.

    :param ds: dataset to add mask variables to.
    :param masks: mask variable names to read.
    :param layout: layout helper used to locate the mask file.
    :param subset: resolved ROI subset to clip the mask to, or ``None``.
    :param chunks: explicit dask chunk spec passed to ``rioxarray``.
    :param use_chunks: whether to open the raster lazily (chunked).
    :return: *ds* with the mask variables merged in.
    """
    path = layout.msk_classi_path()
    if path is None:
        # Pre-04.00 baselines (QA60/GML) and any product without QI_DATA. Warned
        # rather than raised: a caller requesting masks on an older product should
        # still get its measurement data, the same way an absent optional layer is
        # handled elsewhere in this reader.
        warnings.warn(
            "No MSK_CLASSI classification mask found in this product "
            "(expected for processing baselines before 04.00); no mask variables added."
        )
        return ds

    rxr = lazy_rioxarray()
    da = rxr.open_rasterio(path, chunks=chunks if use_chunks else None)

    for name in masks:
        band = MSK_CLASSI_BANDS[name]
        if "band" not in da.dims or da.sizes["band"] < band:
            warnings.warn(
                f"Classification mask {path!r} has no band {band} for {name!r}; skipping. "
                f"Expected a {len(MSK_CLASSI_BANDS)}-band raster."
            )
            continue

        # isel is 0-based; MSK_CLASSI_BANDS records the 1-based band number.
        layer = da.isel(band=band - 1, drop=True)

        if subset is not None:
            if subset.clip_box:
                x_min, y_min, x_max, y_max = subset.clip_box
                # This mask is 60 m -- much coarser than the 10/20 m measurement grids --
                # so a small ROI (e.g. a few-hundred-metre in-situ site box, or a sliver
                # left after clipping to a single MGRS tile near a tile boundary) can
                # easily span less than one pixel in x or y. rioxarray's default clip_box
                # raises OneDimensionalRaster rather than returning that degenerate result;
                # auto_expand retries with a half-pixel-per-side-larger box (up to 3 times)
                # to recover a real 2D window first, and allow_one_dimensional_raster
                # accepts a genuine 1-pixel-wide/tall result as a last resort rather than
                # dropping the whole ROI's cloud cover.
                layer = layer.rio.clip_box(
                    x_min, y_min, x_max, y_max, auto_expand=True, allow_one_dimensional_raster=True
                )
            if subset.geometries:
                layer = layer.rio.clip(subset.geometries, from_disk=True)

        # Stored as 0/1; keep it integral so the flag_values attrs stay meaningful
        # (the measurement bands are floats only because they carry NaN nodata).
        layer = layer.astype(np.uint8)

        rename_map = {d: f"{d}_60m" for d in ("x", "y") if d in layer.dims}
        if rename_map:
            layer = layer.rename(rename_map)
        for coord_name in layer.coords:
            if str(coord_name).startswith(("x", "y")):
                layer.coords[coord_name].attrs = {}

        layer.attrs = dict(MASK_ATTRS.get(name, {}))
        ds[name] = layer

    return ds


if __name__ == "__main__":
    pass
