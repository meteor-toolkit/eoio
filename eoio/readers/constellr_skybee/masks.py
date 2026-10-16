"""eoio.readers.constellr_skybee.masks - Skybee QA mask reading

Every VNIR band ships a ``vnirNN_qa.tiff`` on the band's own grid, with three binary (0/1)
layers: ``blackfill_mask`` (no data), ``saturated_mask`` and ``untested_mask``. To give one
mask per flag, as the other readers do, they are read from the VNIR04 (red) QA raster -- the
10 m band the product is georeferenced against -- so they sit on the same ``x_10m``/``y_10m``
grid as B2, B3, B4 and B8. Pixels outside the ROI polygon are NaN, so the variables are
float32 (as for the PlanetScope masks).
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.constellr_skybee.layout import ConstellrSkybeeLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset

__all__ = ["MASK_OPTIONS", "QA_BANDS", "add_masks", "read_masks"]

# Band index (1-based) of each flag in the QA tiffs, as in the files' band descriptions.
QA_BANDS = {"blackfill_mask": 1, "saturated_mask": 2, "untested_mask": 3}

MASK_OPTIONS: list[str] = list(QA_BANDS)

# QA raster (and so grid) the masks are read from
_REF_BAND = "B4"
_RES = "10m"

MASK_ATTRS = {
    "blackfill_mask": {
        "long_name": "Black fill (no data) mask",
        "flag_values": [0, 1],
        "flag_meanings": "valid_data no_data",
    },
    "saturated_mask": {
        "long_name": "Saturated pixel mask",
        "flag_values": [0, 1],
        "flag_meanings": "valid_data saturated",
    },
    "untested_mask": {
        "long_name": "Untested pixel mask",
        "flag_values": [0, 1],
        "flag_meanings": "valid_data untested",
    },
}


def add_masks(
    *,
    ds: xr.Dataset,
    masks: list[str],
    layout: ConstellrSkybeeLayout,
    subset: ResolvedROISubset | None,
    config: Any,
) -> xr.Dataset:
    """Read the requested QA masks and add them to *ds*.

    :param ds: dataset to add mask variables to.
    :param masks: mask variable names to read (see :py:data:`MASK_OPTIONS`).
    :param layout: layout helper used to locate the QA file.
    :param subset: resolved ROI subset to clip the masks to, or ``None``.
    :param config: resolved reader configuration.
    :return: *ds* with the mask variables merged in.
    :raises ValueError: if an unknown mask name is requested.
    """
    unknown = [m for m in masks if m not in QA_BANDS]
    if unknown:
        raise ValueError(f"Unknown constellr Skybee mask(s): {unknown}. Available: {MASK_OPTIONS}")

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
    masks: list[str],
    layout: ConstellrSkybeeLayout,
    subset: ResolvedROISubset | None = None,
    chunks: dict[str, int] | None = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read QA layers from the VNIR04 QA raster into *ds*, on the 10 m grid.

    :param ds: dataset to add mask variables to.
    :param masks: mask variable names to read.
    :param layout: layout helper used to locate the QA file.
    :param subset: resolved ROI subset to clip the masks to, or ``None``.
    :param chunks: explicit dask chunk spec passed to ``rioxarray``.
    :param use_chunks: whether to open the raster lazily (chunked).
    :return: *ds* with the mask variables merged in.
    """
    rxr = lazy_rioxarray()

    # mask_and_scale: the QA tiffs' nodata (255) -> NaN, flags become float32
    da = rxr.open_rasterio(layout.qa_file(_REF_BAND), chunks=chunks if use_chunks else None, mask_and_scale=True)
    if subset is not None and subset.geometries:
        da = da.rio.clip(subset.geometries)

    for name in masks:
        layer = da.isel(band=QA_BANDS[name] - 1, drop=True)
        layer = layer.rename({d: f"{d}_{_RES}" for d in ("x", "y") if d in layer.dims})
        layer.attrs = dict(MASK_ATTRS[name])
        ds[name] = layer

    return ds


if __name__ == "__main__":
    pass
