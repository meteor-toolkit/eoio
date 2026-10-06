"""eoio.readers.planetscope.masks - PlanetScope UDM2 (Usable Data Mask 2) reading

Each PlanetScope scene ships a ``*_udm2*.tif`` alongside its image: a raster on the
same grid as the image, with one band per quality layer. Bands 1-6 are binary (0/1)
classes, band 7 is Planet's classification confidence (0-100) and band 8 is the
legacy UDM1 flag (1 = unusable/no-data pixel). The band order is fixed by Planet's
UDM2 specification and recorded in the file's own band descriptions
(``clear, snow, shadow, haze_light, haze_heavy, cloud, confidence, udm1``), exposed
here as the named variables in :py:data:`MASK_OPTIONS`.

Pixels outside the ROI polygon are NaN, so the variables are float32 -- unlike the
Sentinel-2 mask, which keeps its 0/1 flags as uint8 -- because a flag of 0 outside the
polygon would otherwise be indistinguishable from "not flagged" inside it (and so, for
``clear``, from "not clear").
"""

from typing import Any, List, Optional

import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.planetscope.layout import PlanetScopeLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset

__all__ = ["MASK_OPTIONS", "UDM2_BANDS", "add_masks", "read_masks"]

# Band index (1-based, as stored in the UDM2 tif) of each layer. Order is fixed by the
# Planet UDM2 specification.
UDM2_BANDS = {
    "clear": 1,
    "snow": 2,
    "shadow": 3,
    "haze_light": 4,
    "haze_heavy": 5,
    "cloud": 6,
    "confidence": 7,
    "unusable": 8,
}

MASK_OPTIONS: List[str] = list(UDM2_BANDS)

# Same grid and naming as the image bands (see eoio.readers.planetscope.data_io).
_RES = "3m"

MASK_ATTRS = {
    "clear": {
        "long_name": "Clear mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_clear clear",
    },
    "snow": {
        "long_name": "Snow and ice mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_snow snow",
    },
    "shadow": {
        "long_name": "Shadow mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_shadow shadow",
    },
    "haze_light": {
        "long_name": "Light haze mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_light_haze light_haze",
    },
    "haze_heavy": {
        "long_name": "Heavy haze mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_heavy_haze heavy_haze",
    },
    "cloud": {
        "long_name": "Cloud mask",
        "standard_name": "cloud_binary_mask",
        "flag_values": [0, 1],
        "flag_meanings": "not_cloud cloud",
    },
    "confidence": {
        "long_name": "Classification confidence",
        "units": "percent",
        "valid_range": [0, 100],
    },
    "unusable": {
        "long_name": "Unusable data mask (UDM1)",
        "flag_values": [0, 1],
        "flag_meanings": "usable unusable",
    },
}


def add_masks(
    *,
    ds: xr.Dataset,
    masks: List[str],
    layout: PlanetScopeLayout,
    subset: Optional[ResolvedROISubset],
    config: Any,
) -> xr.Dataset:
    """Read the requested UDM2 mask layers and add them to *ds*.

    Validates the selection, then delegates the IO to :py:func:`read_masks`.

    :param ds: dataset to add mask variables to.
    :param masks: mask variable names to read (see :py:data:`MASK_OPTIONS`).
    :param layout: layout helper used to locate the mask file.
    :param subset: resolved ROI subset to clip the mask to, or ``None``.
    :param config: resolved reader configuration.
    :return: *ds* with the mask variables merged in.
    :raises ValueError: if an unknown mask name is requested.
    """
    unknown = [m for m in masks if m not in UDM2_BANDS]
    if unknown:
        raise ValueError(f"Unknown PlanetScope mask(s): {unknown}. Available: {MASK_OPTIONS}")

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
    layout: PlanetScopeLayout,
    subset: Optional[ResolvedROISubset] = None,
    chunks: Optional[dict] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read UDM2 layers from the scene's ``*_udm2*.tif`` into *ds*.

    The mask shares the image's 3 m grid, so its variables use the same ``x_3m``/``y_3m``
    dimensions as the measurement bands.

    :param ds: dataset to add mask variables to.
    :param masks: mask variable names to read.
    :param layout: layout helper used to locate the mask file.
    :param subset: resolved ROI subset to clip the mask to, or ``None``.
    :param chunks: explicit dask chunk spec passed to ``rioxarray``.
    :param use_chunks: whether to open the raster lazily (chunked).
    :return: *ds* with the mask variables merged in.
    :raises PlanetScopeLayoutError: if the scene has no (or more than one) UDM2 file.
    """
    path = layout.mask_file()

    rxr = lazy_rioxarray()
    da = rxr.open_rasterio(path, chunks=chunks if use_chunks else None)

    for name in masks:
        band = UDM2_BANDS[name]
        # Older UDM2 deliveries carry fewer than 8 bands (no UDM1 flag): fail clearly
        # rather than read a wrong band.
        if "band" not in da.dims or da.sizes["band"] < band:
            raise ValueError(
                f"UDM2 mask {path!r} has {da.sizes.get('band', 0)} band(s), no band {band} for {name!r}."
            )

        # isel is 0-based; UDM2_BANDS records the 1-based band number.
        layer = da.isel(band=band - 1, drop=True)

        if subset is not None:
            if subset.clip_box:
                x_min, y_min, x_max, y_max = subset.clip_box
                layer = layer.rio.clip_box(
                    x_min, y_min, x_max, y_max, auto_expand=True, allow_one_dimensional_raster=True
                )
            if subset.geometries:
                # Cast to float *before* the polygon clip: rioxarray fills the outside of
                # the polygon with NaN, which an integer array silently turns into 0 --
                # i.e. "not clear" -- so the outside would count as non-clear pixels.
                layer = layer.astype("float32").rio.clip(subset.geometries, from_disk=False, drop=True)

        layer = layer.astype("float32")

        rename_map = {d: f"{d}_{_RES}" for d in ("x", "y") if d in layer.dims}
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
