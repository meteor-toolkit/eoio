"""eoio.readers.sentinel2.data_io - data reading functionality for Sentinel-2 IMG_DATA layers"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.sentinel2.layout import S2Layout
from eoio.readers.sentinel2.metadata.extractor import S2MSIMetadataExtractor
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.utils.rasterio_utils import suggest_raster_chunks


def _grid_key_from_dataarray(da: xr.DataArray) -> str:
    """
    Derive a Sentinel-2 grid key from raster resolution.

    :param da:
        Raster DataArray opened with rioxarray.
    :returns:
        Grid key such as ``"10m"``, ``"20m"``, or ``"60m"``.
    """
    try:
        xres, _ = da.rio.resolution()
        return f"{int(round(abs(float(xres))))}m"
    except Exception:
        return "grid"


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    layout: S2Layout,
    meas_vars: List[str],
    subset: Optional[ResolvedROISubset],
    mtd: S2MSIMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
    preferred_resolution: Optional[int] = None,
    mask_zero_as_nodata: bool = True,
) -> xr.Dataset:
    """
    Read Sentinel-2 JP2 IMG_DATA layers into an existing Dataset.

    This function performs low-level raster IO: it opens each requested JP2
    IMG_DATA layer using ``rioxarray``, applies variable-specific decoding,
    optionally subsets the data spatially, and inserts the resulting
    DataArray into the provided Dataset.

    For reflectance bands, additive offsets and quantification scaling are
    applied using metadata extracted from the product XML. For L2A auxiliary
    layers such as ``AOT`` and ``WVP``, dedicated quantification values are
    used where available. Categorical or browse layers such as ``SCL`` and
    ``TCI`` are left unscaled.

    Dimension names are rewritten using the raster resolution, for example
    ``x_10m`` / ``y_10m`` or ``x_60m`` / ``y_60m``, so variables on different
    native grids can coexist safely in the same Dataset.

    :param ds:
        Dataset to which band DataArrays will be added.
    :param layout:
        Sentinel-2 SAFE layout helper used to locate JP2 image files.
    :param meas_vars:
        List of measurement variable names to read.
    :param subset:
        Optional resolved region-of-interest subset defining a clip box and/or
        vector geometries.
    :param mtd:
        Metadata extractor providing product-level and variable-level metadata
        used for decoding the requested layers.
    :param chunks:
        Optional mapping of dimension names to chunk sizes passed directly to
        ``rioxarray.open_rasterio``. If provided, this overrides any
        automatically derived chunk sizes.
    :param use_chunks:
        If ``True`` and ``chunks`` is not provided, enable Dask-backed lazy
        reading using block-aligned chunk sizes inferred from the raster
        metadata.
    :param preferred_resolution:
        Optional preferred IMG_DATA resolution in metres. If supplied, layout
        selection will attempt to use this resolution when available.
    :param mask_zero_as_nodata:
        If ``True``, raw values equal to zero are masked as missing data for
        reflectance bands before radiometric scaling is applied.
    :returns:
        The input Dataset with the requested variables added.
    :raises ValueError:
        If reflectance quantification metadata is missing or invalid when
        reflectance bands are requested.
    """
    rxr = lazy_rioxarray()

    # ------------------------------------------------------------------
    # Product metadata used for scaling
    # ------------------------------------------------------------------
    prod_mtd = mtd.product_metadata or {}

    quant_values = prod_mtd.get("quantification_level") or {}
    reflectance_offsets = prod_mtd.get("reflectance_offsets") or {}

    # ------------------------------------------------------------------
    # Resolve image paths
    # ------------------------------------------------------------------
    band_paths = layout.img_jp2_paths(
        meas_vars,
        prefer_res_m=preferred_resolution,
    )

    # ------------------------------------------------------------------
    # Set up chunking
    # ------------------------------------------------------------------
    if use_chunks and chunks is None:
        first_path = next(iter(band_paths.values()))
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    # ------------------------------------------------------------------
    # Read each layer
    # ------------------------------------------------------------------
    for var, path in band_paths.items():
        var_u = var.upper()
        var_mtd = mtd.variable_product_metadata(var) or {}
        offset_legacy = var_mtd.get("radiometric_offset", 0.0)

        da = rxr.open_rasterio(path, chunks=chunks)

        # Collapse single-band rasters; keep TCI as RGB
        if "band" in da.dims:
            if var_u == "TCI" and da.sizes.get("band", 0) == 3:
                da = da.rename({"band": "rgb"})
                da = da.assign_coords(rgb=["R", "G", "B"])
            else:
                da = da.isel(band=0, drop=True)

        # --------------------------------------------------------------
        # Mask nodata before scaling
        # --------------------------------------------------------------
        try:
            encoded_nodata = da.rio.encoded_nodata
        except Exception:
            encoded_nodata = None

        if encoded_nodata is not None:
            da = da.where(da != encoded_nodata)

        if mask_zero_as_nodata and var_u.startswith("B"):
            da = da.where(da != 0)

        # --------------------------------------------------------------
        # Variable-specific decoding
        # --------------------------------------------------------------
        if var_u.startswith("B"):
            q = quant_values.get("reflectance")
            if q in (None, 0):
                raise ValueError(f"Invalid reflectance quantification value in product metadata: {q!r}")

            off = reflectance_offsets.get(var_u, offset_legacy)

            da = da.astype("float32")
            da += np.float32(off)
            da /= np.float32(q)

        elif var_u == "AOT":
            q = quant_values.get("aot")
            da = da.astype("float32")
            if q not in (None, 0):
                da /= np.float32(q)

        elif var_u == "WVP":
            q = quant_values.get("wvp")
            da = da.astype("float32")
            if q not in (None, 0):
                da /= np.float32(q)

        elif var_u == "SCL":
            da = da.astype("int16")

        elif var_u == "TCI":
            # Keep as stored
            pass

        else:
            # Unknown / future layer type: leave as read
            pass

        # --------------------------------------------------------------
        # Apply subset clipping
        # --------------------------------------------------------------
        if subset is not None:
            if subset.clip_box:
                x_min, y_min, x_max, y_max = subset.clip_box
                da = da.rio.clip_box(x_min, y_min, x_max, y_max)

            if subset.geometries:
                da = da.rio.clip(subset.geometries, from_disk=True)

        # --------------------------------------------------------------
        # Rename spatial dimensions by actual raster resolution
        # --------------------------------------------------------------
        grid = _grid_key_from_dataarray(da)

        rename_map = {}
        if "x" in da.dims:
            rename_map["x"] = f"x_{grid}"
        if "y" in da.dims:
            rename_map["y"] = f"y_{grid}"
        if rename_map:
            da = da.rename(rename_map)

        # Strip attrs that rioxarray may attach to spatial coords
        for coord_name in da.coords:
            if coord_name == "x" or coord_name == "y" or coord_name.startswith("x_") or coord_name.startswith("y_"):
                da.coords[coord_name].attrs = {}

        ds[var] = da

    return ds


if __name__ == "__main__":
    pass
