import xarray as xr
from pathlib import Path
from eoio.deps import lazy_rioxarray
from eoio.utils.rasterio_utils import suggest_raster_chunks

def append_data_vars(
    ds: xr.Dataset,
    layout,
    subset=None,
    chunks=None,
    use_chunks=False,
) -> xr.Dataset:

    rxr = lazy_rioxarray()

    if use_chunks and chunks is None:
        first_path = layout
        chunks = suggest_raster_chunks(str(first_path))

    da = rxr.open_rasterio(layout, chunks=chunks).squeeze()

    if subset and subset.geometries is not None:
        da = da.rio.clip(subset.geometries)

    ds["elevation"] = da

    return ds