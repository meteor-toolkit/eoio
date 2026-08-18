import xarray as xr
from eoio.readers.copernicus_dem.layout import CopernicusDEMLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.readers.base import ReaderConfig
from eoio.readers.copernicus_dem.data_io import lazy_rioxarray
from eoio.utils.rasterio_utils import suggest_raster_chunks
from typing import Optional, Dict, Any, List


def add_masks(
    *,
    ds: xr.Dataset,
    masks: list[str],
    layout: CopernicusDEMLayout,
    subset: Optional[ResolvedROISubset],
    config: Any,
):
    """Read quality masks and add them to an xarray Dataset.

    This function delegates to ``read_masks`` which performs the actual
    IO and decoding; here we perform a small validation of selected masks
    before calling the helper.

    :param ds: Dataset to which quality mask variables will be added.
    :param masks: List of requested mask names to read and add to the dataset.
    :param layout: Layout helper for locating the relevant mask file.
    :param subset: Resolved ROI subset for subsetting the mask data, or ``None``.
    :param config: Resolved reader configuration containing parameters used by the reader.
    :returns: Dataset with mask variables merged in.
    """

    mask_names = config.vars_sel.get("mask", None)

    if mask_names is None:
        raise ValueError("No masks have been selected. Please set mask_names before reading.")

    ds = read_masks(
        ds=ds,
        masks=masks,
        layout=layout,
        subset=subset,
        config=config,
        use_chunks=config.read_params.get("use_chunks", False),
        chunks=config.read_params.get("chunks", None),
    )

    return ds


def read_masks(
    *,
    ds: xr.Dataset,
    masks: List[str],
    layout: CopernicusDEMLayout,
    subset: Optional[ResolvedROISubset] = None,
    config: ReaderConfig,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read quality masks and add to dataset.

    Reads quality mask data from the appropriate file via rioxarray and adds
    the mask variables to the dataset with appropriate metadata. This
    function is used when the user has requested any quality masks (e.g.
    cloud mask).

    :param ds: Dataset to which quality mask variables will be added.
    :param masks: List of requested mask names to read and add to the
        dataset.
    :param layout: Layout helper for locating the relevant mask file.
    :param subset: Optional resolved ROI subset; not currently used but
        included for potential future use in subsetting mask data.
    :param config: Reader configuration object containing variable selection
        and other parameters.
    :param chunks: Optional chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Updated dataset with quality mask variables added.
    """

    rxr = lazy_rioxarray()

    MASK_PATHS = {
        "water_body_mask": layout.wbm,
        "filling_mask": layout.flm,
        "editing_mask": layout.edm,
        "height_error_mask": layout.hem,
    }
    if not masks:
        return ds

    for mask_name in masks:
        mask_path = MASK_PATHS[mask_name]

        if use_chunks and chunks is None:
            chunks = suggest_raster_chunks(str(mask_path), target_mb=32.0)

        da = rxr.open_rasterio(
            mask_path,
            chunks=chunks,
        ).squeeze()

        if subset is not None:
            da = da.rio.write_crs(4326)
            da = da.rio.clip(subset.geometries)
        ds[mask_name] = da
    return ds
