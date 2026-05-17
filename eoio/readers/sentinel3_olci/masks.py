import xarray as xr
from eoio.readers.sentinel3_olci.layout import S3OLCILayout
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.readers.base import ReaderConfig
from eoio.readers.sentinel3_olci.data_io import lazy_rioxarray
from eoio.utils.rasterio_utils import suggest_raster_chunks
from typing import Optional, Dict, Any, List


def add_masks(
    *,
    ds: xr.Dataset,
    masks: list[str],
    layout: S3OLCILayout,
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
    layout: S3OLCILayout,
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

    meas = config.vars_sel.get("meas", None)

    mask_path = layout.quality_flags_path()

    if not masks:
        return ds

    # add individual saturated meas if requested
    if "saturated" in masks and meas is not None:
        saturated_masks = [f"saturated@{i}" for i in meas]
        masks = saturated_masks + [i for i in masks if i != "saturated"]

    # set up chunking
    if use_chunks and chunks is None and mask_path:
        if hasattr(mask_path, "values"):
            first_path = next(iter(mask_path.values()))
        else:
            first_path = mask_path
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    # open ds
    mask_ds = rxr.open_rasterio(mask_path, chunks=chunks).squeeze()

    if subset and subset.xy_clip_box is not None:
        x_min, y_min, x_max, y_max = subset.xy_clip_box
        mask_ds = mask_ds.rio.write_crs(4326)
        mask_ds = mask_ds.rio.clip_box(x_min, y_min, x_max, y_max)

    # extract flag info
    flag_meanings = mask_ds.flag_meanings.split()
    flag_masks = [f"{int(i)}" for i in list(mask_ds.flag_masks)]

    # # get flag value for all non-requested masks
    # non_requested_masks = sum(
    #     [int(i[1]) for i in zip(flag_meanings, flag_masks) if i[0] not in masks]
    # )

    # choose desired flags using self.masks property
    flag_meanings, flag_masks = list(zip(*[i for i in zip(flag_meanings, flag_masks) if i[0] in masks]))  # type: ignore[assignment]

    # assign flags as flag variables
    ds["quality_flags"] = (
        ("y_300m", "x_300m"),
        mask_ds.data,
    )

    ds["quality_flags"].attrs = {
        "flag_meanings": " ".join(flag_meanings),
        "flag_masks": ",".join(flag_masks),
    }  # flag_meanings and flag_masks are joined for flag obsarray format

    return ds
