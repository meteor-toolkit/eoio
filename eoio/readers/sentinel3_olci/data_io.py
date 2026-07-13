"""Sentinel-3 OLCI image reading utilities.

This module contains helpers to read radiance bands, uncertainty bands,
latitude/longitude coordinates into an xarray
``Dataset``.

Functions follow Sphinx/reST docstring conventions (``:param:``,
``:type:``, ``:returns:``, ``:rtype:``).
"""

import xarray as xr
import obsarray  # noqa: F401 - registers the obsarray xarray accessor
from typing import List, Dict, Any, Optional, Tuple


from eoio.readers.sentinel3_olci.layout import S3OLCILayout
from eoio.deps import lazy_rioxarray

from eoio.utils.crs_utils import get_nearest_lon_lat_coords
from eoio.readers.sentinel3_olci.metadata.extractor import S3OLCIMetadataExtractor
from eoio.utils.rasterio_utils import suggest_raster_chunks
from eoio.readers.base import ReaderConfig
from eoio.readers.subset.roi_subset import ResolvedROISubset


instrument_aux = [
    "FWHM",
    "detector_index",
    "frame_offset",
    "lambda0",
    "relative_spectral_covariance",
    "solar_flux",
]

meteo_aux = [
    "atmospheric_temperature_profile",
    "horizontal_wind",
    "humidity",
    "sea_level_pressure",
    "total_columnar_water_vapour",
    "total_ozone",
]

MASK_OPTIONS = [
    "saturated",
    "dubious",
    "sun-glint_risk",
    "duplicated",
    "cosmetic",
    "invalid",
    "straylight_risk",
    "bright",
    "tidal_region",
    "fresh_inland_water",
    "coastline",
    "land",
]


def read_bands_into_dataset(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    meas: List[str],
    subset: Optional[ResolvedROISubset],
    mtd: S3OLCIMetadataExtractor,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read radiance bands into an xarray ``Dataset``.

    :param ds: Base dataset to which band variables will be added. The
        dataset is mutated and returned.
    :param layout: Layout helper used to locate band files on disk.
    :param meas: Sequence of band tokens to read (e.g. ``['Oa01']``).
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

    band_paths = layout.radiance_paths(meas)

    # set up chunking
    if use_chunks and chunks is None:
        first_path = next(iter(band_paths.values()))
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    for band, path in band_paths.items():
        if not path:
            continue

        da = rxr.open_rasterio(path, chunks=chunks).squeeze()

        if subset and subset.xy_clip_box is not None:
            x_min, y_min, x_max, y_max = subset.xy_clip_box
            da.rio.write_crs(4326, inplace=True)
            da = da.rio.clip_box(x_min, y_min, x_max, y_max)

        # Optionally drop 'band' dim if present
        if "band" in da.dims:
            da = da.drop_vars("band")

        # Apply scaling and valid range masking
        da = da.where(
            (da.attrs.pop("valid_min") <= da) & (da <= da.attrs.pop("valid_max")) & (0.0 != ds["longitude_300m"].data)
        )
        da = da * da.attrs.pop("scale_factor")

        # Rename coords
        da = da.rename({"x": "x_300m", "y": "y_300m"})

        # Assign to dataset
        ds[band] = da

        # Update attrs
        ds[band].attrs.update({k: v for (k, v) in da.attrs.items() if k not in ["_FillValue", "valid_min"]})

    return ds


def read_uncertainty_into_dataset(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    meas: List[str],
    subset: Optional[ResolvedROISubset],
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read uncertainty (error) bands and attach them to ``ds.unc``.

    :param ds: Dataset to which uncertainty arrays will be attached.
    :param layout: Layout helper for locating uncertainty files.
    :param meas: Bands for which uncertainties should be read.
    :param subset: Optional ROI subset used for clipping.
    :param chunks: Chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Dataset containing attached uncertainty variables.
    """
    rxr = lazy_rioxarray()

    band_paths = layout.requested_uncertainty_paths(meas)

    # set up chunking
    if use_chunks and chunks is None:
        first_path = next(iter(band_paths.values()))
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    for band, path in band_paths.items():
        if not path:
            continue
        da = rxr.open_rasterio(path, chunks=chunks).squeeze()
        if subset and subset.xy_clip_box is not None:
            x_min, y_min, x_max, y_max = subset.xy_clip_box
            da.rio.write_crs(4326, inplace=True)
            da = da.rio.clip_box(x_min, y_min, x_max, y_max)

        # Optionally drop 'band' dim if present
        if "band" in da.dims:
            da = da.drop_vars("band")

        # Rename coords
        da = da.rename({"x": "x_300m", "y": "y_300m"})

        # Scale uncertainty
        da = 10 ** (
            da.where(da < da.attrs.pop("valid_max")) * da.attrs.pop("scale_factor") + da.attrs.pop("add_offset")
        )

        # Assign unc information
        err_corr_def = [
            {
                "dim": ["y_300m", "x_300m"],
                "form": "systematic",
                "params": [],
                "units": [da.attrs["units"].split("(")[-1].split(")")[0]],
            }
        ]

        ds.unc[band][f"u_radiance_{band}"] = (
            ["y_300m", "x_300m"],
            da.data,
            {"err_corr": err_corr_def},
        )
        ds[f"u_radiance_{band}"].attrs.update({"long_name": da.attrs["long_name"].split("scaled ")[-1]})

    return ds


def read_lat_lon_coordinates(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    subset: Optional[ResolvedROISubset] = None,
    config: ReaderConfig,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> Tuple[xr.Dataset, Optional[Dict[str, Any]]]:
    """Read latitude/longitude coordinate arrays and attach to ``ds``.

    Reads the product's ``geo_coordinates.nc`` file via rioxarray and
    normalises the stored integer coordinates to degrees. Optionally applies
    ROI clipping based on a resolved subset and returns a clip-box suitable
    for later use by image readers.

    :param ds: Base dataset to which ``longitude_300m`` and
        ``latitude_300m`` will be assigned and set as coordinates.
    :param layout: Layout helper used to locate the ``geo_coordinates.nc``
        file.
    :param subset: Optional resolved ROI subset; when provided the function
        will compute and return an ``(x_min, y_min, x_max, y_max)`` clip
        box.
    :param chunks: Optional chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Tuple with the updated dataset and an optional clip-box when a
        subset was supplied.
    """

    geo_path = layout.geo_coordinates_path()
    aux_names = config.vars_sel.get("aux", None)
    rxr = lazy_rioxarray()

    clip_boxes = {}

    # set up chunking
    if use_chunks and chunks is None and geo_path:
        if hasattr(geo_path, "values"):
            first_path = next(iter(geo_path.values()))
        else:
            first_path = geo_path
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    # Read lat/lon coordinates
    geo_ds = rxr.open_rasterio(geo_path, chunks=chunks).squeeze().drop_vars(["band", "spatial_ref"])
    lons, lats = (
        geo_ds.longitude * 10**-6,
        geo_ds.latitude * 10**-6,
    )

    # Apply ROI subsetting if provided
    if subset is not None and subset.geometries is not None:
        coords = get_nearest_lon_lat_coords(lons.data, lats.data, list(subset.geometries[0]["coordinates"][0]))
        lons.rio.write_crs(4326, inplace=True), lats.rio.write_crs(4326, inplace=True)

        xs, ys = [i[0] for i in coords], [i[1] for i in coords]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
        xy_clip_box = (x_min, y_min, x_max, y_max)
        lons, lats = (
            lons.rio.clip_box(x_min, y_min, x_max, y_max),
            lats.rio.clip_box(x_min, y_min, x_max, y_max),
        )

        # Store the lat/lon clip box for use by band readers
        clip_boxes["lat_lon"] = xy_clip_box

    geom = [f"{i} m" for i in geo_ds.attrs["resolution"][2:-2].split(" ")]

    lons = lons.rename({"x": "x_300m", "y": "y_300m"})
    lats = lats.rename({"x": "x_300m", "y": "y_300m"})

    for arr in [lons, lats]:
        arr.x_300m.attrs.update({"geometry_id": geom})
        arr.y_300m.attrs.update({"geometry_id": geom})

    # Assign the coords
    ds = ds.assign({"longitude_300m": lons, "latitude_300m": lats}).set_coords(["longitude_300m", "latitude_300m"])

    # Update attrs
    ds["latitude_300m"].attrs.update(
        {
            "units": "degrees_north",
            "standard_name": "latitude 300 m",
            "long_name": "latitude 300 m resolution",
            "geometry_id": geom,
        }
    )

    ds["longitude_300m"].attrs.update(
        {
            "units": "degrees_east",
            "standard_name": "longitude 300 m",
            "long_name": "longitude 300 m resolution",
            "geometry_id": geom,
        }
    )

    if any(i in ["observation_geometry"] + meteo_aux for i in (aux_names or [])):
        ds, tie_xy_clipbox = read_tie_geo_coordinates(
            ds=ds,
            layout=layout,
            subset=subset,
            config=config,
            chunks=chunks,
            use_chunks=use_chunks,
        )
        if tie_xy_clipbox is not None:
            # Store the tie-point clip box for use by auxiliary data readers that require tie-point coordinates
            clip_boxes["tie"] = tie_xy_clipbox

    return ds, clip_boxes if subset is not None else None


def read_tie_geo_coordinates(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    subset: Optional[ResolvedROISubset] = None,
    config: ReaderConfig,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> Tuple[xr.Dataset, Optional[Any]]:
    """Read tie-point grid coordinates and add them to ``ds``.

    Reads the ``tie_geo_coordinates.nc`` file via rioxarray and extracts
    tie-point longitude and latitude arrays, applying any requested
    ROI clipping. The coordinates are scaled using the stored
    ``scale_factor`` attributes and are renamed to ``x_tie``/``y_tie``
    coordinate names before being merged into ``ds``.

    :param ds: Dataset to which tie-point longitude and latitude will be added.
    :param layout: Layout helper for locating the ``tie_geo_coordinates.nc`` file.
    :param subset: Optional resolved ROI subset for clipping; may be ``None``.
    :param config: Reader configuration containing variable selection and read parameters.
    :param chunks: Optional chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Tuple of (updated dataset, tie-point clip box) where the clip box
        is returned when ``subset`` was provided, otherwise ``None``.
    """

    rxr = lazy_rioxarray()

    tie_geo_path = layout.tie_geo_coordinates_path()

    # set up chunking
    if use_chunks and chunks is None and tie_geo_path:
        if hasattr(tie_geo_path, "values"):
            first_path = next(iter(tie_geo_path.values()))
        else:
            first_path = tie_geo_path
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    tie_geo_ds = rxr.open_rasterio(tie_geo_path, chunks=chunks).squeeze().drop_vars(["band", "spatial_ref"])

    tie_lons = tie_geo_ds.longitude * tie_geo_ds.longitude.attrs["scale_factor"]
    tie_lats = tie_geo_ds.latitude * tie_geo_ds.latitude.attrs["scale_factor"]

    # Apply ROI subsetting if provided
    if subset is not None and subset.geometries is not None:
        coords = get_nearest_lon_lat_coords(tie_lons.data, tie_lats.data, list(subset.geometries[0]["coordinates"][0]))
        tie_lons.rio.write_crs(4326, inplace=True), tie_lats.rio.write_crs(4326, inplace=True)

        xs, ys = [i[0] for i in coords], [i[1] for i in coords]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
        tie_xy_clip_box = (x_min, y_min, x_max, y_max)
        tie_lons, tie_lats = (
            tie_lons.rio.clip_box(x_min, y_min, x_max, y_max),
            tie_lats.rio.clip_box(x_min, y_min, x_max, y_max),
        )

    tie_lons = tie_lons.rename({"x": "x_tie", "y": "y_tie"})
    tie_lats = tie_lats.rename({"x": "x_tie", "y": "y_tie"})

    ds["longitude_tie"] = tie_lons
    ds["latitude_tie"] = tie_lats
    ds["longitude_tie"].attrs.update(
        {
            "units": "degrees_east",
            "standard_name": "tie-point longitude",
            "long_name": "longitude on a tie-point grid",
        }
    )
    ds["latitude_tie"].attrs.update(
        {
            "units": "degrees_north",
            "standard_name": "tie-point latitude",
            "long_name": "latitude on a tie-point grid",
        }
    )

    return ds, tie_xy_clip_box if subset is not None else None
