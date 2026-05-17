"""eoio.readers.sentinel3_olci.aux - Auxiliary data reading for Sentinel-3 OLCI."""

import xarray as xr
from eoio.deps import lazy_rioxarray
from eoio.readers.sentinel3_olci.layout import S3OLCILayout
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.utils.rasterio_utils import suggest_raster_chunks
import numpy as np
from eoio.readers.base import ReaderConfig
from typing import Optional, Dict
import warnings

# --- Auxiliary data sub-types ---
meteo_aux = [
    "atmospheric_temperature_profile",
    "horizontal_wind",
    "humidity",
    "sea_level_pressure",
    "total_columnar_water_vapour",
    "total_ozone",
]

instrument_aux = [
    "FWHM",
    "detector_index",
    "frame_offset",
    "lambda0",
    "relative_spectral_covariance",
    "solar_flux",
]


def add_aux(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    subset: ResolvedROISubset,
    config: ReaderConfig,
) -> xr.Dataset:
    """Read in auxiliary data and add to an xarray Dataset for OLCI.

    This helper reads the auxiliary data requested in ``config.vars_sel``
    and merges the content into ``ds``. Supported auxiliary groups include
    meteorological, instrument and geometry-related datasets.

    :param ds: Dataset to update with auxiliary variables.
    :param layout: Layout helper to locate auxiliary files.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :returns: Updated dataset with requested auxiliary variables merged in.
    """
    aux_names = config.vars_sel.get("aux", None)
    meas = config.vars_sel.get("meas", None)

    lazy_rioxarray()

    # --- Observation geometry ---
    if "observation_geometry" in (aux_names or []):
        ds = read_obs_geometry(
            ds=ds,
            layout=layout,
            subset=subset,
            config=config,
            chunks=config.read_params.get("chunks", None),
            use_chunks=config.read_params.get("use_chunks", False),
        )

    # --- Removed pixels ---

    if "removed_pixels" in (aux_names or []) and meas is not None:
        ds = read_removed_pixels(
            ds=ds,
            layout=layout,
            subset=subset,
            config=config,
            chunks=config.read_params.get("chunks", None),
            use_chunks=config.read_params.get("use_chunks", False),
        )

    # --- Instrument data ---

    if any(aux in (aux_names or []) for aux in instrument_aux):
        ds = read_instrument_aux(
            ds=ds,
            layout=layout,
            subset=subset,
            config=config,
            chunks=config.read_params.get("chunks", None),
            use_chunks=config.read_params.get("use_chunks", False),
        )

    # --- Meteorological data ---

    if any(aux in (aux_names or []) for aux in meteo_aux):
        ds = read_meteo_aux(
            ds=ds,
            layout=layout,
            subset=subset,
            config=config,
            chunks=config.read_params.get("chunks", None),
            use_chunks=config.read_params.get("use_chunks", False),
        )

    return ds


def read_obs_geometry(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    subset: Optional[ResolvedROISubset] = None,
    config: ReaderConfig,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read observation geometry variables from the tie-point file.

    Reads ``tie_geometries.nc`` with rioxarray, applies optional ROI clipping,
    renames observational variables to human-friendly names and merges them
    into ``ds``.

    :param ds: Dataset to which observation geometry variables will be added.
    :param layout: Layout helper for locating the ``tie_geometries.nc`` file.
    :param subset: Optional resolved ROI subset for clipping; may be ``None``.
    :param config: Reader configuration containing variable selection and read parameters.
    :param chunks: Optional chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Updated dataset with observation geometry variables merged in.
    """

    rxr = lazy_rioxarray()

    obs_path = layout.tie_geometries_path()

    # set up chunking
    if use_chunks and chunks is None and obs_path:
        if hasattr(obs_path, "values"):
            first_path = next(iter(obs_path.values()))
        else:
            first_path = obs_path
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    obs_var = rxr.open_rasterio(obs_path).squeeze().drop_vars(["band", "spatial_ref"])

    if subset and subset.tie_clip_box is not None:
        x_min, y_min, x_max, y_max = subset.tie_clip_box
        obs_var = obs_var.rio.write_crs(4326)
        obs_var = obs_var.rio.clip_box(x_min, y_min, x_max, y_max)

    obs_var = obs_var.rename(
        {
            "x": "x_tie",
            "y": "y_tie",
            "OAA": "Azimuth_Viewing_Angles",
            "SAA": "Azimuth_Sun_Angles",
            "OZA": "Zenith_Viewing_Angles",
            "SZA": "Zenith_Sun_Angles",
        }
    )

    for i in obs_var:
        obs_attrs = dict(
            [
                ((k, v * obs_var[i].attrs["scale_factor"]) if isinstance(v, int) else (k, v))
                for k, v in obs_var[i].attrs.items()
            ]
        )
        obs_var[i] = obs_var[i] * obs_attrs.pop("scale_factor")
        obs_var[i].attrs.update(obs_attrs)

    if "title" in obs_var.attrs:
        obs_var.attrs.pop("title")

    # add additional attrs from observational geometries
    common_attrs = ds.attrs.keys() & obs_var.attrs.keys()

    for common_attr in common_attrs:
        if ds.attrs[common_attr] != obs_var.attrs[common_attr]:
            ds.attrs[common_attr] = {
                "read": ds.attrs[common_attr],
                "observation_geometry": obs_var.attrs[common_attr],
            }
        obs_var.attrs.pop(common_attr)

    ds = ds.merge(obs_var, combine_attrs="drop_conflicts")

    return ds


def read_removed_pixels(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    subset: Optional[ResolvedROISubset] = None,
    config: ReaderConfig,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read removed-pixels information and merge into ``ds``.

    Reads ``removed_pixels.nc`` and extracts coordinate arrays and per-pixel
    variables (e.g. SZA, detector index). Optionally reads and decodes
    quality flag masks and merges the results into ``ds``.

    :param ds: Dataset to which removed-pixel data will be added.
    :param layout: Layout helper for locating the ``removed_pixels.nc`` file.
    :param subset: Optional resolved ROI subset for clipping; may be ``None``.
    :param config: Reader configuration containing variable selection and read parameters.
    :param chunks: Optional chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Updated dataset with removed-pixel coordinates, variables and
        optional flag masks merged in.
    """

    rxr = lazy_rioxarray()

    meas = config.vars_sel.get("meas", None)
    masks = config.vars_sel.get("mask", None)

    removed_path = layout.removed_pixels_path()

    if not removed_path:
        return ds

    # set up chunking
    if use_chunks and chunks is None and removed_path:
        if hasattr(removed_path, "values"):
            first_path = next(iter(removed_path.values()))
        else:
            first_path = removed_path
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    removed_dims = (
        rxr.open_rasterio(removed_path, variable=["altitude", "latitude", "longitude"], chunks=chunks)
        .squeeze()
        .drop_vars(["band", "spatial_ref"])
    )

    removed_lons = removed_dims.longitude * removed_dims.longitude.attrs["scale_factor"]
    removed_lats = removed_dims.latitude * removed_dims.latitude.attrs["scale_factor"]
    removed_lons = removed_lons.rename({"x": "x_removed", "y": "y_removed"})
    removed_lats = removed_lats.rename({"x": "x_removed", "y": "y_removed"})

    ds["longitude_removed"] = removed_lons
    ds["latitude_removed"] = removed_lats
    ds["longitude_removed"].attrs.update(
        {
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "longitude for removed pixels",
        }
    )
    ds["latitude_removed"].attrs.update(
        {
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "latitude for removed pixels",
        }
    )

    # get variable data

    pix_var = (
        rxr.open_rasterio(
            removed_path,
            variable=[f"{b}_radiance" for b in (meas or [])] + ["SZA", "detector_index"],
            chunks=chunks,
        )
        .squeeze()
        .drop_vars(["band", "spatial_ref"])
    )

    for i in pix_var:
        pix_var_attrs = dict(
            [
                ((k, v * pix_var[i].attrs["scale_factor"]) if isinstance(v, int) else (k, v))
                for k, v in pix_var[i].attrs.items()
            ]
        )
        pix_var[i] = pix_var[i] * pix_var_attrs.pop("scale_factor")
        pix_var[i].attrs.update(pix_var_attrs)
    pix_var = pix_var.rename(
        {
            **{"x": "x_removed", "y": "y_removed"},
            **{str(i): str(i).split("_")[0] + "_removed" for i in pix_var},
        }
    )

    if "title" in pix_var.attrs:
        pix_var.attrs.pop("title")

    ds = ds.merge(pix_var, combine_attrs="drop_conflicts")

    if masks:
        # add individual saturated meas if requested
        if "saturated" in masks:
            saturated_masks = ["saturated@" + i for i in (meas or [])]
            masks = saturated_masks + [i for i in masks if i != "saturated"]

        # open and read in quality masks
        removed_flags = (
            rxr.open_rasterio(removed_path, variable="quality_flags", chunks=chunks)
            .squeeze()
            .drop_vars(["band", "spatial_ref"])
        )

        # get list of flag meanings and flag masks
        flag_meanings = removed_flags.quality_flags.flag_meanings.split()[::-1]
        flag_masks = [f"{int(i)}" for i in list(removed_flags.quality_flags.flag_masks)[::-1]]

        # get flag value for all non-requested masks

        # choose desired flags using self.masks property
        flag_meanings, flag_masks = list(  # type: ignore[assignment]
            zip(*[i for i in zip(flag_meanings, flag_masks) if i[0] in masks])
        )

        # assign flags as flag variables
        ds["removed_quality_flags"] = (
            ("y_removed", "x_removed"),
            removed_flags.quality_flags.data,
        )

        # add more attributes here
        ds.removed_quality_flags.attrs = removed_flags.quality_flags.attrs

        ds.removed_quality_flags.attrs.update(
            {
                "flag_meanings": " ".join(flag_meanings),
                "flag_masks": ",".join(flag_masks),
            }
        )

    return ds


def read_instrument_aux(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    subset: Optional[ResolvedROISubset] = None,
    config: ReaderConfig,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read instrument auxiliary data (e.g. FWHM, solar flux) and merge in.

    Opens the instrument data file and selects requested variables. When
    ``detector_index`` is requested it is read and regridded to match the
    product coordinates before assignment.

    :param ds: Dataset to which instrument auxiliary variables will be added.
    :param layout: Layout helper for locating the instrument auxiliary file.
    :param subset: Optional resolved ROI subset for clipping; may be ``None``.
    :param config: Reader configuration containing variable selection and read parameters.
    :param chunks: Optional chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Updated dataset with instrument auxiliary variables merged in.
    """

    lazy_rioxarray()

    aux_path = layout.instrument_data_path()
    aux_names = config.vars_sel.get("aux", None)
    non_selected_instr = [i for i in instrument_aux if i not in (aux_names or [])]

    # set up chunking
    if use_chunks and chunks is None and aux_path:
        if hasattr(aux_path, "values"):
            first_path = next(iter(aux_path.values()))
        else:
            first_path = aux_path
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    # Open auxiliary file and read requested variables
    instr_ds = xr.open_dataset(aux_path, chunks=chunks).drop_vars(list(set(non_selected_instr + ["detector_index"])))

    # instr_ds = instr_ds.rename({"columns": "x", "rows": "y"})
    # instr_ds = instr_ds.assign_coords({"x": instr_ds.x, "y":instr_ds.y})
    # instr_ds = instr_ds.rio.write_crs(4326)

    # columns, rows only dims for detector_index and frame_offset data_vars
    if "columns" in instr_ds.dims:
        instr_ds = instr_ds.rename({"columns": "x_instr", "rows": "y_instr"})

    instr_ds_attrs = instr_ds.attrs

    instr_ds.attrs.update(instr_ds_attrs)

    # Get detector index if requested
    if "detector_index" in (aux_names or []):
        instr_det = (
            xr.open_dataset(aux_path, chunks=chunks)
            .squeeze()
            .drop_vars(list(set(instrument_aux) - set(["detector_index"])))
        )

        # Update coordinates to match product x/y coordinates
        instr_det = instr_det.rename({"columns": "x", "rows": "y"})
        instr_det = instr_det.assign_coords({"x": instr_det.x, "y": instr_det.y})

        if not np.array_equal(instr_det.y.data, ds.y_300m.data) and not np.array_equal(
            instr_det.x.data, ds.x_300m.data
        ):
            # subset to match meas_var data
            if subset and subset.xy_clip_box is not None:
                x_min, y_min, x_max, y_max = subset.xy_clip_box
                instr_det.rio.write_crs(4326, inplace=True)
                instr_det = instr_det.rio.clip_box(x_min, y_min, x_max, y_max)
                warnings.warn(
                    "Clipping detector_index using ROI does not exactly match band data clipping, due to 0.5 offset in x and y coordinates."
                )

        instr_det = instr_det.rename({"x": "x_300m", "y": "y_300m"})
        instr_det_attrs = instr_det.attrs

        instr_det.attrs.update(instr_det_attrs)

        # Note: solar flux found from detector_index has np.nanmean difference between adjacent fluxes of 0.0025960484

        instr_det["x_300m_cor"], instr_det["y_300m_cor"] = (
            instr_det.x_300m.data[:-1] + 0.5,
            instr_det.y_300m.data[:-1] + 0.5,
        )
        instr_det = instr_det.assign(
            {
                "detector_index_cor": (
                    ("y_300m_cor", "x_300m_cor"),
                    instr_det.detector_index.data[:-1, :-1],
                )
            }
        )
        instr_det = instr_det.drop_vars(["x_300m", "y_300m", "detector_index"])
        instr_det = instr_det.rename(
            {
                "x_300m_cor": "x_300m",
                "y_300m_cor": "y_300m",
                "detector_index_cor": "detector_index",
            }
        )

        ds["detector_index"] = instr_det.detector_index

    try:
        instr_ds.attrs.pop("title")
        instr_ds.attrs.pop("comment")
    except KeyError:
        pass

    ds = ds.merge(instr_ds, combine_attrs="override")

    return ds


def read_meteo_aux(
    *,
    ds: xr.Dataset,
    layout: S3OLCILayout,
    subset: Optional[ResolvedROISubset] = None,
    config: ReaderConfig,
    chunks: Optional[Dict[str, int]] = None,
    use_chunks: bool = False,
) -> xr.Dataset:
    """Read meteorological auxiliary data and merge into ``ds``.

    Reads the tie meteo file and extracts requested meteorological fields such
    as atmospheric temperature profiles, horizontal wind, humidity,
    surface pressure and columnar water vapour. Data are optionally clipped
    to the ROI and merged into the provided dataset.

    :param ds: Dataset to which meteorological auxiliary variables will be added.
    :param layout: Layout helper for locating the tie meteo file.
    :param subset: Optional resolved ROI subset for clipping; may be ``None``.
    :param config: Reader configuration containing variable selection and read parameters.
    :param chunks: Optional chunking specification for raster reads.
    :param use_chunks: Whether to compute and apply chunking heuristics.
    :returns: Updated dataset with meteorological auxiliary variables merged in.
    """

    rxr = lazy_rioxarray()

    aux_path = layout.tie_meteo_path()

    aux_names = config.vars_sel.get("aux", None)

    # set up chunking
    if use_chunks and chunks is None and aux_path:
        if hasattr(aux_path, "values"):
            first_path = next(iter(aux_path.values()))
        else:
            first_path = aux_path
        chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    # Open auxiliary file and read requested variables

    meteo_var = [
        i.squeeze().drop_vars("spatial_ref")  # type: ignore[union-attr]
        for i in rxr.open_rasterio(aux_path, chunks=chunks)
    ]

    if "atmospheric_temperature_profile" in (aux_names or []):
        met_0 = meteo_var[0]["atmospheric_temperature_profile"].rename({"band": "y", "x": "profile_altitude", "y": "x"})

        met_0["y_tie"], met_0["x_tie"] = met_0.y[:-1] + 0.5, met_0.x
        met_0 = met_0.swap_dims({"x": "x_tie", "y": "y_tie"})
        met_0 = met_0.drop_vars(["x", "y"])
        met_0 = met_0[:-1, :, :]

        if not np.array_equal(met_0.y_tie.data, ds.y_tie.data):
            met_0["y_tie"] = ds.y_tie.data

        ds = ds.merge(met_0, combine_attrs="drop_conflicts")

    if "horizontal_wind" in (aux_names or []):
        met_1 = meteo_var[1]["horizontal_wind"].rename({"band": "y", "x": "direction", "y": "x"})

        met_1["y_tie"], met_1["x_tie"] = met_1.y[:-1] + 0.5, met_1.x
        met_1 = met_1.swap_dims({"x": "x_tie", "y": "y_tie"})
        met_1 = met_1.drop_vars(["x", "y"])
        met_1 = met_1[:-1, :, :]

        if not np.array_equal(met_1.y_tie.data, ds.y_tie.data):
            met_1["y_tie"] = ds.y_tie.data
        met_1.direction.attrs.update({"long_name": "Horizontal wind direction components"})

        ds = ds.merge(met_1, combine_attrs="drop_conflicts")

    if any([i in meteo_aux[2:] for i in (aux_names or [])]):
        met_2 = meteo_var[2].drop_vars([i for i in meteo_aux[2:] if i not in (aux_names or [])])

        met_2 = met_2.rename({"x": "x_tie", "y": "y_tie"})
        met_2.attrs.pop("title")

        ds = ds.merge(met_2, combine_attrs="drop_conflicts")

    return ds
