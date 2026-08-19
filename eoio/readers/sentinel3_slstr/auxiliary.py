"""Auxiliary-data helpers for Sentinel-3 SLSTR reader."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import xarray as xr
from eoio.deps import lazy_rioxarray
from eoio.readers.base import ReaderConfig
from eoio.readers.sentinel3_slstr.layout import S3SLSTRLayout
from eoio.readers.subset.roi_subset import ResolvedROISubset
from eoio.utils.rasterio_utils import suggest_raster_chunks, _first_path
from eoio.utils.aux_read import warn_on_aux_failure
from eoio.readers.sentinel3_slstr.utils import GRID_RES_MAP


def add_aux(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    grids: List[str],
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
    use_chunks: bool = False,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Read in auxiliary data and add to an xarray Dataset for SLSTR.

    This helper reads the auxiliary data requested in ``config.vars_sel``
    and merges the content into ``ds``. Supported auxiliary groups include
    meteorological, instrument and geometry-related datasets.

    :param ds: Dataset to update with auxiliary variables.
    :param layout: Layout helper to locate auxiliary files.
    :param grids: Dictionary mapping grid names to their spatial resolutions.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :param use_chunks: Whether to compute and apply chunking heuristics to auxiliary data.
    :param chunks: Optional chunking specification to apply to auxiliary data.
    :returns: Updated dataset with requested auxiliary variables merged in.
    """
    aux_names = config.vars_sel.get("aux", None)
    if not aux_names:
        return ds
    lazy_rioxarray()

    if "cartesian" in aux_names:
        with warn_on_aux_failure("Sentinel-3 SLSTR cartesian geometry"):
            ds = read_cartesian_geometry(
                ds=ds,
                layout=layout,
                grids=grids,
                subset=subset,
                clip_boxes=clip_boxes,
                config=config,
                chunks=config.read_params.get("chunks", None),
                use_chunks=config.read_params.get("use_chunks", False),
            )

    if "indices" in aux_names:
        with warn_on_aux_failure("Sentinel-3 SLSTR indices"):
            ds = read_indices(
                ds=ds,
                layout=layout,
                grids=grids,
                subset=subset,
                clip_boxes=clip_boxes,
                config=config,
                chunks=config.read_params.get("chunks", None),
                use_chunks=config.read_params.get("use_chunks", False),
            )

    if "met" in aux_names:
        with warn_on_aux_failure("Sentinel-3 SLSTR met"):
            ds = read_met(
                ds=ds,
                layout=layout,
                subset=subset,
                clip_boxes=clip_boxes,
                config=config,
                chunks=config.read_params.get("chunks", None),
                use_chunks=config.read_params.get("use_chunks", False),
            )

    if "time" in aux_names:
        with warn_on_aux_failure("Sentinel-3 SLSTR time"):
            ds = read_time(
                ds=ds,
                layout=layout,
                grids=grids,
                subset=subset,
                clip_boxes=clip_boxes,
                config=config,
            )

    if "viscal" in aux_names:
        with warn_on_aux_failure("Sentinel-3 SLSTR viscal"):
            ds = read_viscal(
                ds=ds,
                layout=layout,
                subset=subset,
                clip_boxes=clip_boxes,
                config=config,
            )

    if "observation_geometry" in aux_names:
        with warn_on_aux_failure("Sentinel-3 SLSTR observation geometry"):
            ds = read_observation_geometry(
                ds=ds,
                layout=layout,
                grids=grids,
                subset=subset,
                clip_boxes=clip_boxes,
                config=config,
                use_chunks=config.read_params.get("use_chunks", False),
                chunks=config.read_params.get("chunks", None),
            )

    if "orphan" in aux_names:
        with warn_on_aux_failure("Sentinel-3 SLSTR orphan"):
            ds = read_orphan(
                ds=ds,
                layout=layout,
                subset=subset,
                clip_boxes=clip_boxes,
                config=config,
            )

    return ds


def read_cartesian_geometry(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    grids: List[str],
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
    use_chunks: bool = False,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Read in cartesian geometry auxiliary data and merge into dataset.

    :param ds: Dataset to update with cartesian geometry variables.
    :param layout: Layout helper to locate cartesian geometry files.
    :param grids: Dictionary mapping grid names to their spatial resolutions.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :param use_chunks: Whether to compute and apply chunking heuristics to auxiliary data.
    :param chunks: Optional chunking specification to apply to auxiliary data.
    :returns: Updated dataset with cartesian geometry variables merged in.
    """
    aux_names = config.vars_sel.get("aux", None)
    rxr = lazy_rioxarray()

    for grid in grids:
        # set up chunking
        if use_chunks and chunks is None:
            first_path = _first_path(layout.cartesian_path(grid))
            if first_path:
                chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

        if grid == "tx":
            cartesian_ds = rxr.open_rasterio(layout.cartesian_path(grid), chunks=chunks)
        else:
            cartesian_ds = rxr.open_rasterio(layout.cartesian_path(grid), chunks=chunks)[0]

        cartesian_ds = cartesian_ds.squeeze().drop_vars(["band"])

        if clip_boxes and grid in clip_boxes:
            x_min, y_min, x_max, y_max = clip_boxes[grid]
            cartesian_ds.rio.write_crs(4326, inplace=True)
            cartesian_ds = cartesian_ds.rio.clip_box(x_min, y_min, x_max, y_max)
            cartesian_ds = cartesian_ds.to_dataframe().to_xarray()
            cartesian_ds = cartesian_ds.drop_vars("spatial_ref")

        cartesian_ds = cartesian_ds.rename(
            {
                "x": f"x_{GRID_RES_MAP[grid]}m_{grid}",
                "y": f"y_{GRID_RES_MAP[grid]}m_{grid}",
                f"x_{grid}": f"x_cartesian_{GRID_RES_MAP[grid]}m_{grid}",
                f"y_{grid}": f"y_cartesian_{GRID_RES_MAP[grid]}m_{grid}",
            }
        )

        ds = ds.merge(cartesian_ds, combine_attrs="drop_conflicts")

        if "orphan" in (aux_names or []):
            if grid == "tx":
                pass
            else:
                cartesian_orphan_ds = rxr.open_rasterio(layout.cartesian_path(grid), chunks=chunks)[1]
                cartesian_orphan_ds = cartesian_orphan_ds.squeeze().drop_vars(["band"])
                cartesian_orphan_ds = cartesian_orphan_ds.rename(
                    {
                        "x": f"x_orphan_{GRID_RES_MAP[grid]}m_{grid}",
                        "y": f"y_orphan_{GRID_RES_MAP[grid]}m_{grid}",
                        f"x_orphan_{grid}": f"x_orphan_cartesian_{GRID_RES_MAP[grid]}m_{grid}",
                        f"y_orphan_{grid}": f"y_orphan_cartesian_{GRID_RES_MAP[grid]}m_{grid}",
                    }
                )
                ds = ds.merge(cartesian_orphan_ds, combine_attrs="drop_conflicts")
    return ds


def read_indices(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    grids: List[str],
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
    use_chunks: bool = False,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Read in indices auxiliary data and merge into dataset.

    :param ds: Dataset to update with indices variables.
    :param layout: Layout helper to locate indices files.
    :param grids: Dictionary mapping grid names to their spatial resolutions.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :param use_chunks: Whether to compute and apply chunking heuristics to auxiliary data.
    :param chunks: Optional chunking specification to apply to auxiliary data.
    :returns: Updated dataset with indices variables merged in.
    """
    aux_names = config.vars_sel.get("aux", None)
    rxr = lazy_rioxarray()

    for grid in grids:
        # set up chunking
        if use_chunks and chunks is None:
            first_path = _first_path(layout.indices_path(grid))
            if first_path:
                chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

        if grid == "tx":
            pass
        else:
            indices_ds = rxr.open_rasterio(layout.indices_path(grid), chunks=chunks)[0]
            ds = ds.merge(indices_ds, combine_attrs="drop_conflicts")
            if "orphan" in (aux_names or []):
                indices_orphan_ds = rxr.open_rasterio(layout.indices_path(grid), chunks=chunks)[1]
                ds = ds.merge(indices_orphan_ds, combine_attrs="drop_conflicts")
    return ds


def read_met(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
    use_chunks: bool = False,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Read in meteorological auxiliary data and merge into dataset.

    :param ds: Dataset to update with meteorological variables.
    :param layout: Layout helper to locate meteorological files.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :param use_chunks: Whether to compute and apply chunking heuristics to auxiliary data.
    :param chunks: Optional chunking specification to apply to auxiliary data.
    :returns: Updated dataset with meteorological variables merged in.
    """

    met_path = layout.met_path()
    if met_path is None:
        return ds

    # set up chunking
    if use_chunks and chunks is None:
        first_path = _first_path(met_path)
        if first_path:
            chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

    met_ds = xr.open_dataset(met_path, chunks=chunks)

    ds = ds.merge(met_ds, combine_attrs="drop_conflicts")

    return ds


def read_time(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    grids: List[str],
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
) -> xr.Dataset:
    """Read in time auxiliary data and merge into dataset.

    :param ds: Dataset to update with time variables.
    :param layout: Layout helper to locate time files.
    :param grids: Dictionary mapping grid names to their spatial resolutions.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :returns: Updated dataset with time variables merged in.
    """

    if "an" in grids or "ao" in grids:
        an_path = layout.time_path("an")
        if an_path is not None:
            time_ds = xr.open_dataset(an_path)
            time_ds = time_ds.rename({"rows": "y_an"})
            ds = ds.merge(time_ds, combine_attrs="drop_conflicts")

    if "bn" in grids or "bo" in grids:
        bn_path = layout.time_path("bn")
        if bn_path is not None:
            time_ds = xr.open_dataset(bn_path)
            time_ds = time_ds.rename({"rows": "y_bn"})
            ds = ds.merge(time_ds, combine_attrs="drop_conflicts")

    if "in" in grids or "io" in grids:
        in_path = layout.time_path("in")
        if in_path is not None:
            time_ds = xr.open_dataset(in_path)
            time_ds = time_ds.rename({"rows": "y_in"})
            ds = ds.merge(time_ds, combine_attrs="drop_conflicts")

    return ds


def read_viscal(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
) -> xr.Dataset:
    """Read in VISCAL auxiliary data and merge into dataset.

    :param ds: Dataset to update with VISCAL variables.
    :param layout: Layout helper to locate VISCAL files.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :returns: Updated dataset with VISCAL variables merged in.
    """

    viscal_path = layout.viscal_path()
    if viscal_path is None:
        return ds
    viscal = xr.open_dataset(viscal_path)
    ds = ds.merge(viscal, combine_attrs="drop_conflicts")

    return ds


def read_observation_geometry(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    grids: List[str],
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
    use_chunks: bool = False,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Read in geometry auxiliary data and merge into dataset.

    :param ds: Dataset to update with geometry variables.
    :param layout: Layout helper to locate geometry files.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :param use_chunks: Whether to compute and apply chunking heuristics to auxiliary data.
    :param chunks: Optional chunking specification to apply to auxiliary data.
    :returns: Updated dataset with geometry variables merged in.
    """

    rxr = lazy_rioxarray()

    if "an" in grids or "bn" in grids or "in" in grids:
        # set up chunking
        if use_chunks and chunks is None:
            first_path = _first_path(layout.geometry_tn_path())
            if first_path:
                chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

        nadir_angle_ds = rxr.open_rasterio(layout.geometry_tn_path(), chunks=chunks)
        nadir_angle_ds = nadir_angle_ds.squeeze().drop_vars(["band"])
        ds_attrs = nadir_angle_ds.attrs
        attr_dict = {i: nadir_angle_ds[i].attrs for i in nadir_angle_ds}

        if clip_boxes and "tx" in clip_boxes:
            x_min, y_min, x_max, y_max = clip_boxes["tx"]
            nadir_angle_ds.rio.write_crs(4326, inplace=True)
            nadir_angle_ds = nadir_angle_ds.rio.clip_box(x_min, y_min, x_max, y_max)
            nadir_angle_ds = nadir_angle_ds.to_dataframe().to_xarray()
            nadir_angle_ds = nadir_angle_ds.drop_vars("spatial_ref")

        nadir_angle_ds = nadir_angle_ds.rename({"x": "x_16000m_tx", "y": "y_16000m_tx"})
        nadir_angle_ds.attrs.update(ds_attrs)

        # add additional attrs from geometries
        common_attrs = ds.attrs.keys() & nadir_angle_ds.attrs.keys()

        for common_attr in common_attrs:
            if ds.attrs[common_attr] != nadir_angle_ds.attrs[common_attr]:
                ds.attrs[common_attr] = {
                    "read": ds.attrs[common_attr],
                    "geometry": nadir_angle_ds.attrs[common_attr],
                }
            nadir_angle_ds.attrs.pop(common_attr)

        ds = ds.merge(nadir_angle_ds, combine_attrs="drop_conflicts")

        for i in nadir_angle_ds:
            ds[i].attrs.update(attr_dict[i])
        for var in nadir_angle_ds.data_vars:
            ds[var] = ds[var].assign_coords(
                {
                    f"lon_{GRID_RES_MAP['tx']}m_tx": ds.coords[f"lon_{GRID_RES_MAP['tx']}m_tx"],
                    f"lat_{GRID_RES_MAP['tx']}m_tx": ds.coords[f"lat_{GRID_RES_MAP['tx']}m_tx"],
                }
            )

    if "ao" in grids or "bo" in grids or "io" in grids:
        # set up chunking
        if use_chunks and chunks is None:
            first_path = _first_path(layout.geometry_to_path())
            if first_path:
                chunks = suggest_raster_chunks(str(first_path), target_mb=32.0)

        ob_angle_ds = rxr.open_rasterio(layout.geometry_to_path(), chunks=chunks)
        ob_angle_ds = ob_angle_ds.squeeze().drop_vars(["band"])
        ds_attrs = ob_angle_ds.attrs
        attr_dict = {i: ob_angle_ds[i].attrs for i in ob_angle_ds}

        if clip_boxes and "tx" in clip_boxes:
            x_min, y_min, x_max, y_max = clip_boxes["tx"]
            ob_angle_ds.rio.write_crs(4326, inplace=True)
            ob_angle_ds = ob_angle_ds.rio.clip_box(x_min, y_min, x_max, y_max)
            ob_angle_ds = ob_angle_ds.to_dataframe().to_xarray()
            ob_angle_ds = ob_angle_ds.drop_vars("spatial_ref")

        ob_angle_ds = ob_angle_ds.rename({"x": "x_16000m_tx", "y": "y_16000m_tx"})
        ob_angle_ds.attrs.update(ds_attrs)
        ds = ds.merge(ob_angle_ds, combine_attrs="drop_conflicts")
        for i in ob_angle_ds:
            ds[i].attrs.update(attr_dict[i])
        for var in ob_angle_ds.data_vars:
            ds[var] = ds[var].assign_coords(
                {
                    f"lon_{GRID_RES_MAP['tx']}m_tx": ds.coords[f"lon_{GRID_RES_MAP['tx']}m_tx"],
                    f"lat_{GRID_RES_MAP['tx']}m_tx": ds.coords[f"lat_{GRID_RES_MAP['tx']}m_tx"],
                }
            )

    return ds


def read_orphan(
    *,
    ds: xr.Dataset,
    layout: S3SLSTRLayout,
    subset: ResolvedROISubset,
    clip_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    config: ReaderConfig,
    use_chunks: bool = False,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Read in orphan auxiliary data and merge into dataset.

    :param ds: Dataset to update with orphan variables.
    :param layout: Layout helper to locate orphan files.
    :param subset: Optional resolved ROI subset for clipping auxiliary data.
    :param clip_boxes: Optional pre-computed clip boxes for auxiliary data.
    :param config: Reader configuration containing variable selection and read parameters.
    :param use_chunks: Whether to compute and apply chunking heuristics to auxiliary data.
    :param chunks: Optional chunking specification to apply to auxiliary data.
    :returns: Updated dataset with orphan variables merged in.
    """

    rxr = lazy_rioxarray()

    meas = config.vars_sel.get("meas", None)
    band_paths = layout.meas_paths(meas)
    bands = band_paths.keys()

    for bnd in bands:
        orphan_ds = rxr.open_rasterio(band_paths[bnd])
        orphan_var = bnd[:-3] + "_orphan_" + bnd[-2:]
        grid = bnd[-2:]
        orphan_data = orphan_ds[1][orphan_var]
        orphan_data = orphan_data.squeeze().drop_vars(["band"])
        orphan_data = orphan_data.rename(
            {
                "x": f"x_orphan_{GRID_RES_MAP[grid]}m_{grid}",
                "y": f"y_orphan_{GRID_RES_MAP[grid]}m_{grid}",
            }
        )

        ds = ds.merge(orphan_data, combine_attrs="drop_conflicts")

    return ds
