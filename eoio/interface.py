"""eoio.interface - interface functions module"""

import glob
from typing import Any, Callable, Dict, List, Optional, Union
import xarray as xr
import os
from processor_tools import Context
from eoio.processors.processor_pipeline import ProcessorPipeline
from eoio.processors.registry import PROCESSOR_REGISTRY
from eoio.processors.stack._concat import _concat_datasets
from eoio.readers.factory import ReaderFactory
from eoio.utils.read_utils import setup_file

__all__ = [
    "read",
    "read_multi",
    # "product_bounds",
    "product_processors",
    # "mid_lon_lat",
    "product_options",
]


def read(
    path: str,
    vars_sel: Optional[Dict[str, List[str]]] = None,
    subset: Optional[Dict[str, Any]] = None,
    read_params: Optional[Dict[str, Any]] = None,
    processors: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    *args,
    **kwargs,
) -> xr.Dataset:
    """
    Reads an Earth Observation (EO) data product and returns the requested variables,
    optionally applying spatial subsetting and post-processing.

    :param path:
        Path to the EO data product (file or directory, depending on product type).

    :param vars_sel:
        Variable selection dictionary defining which variables to read from the product.

        Supported keys are:

        * ``"meas"`` (*list[str] | str | None*, default: ``"all"``) –
          Measurement variables to read. Options are:

          - ``"all"`` – read all available measurement variables
          - ``list[str]`` – explicit list of measurement variable names
          - ``None`` – do not read any measurement variables

        * ``"mask"`` (*list[str] | None*, default: ``None``) –
          Mask variable names to read. Use ``None`` to disable mask reading.

        * ``"aux"`` (*list[str] | None*, default: ``None``) –
          Auxiliary data variable names to read. Use ``None`` to disable auxiliary data reading.

        Available variable names and supported combinations for a given product
        can be inspected via ``eoio.product_options``.

    :param subset:
        Optional definition of subsetting parameters. If omitted or ``None``,
        the full data product is read without subsetting.

        Supported keys (availability depends on the product type):

        * ``"roi"`` –
          Spatial region of interest for raster data. Supported forms are:

          - ``None`` – no spatial subsetting
          - ``shapely`` geometry (interpreted in ``roi_crs_epsg``)
          - Bounding box tuple ``(xmin, ymin, xmax, ymax)`` in ``roi_crs_epsg``
          - GeoJSON-like ``dict`` with a ``"type"`` key
          - List of ``[x, y]`` coordinate pairs defining a polygon
          - ``((x, y), half_width_m)`` defining a square region centred on a point

        * ``"roi_crs_epsg"`` (*str*) –
          EPSG code defining the coordinate reference system of ``roi``
          (e.g. ``"EPSG:4326"``).

        * ``"angle"`` – (*Not implemented*) Observation geometry range of interest

        * ``"spectral"`` – (*Not implemented*) Spectral range of interest

    :param read_params:
        Optional parameters controlling how the data are read.
        If omitted, default behaviour is used.

        Supported options include:

        * ``"save_extracted"`` (*bool*, default: ``False``) –
          If the data product is extracted or uncompressed during reading,
          controls whether the extracted files are saved to disk.

        * ``"metadata_level"`` (*None | bool*, default: ``None``) –
          Level of metadata to read:

          - ``None`` – Standard/core metadata only
          - ``False`` – Do not read metadata
          - ``True`` – Read all available metadata

    :param processors:
        Optional definition of post-processing steps to apply after reading
        (e.g. interpolation, unit conversion, etc.).

        Dictionary keys should be the names of the processors to run.
        The associated entry should be a subdictionary of parameters for that processor.
        Required parameters are defined at the processor level.

    :return:
        ``xarray.Dataset`` containing the requested variables and associated
        metadata from the EO data product.

    Examples
    --------
    .. doctest-skip::

    Read all measurement variables from an EO product:

    >>> ds = eoio.read("/path/to/product")

    Read selected variables over a spatial region of interest:

    >>> from shapely.geometry import box
    >>> ds = read_product(
    ...     path="/path/to/product",
    ...     vars_sel={
    ...         "meas": ["B02", "B03", "B04"],
    ...         "mask": ["cloud_mask"],
    ...     },
    ...     subset={
    ...         "roi": box(-2.0, 50.0, 0.0, 52.0),
    ...         "roi_crs_epsg": "EPSG:4326",
    ...     },
    ... )
    """

    if path is None:
        raise TypeError("path must not be None")

    if not os.path.exists(path):
        # Try common product file extensions as a fallback (e.g. user passed
        # un-suffixed product id but the archive file exists with an extension).
        common_exts = [".nc", ".tar.gz", ".SEN3", ".SAFE", ".zip"]
        found_path = None
        for ext in common_exts:
            alt = path + ext
            if os.path.exists(alt):
                found_path = alt
                break

        if found_path:
            path = found_path
        else:
            raise FileNotFoundError(f"File not found: {path}")

    # setup_file uncompresses file if necessary, and cleans up files after run
    with setup_file(path, read_params) as path:
        # Initialise reader
        reader_factory = ReaderFactory()
        reader_cls = reader_factory.get_reader(path)
        reader_obj = reader_cls(path, vars_sel=vars_sel, subset=subset, read_params=read_params)

        # Open dataset
        ds = reader_obj.open()

        # No post-processing requested
        if not processors:
            return ds

        # Normalise context passed to processors
        context = Context({**(subset or {}), "path": path})

        # Run processor pipeline
        pp = ProcessorPipeline(processor_params=processors, context=context)

        ds = pp.run(ds)

        return ds


def _resolve_multi_paths(paths: Union[str, List[str]]) -> List[str]:
    """
    Resolve the ``paths`` argument of :func:`read_multi` to an explicit,
    ordered list of file paths.

    * ``list``/``tuple`` – returned as-is (order preserved).
    * ``str`` pointing at an existing file – single-element list.
    * ``str`` pointing at an existing directory – every entry inside it that
      a registered reader recognises (via :class:`ReaderFactory`), sorted by
      name. Entries no reader recognises are silently skipped.
    * ``str`` – otherwise treated as a glob pattern, resolved and sorted.
    """
    if isinstance(paths, (list, tuple)):
        resolved = list(paths)
    elif isinstance(paths, str):
        if os.path.isdir(paths):
            reader_factory = ReaderFactory()
            resolved = []
            for name in sorted(os.listdir(paths)):
                candidate = os.path.join(paths, name)
                try:
                    reader_factory.get_reader(candidate)
                except ValueError:
                    continue
                resolved.append(candidate)
        elif os.path.isfile(paths):
            resolved = [paths]
        else:
            resolved = sorted(glob.glob(paths))
    else:
        raise TypeError(f"read_multi: paths must be a str or list of str, got {type(paths).__name__}")

    if not resolved:
        raise ValueError(f"read_multi: no files found for paths={paths!r}")

    return resolved


def read_multi(
    paths: Union[str, List[str]],
    vars_sel: Optional[Dict[str, List[str]]] = None,
    subset: Optional[Dict[str, Any]] = None,
    read_params: Optional[Dict[str, Any]] = None,
    processors: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    concat_dim: str = "time",
    concat_coord: Optional[Union[str, List[Any], Callable[[str, xr.Dataset], Any]]] = None,
    concat_processors: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    on_mismatch: str = "error",
    attrs: str = "reconcile",
    coord_attrs: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> xr.Dataset:
    """
    Read several EO product files as one call and concatenate them along a
    new (or existing) dimension.

    Each path is read independently via :func:`read` (with identical
    ``vars_sel``/``subset``/``read_params``/``processors`` applied to every
    file), then the resulting datasets are concatenated along ``concat_dim``.

    Typical uses are an in-situ time series (e.g. one Hypernets file per
    measurement cycle, each already carrying its own ``time`` dimension) or a
    multi-date raster stack (e.g. several Sentinel-2 scenes stacked along a
    new ``time`` dimension).

    :param paths:
        Explicit list of product paths, a directory (all files inside it
        recognised by a registered reader), or a glob pattern. All entries
        are expected to resolve to the same reader type.
    :param vars_sel: See :func:`read`. Applied identically to every file.
    :param subset: See :func:`read`. Applied identically to every file.
    :param read_params: See :func:`read`. Applied identically to every file.
    :param processors:
        Post-processing steps run per file, before concatenation. See
        :func:`read`.
    :param concat_dim:
        Name of the dimension to concatenate along. Default ``"time"``.

        If ``concat_dim`` is already a dimension on every per-file dataset
        (e.g. in-situ readers that carry their own ``time`` dimension), the
        datasets are concatenated directly. Otherwise a new dimension is
        created per file via ``expand_dims``, using the coordinate value
        resolved from ``concat_coord`` (e.g. a multi-date raster stack).
    :param concat_coord:
        How to compute the coordinate value for each file along
        ``concat_dim`` when a new dimension needs to be created:

        * ``str`` – dot-path into ``ds.attrs``, e.g.
          ``"product_metadata.sensing_time"``.
        * ``list`` – explicit values, one per path, same order as the
          resolved ``paths``.
        * ``callable(path, ds) -> value`` – escape hatch, e.g. to parse a
          timestamp out of the filename.
        * ``None`` (default) – if ``concat_dim`` already exists on every
          dataset, used as-is; otherwise falls back to file order as an
          integer index coordinate, with a warning.
    :param concat_processors:
        Post-processing steps run once on the final concatenated dataset.
    :param on_mismatch:
        How to handle datasets whose variable set or per-file shape (other
        than ``concat_dim``) doesn't match — this also covers raster grid
        mismatches (a mismatched x/y shape is just another schema mismatch):

        * ``"error"`` (default) – raise, naming the offending files.
        * ``"skip"`` – drop mismatched files, with a warning.
        * ``"union"`` – outer-join concat; missing values become NaN.
    :param attrs:
        How to combine dataset-level and per-variable attrs across files:

        * ``"reconcile"`` (default) – attrs identical across every file are
          kept; attrs that differ are kept as a per-file list attr (or
          promoted to a coordinate, see ``coord_attrs``) rather than being
          silently dropped.
        * ``"first"`` – cheap legacy behaviour: just keep the first file's
          attrs, discarding the rest (xarray's own ``xr.concat`` default).
    :param coord_attrs:
        Dataset-level attribute names to promote to a coordinate on
        ``concat_dim`` instead of being kept as a list attr — e.g.
        ``coord_attrs=["product_name"]`` gives you a per-file ``product_name``
        coordinate alongside ``time``. Only used when ``attrs="reconcile"``.
    :return:
        The concatenated ``xarray.Dataset``.
    """
    resolved_paths = _resolve_multi_paths(paths)

    datasets = [
        read(path, vars_sel=vars_sel, subset=subset, read_params=read_params, processors=processors)
        for path in resolved_paths
    ]

    ds = _concat_datasets(
        datasets=datasets,
        paths=resolved_paths,
        concat_dim=concat_dim,
        concat_coord=concat_coord,
        on_mismatch=on_mismatch,
        attrs=attrs,
        coord_attrs=coord_attrs,
    )

    steps: list = ds.attrs.get("eoio:processing_steps", [])
    if not isinstance(steps, list):
        steps = [str(steps)]
    steps.append(
        {
            "processor": "read_multi",
            "concat_dim": concat_dim,
            "on_mismatch": on_mismatch,
            "attrs": attrs,
            "n_files": len(resolved_paths),
        }
    )
    ds.attrs["eoio:processing_steps"] = steps

    if concat_processors:
        context = Context({"paths": resolved_paths})
        ds = process(ds, concat_processors, context)

    return ds


# def write(
#     path_original: Union[str, List[str]],
#     correction: Dict[str, Union[float, np.ndarray, int]],
#     write_params: Optional[Dict[str, Any]] = None,
# ) -> None:
#     """
#     writer function
#
#     :param path_original: satellite data product
#     :param correction: dictionary with band names to be corrected as keys, and either corrected data or bias correction per band as values
#     :param write_params: definition of desired writing parameters, by default None
#     """
#     if isinstance(path_original, str) or len(list(path_original)) == 1:
#         path_original = [path_original]
#
#     for fn in path_original:
#         with setup_file(fn, write_params) as path:
#             writer_factory = WriterFactory()
#
#             writer = writer_factory.get_writer(path)
#
#             writer_obj = writer()
#
#             writer_obj.write(path, correction, write_params)
#
#
# def product_bounds(
#     path: str,
#     read_params: Optional[Dict[str, Any]] = None,
#     *args,
#     **kwargs,
# ) -> dict:
#     """
#     Return coordinate bounds of the product.
#
#     Example output:
#         {
#             'EPSG:4326':
#                 [
#                     [108.60281738078888, 41.52685230104365],
#                     [109.91862807140421, 41.54675989693983],
#                     [109.93470404984265, 40.55774811423141],
#                     [108.63842218519015, 40.5385178383516]],
#             'EPSG:32649':
#                 [
#                     [300000.0, 4600020.0],
#                     [409810.0, 4600020.0],
#                     [409810.0, 4490210.0],
#                     [300000.0, 4490210.0]
#                 ]
#         }
#
#     :param path: satellite data product
#     :return: dictionary with coordinate reference system as a key and the corresponding coordinate bounds as values
#     """
#     with setup_file(path, read_params) as path:
#         reader_factory = ReaderFactory()
#
#         reader = reader_factory.get_reader(path)
#
#         reader_obj = reader(path)
#
#         try:
#             return dict(
#                 [(k, list(v.exterior.coords)) for k, v in reader_obj.bounds.items()]
#             )
#         except AttributeError:
#             raise ValueError(
#                 """'product_bounds' cannot be determined from '{}'.
#             Either the bounds of the product cannot be parsed without reading in
#             the full product or the reader is not yet fully configured.""".format(
#                     path
#                 )
#             )
#
#
# def mid_lon_lat(
#     path: str,
#     *args,
#     **kwargs,
# ) -> Tuple[float, float]:
#     """
#     Return mid point of satellite product in as a (lon, lat) coordinate
#
#     :param path: satellite data product
#     :return : tuple of the (lon, lat) coordinate for the centre of the satellite product
#     """
#     bounds = product_bounds(path, *args, **kwargs)["EPSG:4326"]
#
#     lons, lats = [*set([i[0] for i in bounds])], [*set([i[1] for i in bounds])]
#     lon_0, lon_1, lat_0, lat_1 = min(lons), max(lons), min(lats), max(lats)
#     return lon_0 + (lon_1 - lon_0) / 2, lat_0 + (lat_1 - lat_0) / 2


def product_processors(path: str, *args, **kwargs) -> Optional[dict]:
    """
    Return dictionary of available post processors for the requested satellite data product
    and their optional parameters. Return None if none available or the product is not recognised.

    :param path: satellite data product
    :return : dictionary of available post processors and their optional parameters
    """

    processor_info = {}
    for processor_name in PROCESSOR_REGISTRY:
        processor_info[processor_name] = PROCESSOR_REGISTRY[processor_name]._all_options
    return processor_info


def product_options(path: str, read_params: Optional[Dict[str, Any]] = None, *args, **kwargs) -> dict:
    """
    Return dictionary of available `meas_vars` options for the requested EO
    data product

    :param path: satellite data product
    :return : dictionary of available subsetting parameters
    """
    with setup_file(path, read_params) as path:
        reader_factory = ReaderFactory()

        reader = reader_factory.get_reader(path)

        reader_obj = reader(path)

        return reader_obj.all_options


def process(
    ds: xr.Dataset, processors: Union[Dict[str, Any], List[Dict[str, Any]]], context: Optional[Context]
) -> xr.Dataset:
    """
    Runs a user-defined processing pipeline on an xarray Dataset.

    :param ds:
        Input xarray.Dataset to be processed.

        This dataset is passed sequentially through each processor defined
        in ``processors``

    :param processors:
        Definition of post-processing steps to apply after reading
        (e.g. interpolation, unit conversion, etc.).

        Dictionary keys should be the names of the processors to run.
        The associated entry should be a subdictionary of parameters for that processor.
        Required parameters are defined at the processor level.

    :param context:
        Optional processing context (reader info, metadata view, logger, etc.).

    :return:
        ``xarray.Dataset`` The processed dataset after all configured processors have been
        applied successfully
    """

    # Run processor pipeline
    if not context:
        context = {}

    pp = ProcessorPipeline(processor_params=processors, context=context)
    ds = pp.run(ds)
    return ds


if __name__ == "__main__":
    pass
