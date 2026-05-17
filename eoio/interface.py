"""eoio.interface - interface functions module"""

from typing import Any, Dict, Optional, List
import xarray as xr
import os
from processor_tools import Context
from eoio.processors.processor_pipeline import ProcessorPipeline
from eoio.processors.registry import PROCESSOR_REGISTRY
from eoio.readers.factory import ReaderFactory
from eoio.utils.read_utils import setup_file

__all__ = [
    "read",
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
    processors: Optional[Dict[str, Any]] = None,
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


def process(ds: xr.Dataset, processors: Dict[str, Any], context: Optional[Context]) -> xr.Dataset:
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
