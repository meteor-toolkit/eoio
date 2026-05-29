.. currentmodule:: eoio

.. _add_readers:

#######
Readers
#######

This guide describes how to add a new EO product reader to *eoio*.

.. contents::
   :depth: 3

Overview
========

Readers in *eoio* follow a **thin orchestrator** design. Each reader is responsible for
parsing user configuration and coordinating specialised helper modules — it does not
contain heavy IO, geometry, or parsing logic itself. That logic lives in small,
single-responsibility modules alongside the reader.

The reading pipeline is:

.. code-block:: text

    layout → subset resolution → image IO → optional enrichment → conventions

Typical package layout for a reader named ``myreader``::

    eoio/readers/myreader/
        reader.py       # orchestrator – thin, imports helper modules
        layout.py       # product structure and file discovery
        data_io.py      # raster/data reading and subsetting
        subset.py       # optional: ROI or other subset resolution
        aux_data.py     # auxiliary variable ingestion
        conventions.py  # standardise variable names, attrs, provenance
        metadata/       # product metadata parsing (XML, JSON, etc.)
        tests/          # unit tests for each module

Not all modules are required for every reader. Point/in-situ readers, for example,
typically omit ``layout.py`` and ``data_io.py`` and use a simpler metadata approach.

.. note::

   Place the reader package in ``eoio/readers/``. Place test files in a ``tests/``
   sub-package within the reader directory. See existing readers
   (e.g. ``eoio/readers/sentinel2/``, ``eoio/readers/hypernets/``) as reference
   implementations.


Useful API References
=====================

.. autosummary::
   :toctree: ../../api/
   :nosignatures:

   readers.factory.ReaderFactory
   readers.base.BaseReader
   readers.base.BaseRasterReader
   readers.base.BaseInSituReader


Base Classes
============

All readers inherit from one of the base classes in ``eoio.readers.base``. Choose the
most appropriate starting point:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - When to use
   * - ``BaseRasterReader``
     - Gridded satellite imagery (Sentinel-2, Landsat, OLCI, SLSTR, PlanetScope, …)
   * - ``BaseInSituReader``
     - Point / time-series data (Hypernets, RadCalNet, …)
   * - ``BaseReader``
     - Any reader that does not fit the above (e.g. ERA5, generic NetCDF)

The base classes are intentionally small. They own:

- Path validation
- Configuration merging and validation (via :class:`~readers.base.ReaderConfig`)
- The single public entrypoint :meth:`~readers.base.BaseReader.open`

Everything else is the responsibility of the concrete reader.


Creating a Reader Class
=======================

Required Contract
-----------------

A concrete reader must:

1. Inherit from an appropriate base class.
2. Override the class-level configuration dictionaries.
3. Implement ``open_dataset()``.
4. Implement the static method ``get_extension()``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute / Method
     - Purpose
   * - ``default_vars_sel``
     - Default values for variable selection (``meas``, ``aux``, ``mask``)
   * - ``default_subset``
     - Default subsetting parameters (e.g. ``roi``, ``roi_crs``)
   * - ``default_read_params``
     - Default read-time parameters (e.g. ``metadata_level``)
   * - ``meas_def``
     - Mapping of preset names (``"all"``, ``"rgb"``, …) to lists of variable names
   * - ``aux_def``
     - Mapping of preset names to lists of auxiliary variable names
   * - ``mask_def``
     - Mapping of preset names to lists of mask variable names
   * - ``open_dataset()``
     - Read the product and return an ``xarray.Dataset``
   * - ``get_extension()``
     - Return the file extension string for the reader (e.g. ``".SAFE"``)

Public Entrypoint
-----------------

Users call ``reader.open()`` — not ``open_dataset()`` directly. ``open()`` is
provided by the base class and calls ``open_dataset()`` internally::

    reader = MyReader(path, vars_sel={"meas": "all"}, subset={"roi": [...]})
    ds = reader.open()

Configuration is passed via three optional dictionaries:

- ``vars_sel`` — which measurement, auxiliary, and mask variables to include
- ``subset`` — spatial, temporal, or spectral subsetting
- ``read_params`` — read-time options such as ``metadata_level``

Unknown keys raise a ``ValueError`` at initialisation time. Validated, merged
configuration is stored in ``reader.config`` (a :class:`~readers.base.ReaderConfig`
dataclass) and resolved configuration in ``reader.resolved_config``.

Minimal Example
---------------

::

    from pathlib import Path
    from typing import Any, Dict, List, Optional
    import xarray as xr
    from eoio.readers.base import BaseRasterReader


    class MyReader(BaseRasterReader):

        default_read_params = {
            "save_extracted": False,
            "metadata_level": "all",
            "include_uncertainties": False,
        }

        meas_def: Dict[str, List[str]] = {
            "all": ["band1", "band2", "band3"],
            "rgb": ["band1", "band2", "band3"],
        }
        aux_def: Dict[str, List[str]] = {"all": []}
        mask_def: Dict[str, List[str]] = {"all": []}

        @staticmethod
        def get_extension() -> str:
            return ".myformat"

        def open_dataset(self) -> xr.Dataset:
            ds = xr.Dataset()
            # ... read data using helper modules ...
            return ds


Variable Selection
------------------

Variable selection is controlled by the ``vars_sel`` argument. The ``meas_def``,
``aux_def``, and ``mask_def`` class attributes define which names are valid.

Keys in each ``*_def`` dict are preset names. The value under ``"all"`` must list
every available variable of that type. Additional keys (e.g. ``"rgb"``, ``"basic"``)
are optional subsets::

    meas_def = {
        "all": ["B01", "B02", "B03", ..., "B12"],
        "rgb": ["B02", "B03", "B04"],
    }

Users can then request ``vars_sel={"meas": "rgb"}`` or ``vars_sel={"meas": ["B02", "B08"]}``.
``None`` (the default) returns an empty list. The base class resolves these into concrete
lists stored in ``reader.resolved_config.vars_sel``.


Subsetting
----------

Subsetting parameters depend on the base class:

``BaseRasterReader`` default subset keys:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``roi``
     - Region of interest as bounding-box coordinates or polygon
   * - ``roi_crs``
     - CRS of the ROI (default ``4326``)
   * - ``angle``
     - Angle filter (min/max/nearest/tolerance)
   * - ``wavelength``
     - Wavelength filter (min/max/nearest/tolerance)

``BaseInSituReader`` default subset keys:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``wavelength``
     - Wavelength filter
   * - ``datetime``
     - Datetime filter (min/max/nearest/tolerance_days/hours/minutes)
   * - ``angle``
     - Angle filter

Override ``resolve_subset()`` to convert the raw subset dict into an internal
representation. For raster readers that accept an ROI, use
:class:`~readers.subset.roi_subset.ROISubsetResolver`::

    from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset

    def resolve_subset(self, subset):
        if not subset or subset.get("roi") is None:
            return None
        return ROISubsetResolver(
            self.layout, subset["roi"], subset.get("roi_crs", 4326)
        ).resolve()


Helper Modules
--------------

Move complex logic out of the reader class into focused modules. Common patterns:

``layout.py``
    Product structure and file discovery. Knows how to find band images, metadata
    XMLs, and auxiliary files given a product path. Should not perform any IO beyond
    path resolution.

``data_io.py``
    Raster reading and spatial subsetting. Uses lazy imports from :mod:`eoio.deps`
    for ``rasterio`` and ``rioxarray``.

``subset.py``
    Reader-specific ROI or temporal subset resolution. May use ``eoio.readers.subset``
    utilities or implement its own logic.

``aux_data.py``
    Auxiliary variable ingestion (e.g. meteorological grids, ECMWF/CAMS data).

``angles.py``
    Angle grid parsing (e.g. solar/viewing angles from XML).

``conventions.py``
    Apply controlled-vocabulary variable names, attributes, and provenance stamps
    to the assembled ``xarray.Dataset``. See :ref:`controlled_vocabulary` for the
    expected attribute values.

``metadata/``
    XML or JSON metadata parsers. Keep each metadata document type in its own
    module (e.g. ``s2_prod_mtd.py``, ``s2_tl_mtd.py``).


Lazy Optional Dependencies
--------------------------

Heavy dependencies are imported lazily to keep *eoio* lightweight. Use the helpers
in :mod:`eoio.deps` instead of top-level imports::

    from eoio.deps import lazy_rasterio, lazy_rioxarray, lazy_pyproj, lazy_shapely

    def _read_band(path):
        rasterio = lazy_rasterio()
        with rasterio.open(path) as src:
            ...

Available helpers:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Function
     - Install extra
     - Provides
   * - ``lazy_rasterio()``
     - ``eoio[raster]``
     - ``rasterio``
   * - ``lazy_rioxarray()``
     - ``eoio[raster]``
     - ``rioxarray``
   * - ``lazy_pyproj()``
     - ``eoio[geo]``
     - ``pyproj``
   * - ``lazy_shapely()``
     - ``eoio[geo]``
     - ``shapely`` geometry helpers


Output Dataset Requirements
============================

The ``xarray.Dataset`` returned by ``open_dataset()`` must follow the eoio
controlled vocabulary. See :ref:`controlled_vocabulary` for the full specification.
Key requirements are summarised here.

Required Global Attributes
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Attribute
     - Description
     - Example
   * - ``Conventions``
     - CF version string
     - ``"CF-1.8"``
   * - ``title``
     - Human-readable dataset title
     - ``"Sentinel-2A MSI L1C"``
   * - ``institution``
     - Producing institution
     - ``"NPL"``
   * - ``source``
     - Upstream product name
     - ``"S2A_MSIL1C_20230101…SAFE"``
   * - ``history``
     - Audit trail (append, do not overwrite)
     - ``"2025-01-01: read by eoio 1.0"``
   * - ``platform``
     - Controlled platform token
     - ``"Sentinel-2A"``
   * - ``instrument``
     - Controlled instrument token
     - ``"MSI"``
   * - ``processing_level``
     - Controlled level token
     - ``"L1C"``
   * - ``product_name``
     - Full upstream product identifier
     - ``"S2A_MSIL1C_20230101…SAFE"``
   * - ``collection_name``
     - Stable collection identifier
     - ``"S2MSI1C"``

Required Variable Attributes
-----------------------------

Each measurement, auxiliary, and mask variable should carry:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``standard_name``
     - CF standard name where available
   * - ``long_name``
     - Human-readable description
   * - ``units``
     - UDUNITS-compliant unit string

Variable-specific metadata should be stored in the ``DataArray.attrs`` dict of
the variable, not in global dataset attributes.

Flag Variables (Masks)
-----------------------

All mask variables must be stored as CF-convention flag variables using
`obsarray <https://github.com/comet-toolkit/obsarray>`_::

    ds.flag["quality_flags"] = (["y", "x"], {"flag_meanings": ["cloud", "land"]})
    ds.flag["quality_flags"]["cloud"][:, :] = cloud_mask

For products where masks arrive as packed bit fields, assign the raw array and
set ``flag_meanings`` and ``flag_masks`` attributes directly::

    ds["quality_flags"] = (("y", "x"), packed_flags)
    ds.quality_flags.attrs = {
        "flag_meanings": "cloud land shadow",
        "flag_masks": "1,2,4",
    }

Dimension Naming
----------------

Follow the dimension naming rules in :ref:`controlled_vocabulary`. For multi-resolution
raster datasets use the ``x_<resolution>`` / ``y_<resolution>`` pattern
(e.g. ``x_10m``, ``y_10m``, ``x_60m``, ``y_60m``).


.. _add_readers.integrate:

Registering a Reader
====================

Once your reader class exists, register it in
``eoio.readers.factory.ReaderFactory.get_reader``. Add a regular expression that
uniquely matches your product path and a lazy import of your reader class::

    my_pattern = re.compile(r"MY_SENSOR_.*\Z")

    ...

    elif re.search(my_pattern, path):
        from eoio.readers.myreader.reader import MyReader
        return MyReader

Lazy imports inside the ``elif`` branches are intentional — they avoid importing
heavy optional dependencies at package import time.

The order of patterns matters: place more specific patterns before broad catch-all
patterns (e.g. the generic ``.nc`` pattern must come last).
