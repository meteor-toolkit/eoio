Reading Multiple Products at Once
++++++++++++++++++++++++++++++++++

Using the read_multi function
=============================

**eoio** can read several product files in one call and concatenate the resulting
datasets along a new (or existing) dimension via the ``read_multi()`` function. This
covers two common cases:

* **In-situ time series** – e.g. Hypernets acquires one file per measurement cycle
  (~every 10 minutes); ``read_multi()`` returns "all spectra from today" as one
  dataset with a ``time`` dimension, rather than looping over files and merging
  them by hand.
* **Multi-date raster stacks** – e.g. several Sentinel-2/Landsat scenes over the
  same region of interest, read and concatenated into a single ``(time, y, x)``
  cube per variable.

Both are the same operation at the dataset level: read N products (each via the
usual :func:`~eoio.read`), then concatenate matching variables across the N
resulting datasets along a new dimension.

The function can be used as follows::

    ds = read_multi(
        paths,
        vars_sel: Optional[dict] = None,
        subset: Optional[dict] = None,
        read_params: Optional[dict] = None,
        processors: Optional[dict] = None,
        concat_dim: str = "time",
        concat_coord: Optional[Union[str, list, Callable]] = None,
        concat_processors: Optional[dict] = None,
        on_mismatch: str = "error",
        attrs: str = "reconcile",
        coord_attrs: Optional[list] = None,
    )

where only ``paths`` is required.

Input Parameters
-----------------

Paths
^^^^^
``paths`` accepts:

* an explicit list of product paths,
* a directory – every entry inside it that a registered reader recognises is
  included, sorted by name,
* or a glob pattern, e.g. ``"day_2026-07-08/*.nc"``.

All entries are expected to resolve to the same reader type.

Per-file Read Parameters
^^^^^^^^^^^^^^^^^^^^^^^^
``vars_sel``, ``subset``, and ``read_params`` have the same meaning as in
:func:`~eoio.read` and are applied identically to every file (per-file overrides
are not supported). ``processors`` are also applied per file, before
concatenation — this is how a per-scene :ref:`stack processor <add_processor>`
band cube is built prior to stacking scenes along time (see the raster example
below).

Concatenation
^^^^^^^^^^^^^
* ``concat_dim`` – name of the dimension to concatenate along. Default
  ``"time"``.

  If ``concat_dim`` is already a dimension on every per-file dataset (the
  in-situ case, where each file already carries its own ``time`` dimension),
  the datasets are concatenated directly. Otherwise a new dimension is created
  per file (the raster case), using the coordinate value resolved from
  ``concat_coord``.

* ``concat_coord`` – how to compute the coordinate value for each file when a
  new dimension needs to be created:

  - ``str`` – dot-path into ``ds.attrs``, e.g. ``"product_metadata.sensing_time"``.
  - ``list`` – explicit values, one per path, same order as the resolved ``paths``.
  - ``callable(path, ds) -> value`` – escape hatch, e.g. to parse a timestamp out
    of the filename.
  - ``None`` (default) – if ``concat_dim`` already exists on every dataset, used
    as-is; otherwise falls back to file order as an integer index coordinate,
    with a warning.

* ``concat_processors`` – post-processing steps run once on the final
  concatenated dataset (e.g. re-sorting by time), in the same format as
  ``processors``.

* ``on_mismatch`` – how to handle datasets whose variable set or per-file shape
  (other than ``concat_dim``) doesn't match — this also covers raster grid
  mismatches, since a mismatched x/y shape is just another schema mismatch:

  - ``"error"`` (default) – raise, naming the offending files.
  - ``"skip"`` – drop mismatched files, with a warning.
  - ``"union"`` – outer-join concat; missing values become NaN.

Variables with a Repeated Dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some EO products legitimately store a variable with the same dimension twice —
e.g. a ``(measurement, measurement)`` error-correlation matrix. ``xr.concat``
itself cannot handle this (it raises ``ValueError: This function cannot handle
duplicate dimensions``), even though such a variable reads fine from a single
file. ``read_multi`` detects this automatically, concatenates correctly, and
restores the original repeated dimension name in the result — no configuration
needed.

Attribute Handling
^^^^^^^^^^^^^^^^^^^
Each input file contributes its own dataset-level (``ds.attrs``) and per-variable
(``da.attrs``) metadata (e.g. ``product_name``, ``sensing_time``, ``scale_factor``).
``attrs`` controls how these are combined across files:

* ``attrs="reconcile"`` (default) – an attribute identical across every
  contributing file is kept as a scalar; an attribute that differs is kept as a
  list attr, one entry per file, in file order — nothing is silently dropped.
  This mirrors how the :ref:`stack processor <add_processor>` already reconciles
  per-band attrs when building a cube.
* ``attrs="first"`` – cheap legacy behaviour: keep only the first file's attrs,
  discarding the rest (this is xarray's own ``xr.concat`` default).

``coord_attrs`` (only used with ``attrs="reconcile"``) names dataset-level
attributes that should be promoted to a real coordinate on ``concat_dim``
instead of being kept as a list attr — e.g. ``coord_attrs=["product_name"]``
gives you a per-file ``product_name`` coordinate alongside ``time``. If a file
contributes more than one ``concat_dim`` entry (e.g. an in-situ file with
several timestamps), that file's value is repeated across all of its entries.

``eoio:processing_steps`` (the processing-history attribute every eoio reader
and processor writes) is handled specially regardless of ``attrs``: each
file's history is merged by concatenation, in file order, rather than kept
as a list-of-lists or dropped — so the full provenance chain survives.

Example — Combining Metadata Across Scenes
--------------------------------------------
::

    ds = read_multi(
        paths=[p1, p2, p3],
        concat_coord="product_metadata.sensing_time",
        coord_attrs=["product_metadata.tile_id"],  # -> becomes a coordinate
    )
    # ds.attrs["cloud_cover"] == [12.3, 45.6, 3.1]  # differs per scene -> list attr
    # ds.coords["tile_id"] == ["T30UWC", "T30UWC", "T30UWC"]  # promoted coordinate

Example — In-situ Time Series
------------------------------
Reading all Hypernets spectra acquired on a given day, where each file already
carries its own ``time`` dimension::

    from eoio import read_multi

    ds = read_multi(
        paths="HYPERNETS_L1B_.../2026-07-08/*.nc",
        vars_sel={"meas": "all"},
        # concat_dim="time" is the default and already exists per file
    )

Example — Multi-date Raster Stack
-----------------------------------
Reading ten Sentinel-2 scenes over the same region of interest and stacking them
into a ``(time, band, y, x)`` cube::

    from eoio import read_multi

    ds = read_multi(
        paths=[p1, p2, ..., p10],
        vars_sel={"meas": "all"},
        subset={"roi": geom, "roi_crs_epsg": "EPSG:4326"},
        processors={"stack": {"stack_dim": "band"}},  # per-scene band cube
        concat_dim="time",
        concat_coord="product_metadata.sensing_time",  # per-file attr -> time coord
    )
    # -> reflectance_10m(time, band_10m, y_10m, x_10m)

There are also example scripts available in the ``examples`` directory of the
repository which illustrate the use of ``read_multi`` for different satellite
products.
