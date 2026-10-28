.. currentmodule:: eoio

.. _write_func:

##################
Writing a dataset
##################

:func:`eoio.write` writes an ``xarray`` ``Dataset`` or ``DataTree`` to disk. It is the
counterpart to :func:`eoio.read`.

.. contents::
   :depth: 2


Basic usage
===========

::

    import eoio

    ds = eoio.read("/path/to/product")
    eoio.write(ds, "output.nc", writer="netcdf")

``write`` returns the path it wrote, as a string.

.. warning::

   An existing file or store at the path is **overwritten**. The one exception is
   appending to a Zarr store, below.


Choosing the format
===================

The format is chosen explicitly with ``writer``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Writes
   * - ``"netcdf"``
     - A netCDF4 file. A ``DataTree`` becomes netCDF4 groups
   * - ``"zarr"``
     - A Zarr store. A ``DataTree`` becomes Zarr groups. Needs the optional zarr
       package: ``pip install eoio[zarr]``

To see what is available::

    >>> from eoio.writers import WriterFactory
    >>> WriterFactory.list_writers()
    ['netcdf', 'zarr']

.. note::

   The output format is **not** guessed from the file extension. Reading detects the
   format because it is a property of the file being opened; writing is a choice, and
   stating it avoids silently producing something other than what you meant. It also
   does not generalise — a directory-style product such as ``.SAFE`` has no extension to
   inspect.


Write parameters
================

``write_params`` controls how the data are written. Which are accepted depends on the
writer — inspect them with :func:`eoio.writer_options`::

    >>> eoio.writer_options("zarr")["write_params"]["append_dim"]
    'dimension name to append along, or None (not supported for a DataTree)'

Every writer accepts these:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``encode_attrs``
     - ``True``
     - Encode attributes the format cannot store natively — nested dictionaries,
       datetimes, geometries — so they survive the round trip
   * - ``on_nonconformance``
     - ``"warn"``
     - What to do when the dataset does not conform to CF-1.8 / ACDD / the controlled
       vocabulary: ``"warn"``, ``"error"`` or ``"ignore"``
   * - ``update_history``
     - ``True``
     - Record the write in ``history`` and ``eoio:processing_steps``
   * - ``create_parents``
     - ``True``
     - Create missing parent directories of the output path

Format-specific parameters include ``complevel`` and ``engine`` for netCDF, and
``append_dim``, ``chunks`` and ``zarr_format`` for Zarr.


Attributes
==========

Readers put rich Python objects in ``attrs`` — nested metadata dictionaries,
``datetime`` objects, geometries — none of which netCDF or Zarr can store. ``write``
encodes them automatically, so a dataset that reads will write.

Reading a netCDF file back with ``eoio.read`` decodes them for you::

    ds = eoio.read("output.nc", read_params={"metadata_level": "original"})
    ds.attrs["product_metadata"]["datatake"]["id"]

``metadata_level="original"`` returns the file's attributes as they were written. At the
default level, ``"all"``, they are kept under ``ds.attrs["product_metadata"]`` instead,
alongside eoio's standard attributes for the read.

Anywhere else - a file opened with xarray directly, or a Zarr store - decode after
reading::

    from eoio.serialisation import decode_attrs

    ds = decode_attrs(xr.open_dataset("output.nc"))

Nested dictionaries are written as dot-separated attribute names
(``product_metadata.datatake.id``), so they stay readable in ``ncdump`` and any generic
netCDF viewer.


Building up a time series
=========================

The Zarr writer can append along a dimension, which is what you want when observations
arrive one at a time and the stack grows. The same call handles the first write and
every later one — the first creates the store, the rest extend it::

    for scene in scenes:
        eoio.write(
            scene,
            "stack.zarr",
            writer="zarr",
            write_params={"append_dim": "time"},
        )

Only the new data is written; existing chunks are left alone. Memory stays flat however
large the stack gets.

The appended data must have the same variables and the same shape in every dimension
other than ``append_dim``, or the write is refused. Appending a coordinate value the
store already holds, as when re-running a partly failed series, is also refused by
default; pass ``on_overlap="skip"`` to write only what is new. Values are concatenated
in the order written, so sort beforehand if order matters.

.. note::

   Appending is not supported for a ``DataTree`` — ``xarray`` raises
   ``NotImplementedError``.


Compression
===========

netCDF output is compressed by default (zlib, level 4); adjust with ``complevel`` or
turn it off with ``compress=False``. Zarr compresses natively, so there is nothing extra
to configure — pass ``encoding`` to change it.

Variables under 16 KB are left uncompressed for netCDF, since the per-variable overhead
would make them larger. Set ``compress_min_bytes=0`` to compress everything.


Processing before writing
=========================

``processors`` applies a processing pipeline before writing, in the same form
:func:`eoio.read` takes, so read-transform-write is two calls::

    ds = eoio.read("/path/to/product")
    eoio.write(ds, "output.nc", writer="netcdf", processors={"stack": {...}})
