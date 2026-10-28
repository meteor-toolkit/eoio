.. currentmodule:: eoio

.. _add_writers:

#######
Writers
#######

This guide describes how to add a new dataset writer to *eoio*.

.. contents::
   :depth: 3

Overview
========

Writers are the counterpart to readers: a reader turns a product on disk into an
``xarray`` object, a writer turns an ``xarray`` object back into something on disk.

Like readers, writers are **thin orchestrators**. The base class owns everything that is
the same for every format — configuration, capability checking, conformance checking,
provenance, attribute encoding — and the concrete writer owns only the format itself.

The writing pipeline is:

.. code-block:: text

    capability check → conformance check → provenance → attribute encoding → format write

Typical package layout for a writer named ``myformat``::

    eoio/writers/myformat/
        writer.py       # orchestrator - thin
        encoding.py     # optional: format-specific encoding/compression defaults
        tests/          # unit tests

.. note::

   Place the writer package in ``eoio/writers/``. Place test files in a ``tests/``
   sub-package within the writer directory.


Useful API References
=====================

.. autosummary::
   :toctree: ../../api/
   :nosignatures:

   writers.base.BaseWriter
   writers.base.WriteCapability
   writers.base.WriterConfig
   writers.factory.WriterFactory
   writers.factory.register_writer


Selecting a Writer
==================

Writers are selected **by name**, never inferred from the output path::

    from eoio.writers import WriterFactory

    writer_cls = WriterFactory.get_writer("netcdf")
    writer_cls().write(ds, "output.nc")

This is deliberately unlike :class:`~readers.factory.ReaderFactory`, which matches the
product path against regular expressions. On read, the format is a fact about the input
that has to be discovered; on write, it is the user's choice. Inferring it from the
output extension is rejected because it silently guesses, and because it does not extend
to directory-style products such as ``.SAFE``.


Creating a Writer Class
=======================

Required Contract
-----------------

A concrete writer must:

1. Inherit from :class:`~writers.base.BaseWriter`.
2. Implement ``write_dataset()``.
3. Implement the static method ``get_extension()``.
4. Register it with ``@register_writer("name")``, which sets ``name`` for you.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute / Method
     - Purpose
   * - ``name``
     - Identifier the writer is selected under (e.g. ``"netcdf"``). Set by
       ``@register_writer``; only set it by hand for an unregistered writer
   * - ``default_write_params``
     - Default write-time parameters, extending the base defaults
   * - ``all_write_params``
     - Available values for each parameter, surfaced through ``all_options``
   * - ``write_dataset()``
     - Write the object in this format — the only abstract method
   * - ``get_extension()``
     - Return the filename extension produced (e.g. ``".nc"``)
   * - ``can_write()``
     - Optional. Refuse objects the format cannot express
   * - ``check_conformance()``
     - Optional. Report CF/ACDD/vocabulary problems before writing


Public Entrypoint
-----------------

Users call ``writer.write(obj, path)`` — not ``write_dataset()`` directly. ``write()`` is
provided by the base class and performs the shared orchestration before calling
``write_dataset()``.

Unlike a reader, which is constructed around a single product path, a writer instance
holds only configuration and can be reused across many outputs::

    writer = SomeWriter(write_params={"engine": "netcdf4"})
    writer.write(ds_a, "a.nc")
    writer.write(ds_b, "b.nc")

Unknown ``write_params`` keys raise a ``ValueError`` at initialisation, matching the
reader's strict behaviour. Merged configuration is stored in ``writer.config``, a
:class:`~writers.base.WriterConfig` dataclass.


Minimal Example
---------------

::

    from pathlib import Path
    import xarray as xr
    from eoio.writers.base import BaseWriter, WritableObject


    from eoio.writers import register_writer


    @register_writer("myformat")
    class MyWriter(BaseWriter):

        default_write_params = {
            **BaseWriter.default_write_params,
            "compression": "zlib",
        }
        all_write_params = {
            **BaseWriter.all_write_params,
            "compression": "zlib/none",
        }

        @staticmethod
        def get_extension() -> str:
            return ".myformat"

        def write_dataset(self, obj: WritableObject, path: Path) -> None:
            # attrs are already encoded and provenance already applied
            ...


Base Write Parameters
=====================

Every writer inherits these, and should extend rather than replace them:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``encode_attrs``
     - ``True``
     - Encode attributes for serialisation before writing (see :mod:`eoio.serialisation`)
   * - ``on_nonconformance``
     - ``"warn"``
     - What to do when the conformance check reports problems: ``warn``/``error``/``ignore``
   * - ``update_history``
     - ``True``
     - Append this write to ``history`` and ``eoio:processing_steps``
   * - ``create_parents``
     - ``True``
     - Create missing parent directories for the output path

Extend them with ``**BaseWriter.default_write_params`` so the base options keep working.


Declaring Capability
====================

``can_write()`` reports whether the writer can express a given object, and why not::

    def can_write(self, obj):
        base = super().can_write(obj)
        if not base.ok:
            return base
        if obj.attrs.get("instrument") != "MSI":
            return WriteCapability(False, "requires a Sentinel-2 MSI dataset")
        return WriteCapability(True)

General-purpose writers (netCDF, Zarr) accept any ``Dataset`` or ``DataTree`` — anything
expressible in xarray is expressible in those formats — so they simply inherit the
default.

Format-specific writers should refuse anything of the wrong shape. The reason string is
shown to the user, so it should say what is missing, not merely that something is.


Attribute Encoding
==================

Reader output routinely carries attributes that no serialisation format can store:
nested metadata dicts, ``datetime`` objects, geometries. ``write()`` runs
:func:`eoio.serialisation.encode_attrs` before calling ``write_dataset()``, so writers do
not have to think about it.

Writers should **not** encode attributes themselves. If a format needs different
treatment, override ``write_dataset()`` and set ``encode_attrs=False`` in the write
params rather than encoding twice.


Provenance
==========

``write()`` appends to two attributes before writing, so they appear in the written file:

- ``history`` — a CF-style line recording the output filename, writer and eoio version
- ``eoio:processing_steps`` — a step entry in the same shape ``read_multi`` uses,
  so a dataset's processing history reads consistently from either end

The input object is never modified; ``write()`` works on a copy.


Registering a Writer
====================

Built-in writers are listed in ``eoio/writers/factory.py`` as lazy ``(module, class)``
entries, so an optional backend is only imported when that writer is used::

    _BUILTIN_WRITERS = {
        "netcdf": ("eoio.writers.netcdf.writer", "NetCDFWriter"),
    }

Writers defined outside that table register with ``register_writer``, a decorator in the
same style as ``@register_processor``::

    from eoio.writers import register_writer

    @register_writer("myformat")
    class MyWriter(BaseWriter):
        ...

It can also be called directly, which is what you want for a class you do not own::

    register_writer("geotiff", SomeThirdPartyWriter)

Either form sets the class's ``name``, so a registered writer always knows what it is
called — which is what ends up in the provenance a write records.


Testing
=======

Writer tests should not need real product files — construct a small ``Dataset`` in the
test and write it to ``tmp_path``. See ``eoio/writers/tests/test_base.py``, which uses a
dummy writer to exercise the shared orchestration without touching a real format.

Run ``ruff check eoio`` and ``pytest`` before committing.
