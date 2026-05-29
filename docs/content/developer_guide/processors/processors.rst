.. currentmodule:: eoio

.. _add_processor:

##########
Processors
##########

This guide describes how to add a new processor to *eoio*.

.. contents::
   :depth: 3

Overview
========

A processor in *eoio* is a **dataset transform**: it receives an
:class:`xarray.Dataset` and returns an :class:`xarray.Dataset`. Processors are
applied after reading and run in the order specified by the user.

Design principles:

- **Processors are stateless transforms.** A processor must not depend on reader
  internals beyond what is explicitly provided via the dataset and context.
- **Per-call configuration.** All configuration is supplied at call time via a
  ``params`` dict passed to ``__init__``.
- **Explicit ordering.** Processors run in the order the user specifies. There is
  no automatic reordering or dependency resolution.
- **Minimal framework.** Registration, instantiation, and execution are
  straightforward and easy to follow.


Useful API References
=====================

.. autosummary::
   :toctree: ../../api/
   :nosignatures:

   processors.registry.register_processor
   processors.processor_pipeline.ProcessorPipeline


Creating a Processor
====================

All *eoio* processors subclass ``BaseProcessor`` from the ``processor_tools``
package and are registered with the ``@register_processor`` decorator.

Required Contract
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Requirement
   * - ``__init__(self, params, context)``
     - Parse and validate the ``params`` dict; store what is needed; call
       ``super().__init__(context=context)``.
   * - ``run(self, ds)``
     - Core transform method. Receives the current dataset; returns the
       transformed dataset.
   * - ``@register_processor("name")``
     - Decorator that registers the class under a stable string key.

``run`` signature::

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        ...
        return ds

Context
-------

Processors are instantiated with a ``context`` object (a ``processor_tools.Context``
or plain dict) that carries read-only information supplied by *eoio*:

- Input product path
- Reader name or type
- Resolved reader configuration (variable selection, subset)
- Selected measurement variables
- Product metadata (where available)
- Any user-supplied ``processor_context`` values

Access context values inside ``run`` via ``self.context``::

    reader_name = self.context.get("reader_name")


Minimal Example
---------------

::

    from __future__ import annotations
    from typing import Any, Dict, Optional
    import xarray as xr
    from processor_tools import BaseProcessor
    from eoio.processors.registry import register_processor


    @register_processor("my.processor")
    class MyProcessor(BaseProcessor):
        """Scale measurement variables by a constant factor."""

        def __init__(
            self,
            params: Optional[Dict[str, Any]] = None,
            context: Optional[Dict[str, Any]] = None,
        ):
            super().__init__(context=context)
            params = params or {}
            self.scale = params.get("scale", 1.0)
            self.var_names = params.get("var_names")

        def run(self, ds: xr.Dataset) -> xr.Dataset:
            names = self.var_names or list(ds.data_vars)
            for name in names:
                if name in ds:
                    ds[name] = ds[name] * self.scale
            return ds

The string passed to ``@register_processor`` is the key users supply when calling
:func:`eoio.read`::

    ds = read(path, processors={"my.processor": {"scale": 0.0001}})


Error Handling
--------------

The pipeline supports per-processor error handling via an ``on_missing`` parameter
in the processor params dict:

- ``"error"`` (default) — stop the pipeline and raise a
  :class:`~processors.processor_pipeline.ProcessorPipelineError`.
- ``"skip"`` — log a warning and continue with the unmodified dataset.

Example::

    processors = {
        "units.convert": {"to": "reflectance", "on_missing": "skip"},
    }


Registration
============

Built-in Processors
-------------------

Built-in *eoio* processors are imported in ``eoio/processors/__init__.py`` so
they are registered automatically when the package is imported. If you are adding
a new built-in processor, add an import there::

    # eoio/processors/__init__.py
    from eoio.processors.mypackage import processor  # noqa: F401 – triggers @register_processor

User-Defined Processors
-----------------------

Users can define and register processors in their own code. The decorator runs at
class-definition time, so importing the module is sufficient to register it::

    # my_project/my_processor.py
    import eoio  # ensure eoio built-ins are registered first
    from processor_tools import BaseProcessor
    from eoio.processors.registry import register_processor

    @register_processor("myproject.normalise")
    class NormaliseProcessor(BaseProcessor):
        def __init__(self, params=None, context=None):
            super().__init__(context=context)
            ...

        def run(self, ds):
            ...
            return ds

Then in user code::

    import my_project.my_processor  # registers the processor
    from eoio.interface import read

    ds = read(path, processors={"myproject.normalise": {"method": "minmax"}})

Processor Naming Convention
---------------------------

Use dot-separated namespaces to avoid collisions between built-in and
user-defined processors:

- Built-in processors: ``"units.convert"``, ``"interpolate.wavelength"``, …
- Project-specific processors: ``"<project>.<name>"``, e.g. ``"mypro.normalise"``


ProcessorPipeline
=================

:class:`~processors.processor_pipeline.ProcessorPipeline` is the internal
orchestrator that *eoio* uses to run processors. You do not normally need to use
it directly, but it can be useful for testing::

    from eoio.processors.processor_pipeline import ProcessorPipeline

    pipeline = ProcessorPipeline(
        processor_params={
            "units.convert": {"to": "reflectance"},
            "my.processor": {"scale": 0.01, "on_missing": "skip"},
        },
        context={"reader_name": "S2MSIReader", "path": "/data/S2...SAFE"},
    )
    processed_ds = pipeline.run(ds)

The pipeline:

1. Validates that all requested processor names exist in the registry.
2. Instantiates each processor with its params dict and the shared context.
3. Executes processors sequentially, passing the output of each as input to the next.
