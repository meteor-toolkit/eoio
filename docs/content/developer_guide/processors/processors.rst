.. currentmodule:: eoio

.. _add_processor:

###########
Processors
###########

In this guide, you will find detailed descriptions and
examples to describe how to add a processor to *eoio*.

Useful API References
=====================

.. autosummary::
   :toctree: ../../api/
   :nosignatures:

    processors.base

.. note::

  ..  When adding a processor it is crucial for it to be included in
  ..  :py:class:`eoio.processors.factory.ProcessorFactory` ``.'your_sensor'_processors`` dictionary for it to be used by the ``read``
  ..  function. Currently processors are added to the dictionary in the order that they should be run if multiple
  ..  processors are selected. An example for Sentinel-2 processors::

    S2_processors = {
        S2RUT.name: S2RUT,
        S2Converter.name: S2Converter,
        Interpolate.name: Interpolate,
    }

   If all three processors are called in this instance, first the uncertainties would be calculated and added
   to the dataset, followed by any conversions requested and then interpolation would be run.


Creating a Processor
====================

Processors all inherit from :py:class:`eoio.processors.base.BaseProcessor` and can make use of methods
within the class.

Required
++++++++
Processors all inherit from :py:class:`eoio.processors.base.BaseProcessor` and must therefore
contain the following methods, properties and attributes:

.. list-table::
   :stub-columns: 1
   :widths: 10 30 30 30
   :header-rows: 1

   * -
     -
     - Docstring
     - Extra Info
   * - @abstractmethods
     -
     -
     -
   * -
     - ``process_dataset(self) -> xr.Dataset``
     - ``Process data and metadata and return an xarray.Dataset``
     -
   * - @properties
     -
     -
     -
   * -
     - ``all_options``
     - ``Return a dictionary (or list) of all acceptable product processing parameters``
     -
   * - class attributes
     -
     -
     -
   * -
     - ``name``
     - ``str``
     - Name of processor, e.g. convert, angles etc. also to be used by users when requesting this processor.
   * -
     - ``_default_process_dict: Optional[Dict[str, Any]]``
     - ````
     - Default definition of desired processing parameters

.. note::

   Inputs to the processor include the ``xr.Dataset`` returned from ``open_dataset``,
   any ``process_params`` and ``subset_info`` which also contains the product filepath
   (which was added due to a requirement by certain processors).