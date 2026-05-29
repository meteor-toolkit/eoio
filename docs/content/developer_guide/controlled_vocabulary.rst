.. currentmodule:: eoio

.. _controlled_vocabulary:

####################
Controlled Vocabulary
####################

This page defines the controlled vocabulary for *eoio* output datasets. All readers
must produce datasets that conform to this specification to ensure consistency across
missions and products, and to maximise compliance with community standards.

.. contents::
   :depth: 3

Standards Alignment
===================

*eoio* datasets are aligned with two established conventions:

`CF Conventions <https://cfconventions.org/>`_ (CF-1.8)
    Used for coordinate systems, variable naming attributes, units (UDUNITS-2),
    grid mapping variables, and core metadata attributes.

`ACDD <https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3>`_ (Attribute Convention for Data Discovery)
    Used for dataset discovery metadata: platform, instrument, processing level,
    and related discovery fields.

All *eoio* output datasets must declare::

    ds.attrs["Conventions"] = "CF-1.8"


Dataset (Global) Attributes
============================

CF / ACDD Required Attributes
------------------------------

The following attributes are required by CF or ACDD and must be present in
every *eoio* output dataset.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Attribute
     - Description
     - Example
   * - ``Conventions``
     - CF version string
     - ``"CF-1.8"``
   * - ``title``
     - Short human-readable dataset title
     - ``"Sentinel-2A MSI Level-1C"``
   * - ``institution``
     - Organisation producing the dataset
     - ``"NPL"``
   * - ``source``
     - Upstream product filename or identifier
     - ``"S2A_MSIL1C_20230702…SAFE"``
   * - ``history``
     - Audit trail of modifications; append, never overwrite
     - ``"2025-01-01T10:00: read by eoio 1.2"``
   * - ``references``
     - References to product documentation
     - ``"https://sentinel.esa.int/…"``
   * - ``platform``
     - Platform canonical token (see :ref:`cv.platform`)
     - ``"Sentinel-2A"``
   * - ``instrument``
     - Instrument canonical token (see :ref:`cv.instrument`)
     - ``"MSI"``
   * - ``processing_level``
     - Processing level token (see :ref:`cv.level`)
     - ``"L1C"``

*eoio*-Specific Required Attributes
-------------------------------------

In addition to the CF / ACDD attributes, the following *eoio*-specific attributes
are required.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Attribute
     - Description
     - Example
   * - ``product_name``
     - Full upstream product identifier
     - ``"S2A_MSIL1C_20230702…SAFE"``
   * - ``collection_name``
     - Stable collection identifier
     - ``"S2MSI1C"``
   * - ``product_version``
     - Product or algorithm version string
     - ``"03.01"``
   * - ``product_level``
     - Normalised processing level token (same controlled values as ``processing_level``)
     - ``"L1C"``


.. _cv.platform:

Controlled Values — Platform
=============================

The ``platform`` attribute must use one of the following canonical tokens.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Token
     - Notes
   * - ``Sentinel-2A``
     -
   * - ``Sentinel-2B``
     -
   * - ``Sentinel-3A``
     -
   * - ``Sentinel-3B``
     -
   * - ``Landsat-8``
     -
   * - ``Landsat-9``
     -
   * - ``Meteosat-2``
     -
   * - ``Meteosat-7``
     -
   * - ``MSG-1``
     -
   * - ``MSG-2``
     -
   * - ``PlanetScope``
     -

If the platform you are adding is not listed here, add it to this table as part of
your pull request.


.. _cv.instrument:

Controlled Values — Instrument
================================

The ``instrument`` attribute must use one of the following canonical tokens.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Token
     - Notes
   * - ``MSI``
     - Multispectral Instrument (Sentinel-2)
   * - ``OLCI``
     - Ocean and Land Colour Instrument (Sentinel-3)
   * - ``SLSTR``
     - Sea and Land Surface Temperature Radiometer (Sentinel-3)
   * - ``OLI_TIRS``
     - Operational Land Imager + Thermal Infrared Sensor (Landsat 8/9)
   * - ``MVIRI``
     - Meteosat Visible and Infrared Imager
   * - ``SEVIRI``
     - Spinning Enhanced Visible and Infrared Imager
   * - ``SuperDove``
     - PlanetScope 8-band instrument


.. _cv.level:

Controlled Values — Processing Level
======================================

The ``processing_level`` (and ``product_level``) attribute must use one of these tokens.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Token
     - Meaning
   * - ``L0``
     - Raw instrument data
   * - ``L1B``
     - Calibrated, geolocated radiances
   * - ``L1C``
     - Geometrically corrected, top-of-atmosphere reflectance
   * - ``L2A``
     - Atmospherically corrected surface reflectance (sensor-specific algorithm)
   * - ``L2``
     - Surface-level product (generic)
   * - ``L3``
     - Gridded / composited product


Variable Attributes
===================

CF-Aligned Variable Attributes
--------------------------------

Each measurement, auxiliary, and mask variable should carry the following
CF-compliant attributes where applicable.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``standard_name``
     - CF Standard Name Table entry where one exists. Case-sensitive, no whitespace.
   * - ``long_name``
     - Human-readable description, used for plot labels and documentation.
   * - ``units``
     - UDUNITS-2 compliant string. A variable with a ``standard_name`` must have
       physically equivalent units.
   * - ``coordinates``
     - Space-separated list of coordinate variables associated with the variable.
   * - ``grid_mapping``
     - Name of the grid-mapping variable (e.g. ``"crs"``).
   * - ``ancillary_variables``
     - Space-separated list of associated ancillary variables (e.g. uncertainty).
   * - ``comment``
     - Optional free-text note.

See the `CF Conventions documentation <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html>`_
for complete descriptions of acceptable values.

*eoio*-Specific Variable Attributes
-------------------------------------

The following additional attributes are used by *eoio* for classification and
interoperability.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Attribute
     - Description
     - Allowed values
   * - ``measurand``
     - Normalised measurand classification
     - ``toa_radiance``, ``toa_reflectance``, ``surface_reflectance``,
       ``brightness_temperature``, ``digital_number``, ``aod``, ``tcwv``,
       ``tco3``, ``wind_speed``, ``wind_vector``
   * - ``spatial_resolution``
     - Nominal spatial resolution; pattern ``<integer>m``
     - e.g. ``"10m"``, ``"300m"``, ``"5000m"``
   * - ``geometry``
     - Variable geometry classification
     - ``image_grid``, ``angle_grid``, ``aux_grid``, ``point``


Dimension Naming
================

Canonical Dimensions
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Dimension name
     - When to use
   * - ``time``
     - Temporal dimension
   * - ``lat`` / ``lon``
     - Only when coordinates are true 1-D geographic arrays
   * - ``x`` / ``y``
     - Projected grid dimensions; coordinate variables must carry CF
       ``standard_name`` of ``projection_x_coordinate`` / ``projection_y_coordinate``
   * - ``band``
     - Spectral band index dimension

Multi-Resolution Grids
-----------------------

Where a single dataset contains variables at multiple spatial resolutions,
use the pattern ``x_<resolution>`` / ``y_<resolution>``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Examples
     - Notes
   * - ``x_10m``, ``y_10m``
     - 10 m resolution (e.g. Sentinel-2 high-res bands)
   * - ``x_20m``, ``y_20m``
     - 20 m resolution
   * - ``x_60m``, ``y_60m``
     - 60 m resolution
   * - ``x_300m``, ``y_300m``
     - 300 m resolution (e.g. Sentinel-3 OLCI)
   * - ``x_5000m``, ``y_5000m``
     - 5 km resolution (e.g. MSG / SEVIRI)

Each dimension coordinate variable must carry CF-compliant ``standard_name``
and ``units`` attributes.
