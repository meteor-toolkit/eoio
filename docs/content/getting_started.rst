################
Getting Started
################

The getting started guide aims to get you using eoio productively as quickly as possible.
It is designed as an entry point for new users, and it provided an introduction to eoio's main function.

.. _installation:

Installation
++++++++++++

*eoio* is installable via pip. It's always recommended to make a virtual environment for each of your python
projects.

#. Create a python virtual environment::

        $ cd eoio/
        $ python -m venv venv
        $ . venv/bin/activate

#. Install the module with pip. For development it is recommended to install in editable mode with the optional developer dependenies::

        $ pip install -e ".[dev]"

.. _basic_example:

Basic Example
+++++++++++++

**eoio** reads in satellite data from a satellite product into an :py:class:`xarray.Dataset` via the use of the ``read()`` function, subsetting and pre-processing can also be applied::

    # Define input path from test datasets
    SAFE_PATH = r"T:\ECO\EOServer\data\unittest_datasets\S2MSIL1C\S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"

    # Define ROI
    roi_string = "POLYGON ((\
    -0.913910 53.786950,\
    -0.382900 53.776490,\
    -0.406240 53.464690,\
    -0.925060 53.474770,\
    -0.913910 53.786950\
    ))"

    geom = wkt.loads(roi_string)

    # Read dataset
    ds = read(
        SAFE_PATH,
        subset={"roi": geom, "roi_crs": "EPSG:4326"},
        read_params={"use_chunks": True},
        processors={"add_lat_lon": {}}
    )





