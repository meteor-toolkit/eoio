Reading in Satellite Data
+++++++++++++++++++++++++

Using the read function
=============================

**eoio** reads in satellite data from a satellite product into an :py:class:`xarray.Dataset` via the use of the ``read()`` function which is also capable of applying processors to these datasets.

The read function can be used as in the following::

    satellite_ds = read(path, vars_sel: Optional[dict] = None, subset: Optional[dict] = None, read_params: Optional[dict] = None, processors: Optional[dict] = None)

where all inputs (apart from product path) are optionally provided.

Input Parameters
-----------------
Product Path
^^^^^^^^^^^^^
The read function takes in the satellite product filepath as the input to be read. The format of the filepath differs for each satellite product and is listed in :ref:`product_paths`. An example Sentinel-2 product path looks something like::

    path = T:\ECO\EOServer\data\satellite\S2A_MSI\L1\PICS\Libya4\S2A_MSIL1C_20151223T091412_N0201_R050_T34RGS_20151223T092043.SAFE

Subsetting and Reading Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Various options can be specified in order to control the exact usage of the read function.
The read function uses separate parameters for variable selection, subsetting, and reading options:

* ``vars_sel`` - a dictionary specifying which variables to read, with keys like ``"meas"`` (measurement variables) and ``"aux"`` (auxiliary variables)
* ``subset`` - a dictionary containing spatial subsetting parameters such as region of interest (ROI) and its coordinate reference system
* ``read_params`` - a dictionary containing reading parameters such as chunking options

Auxiliary data (e.g. Sentinel-2's ECMWF/CAMS meteo, angles, atmospheric inputs) is read on a
best-effort basis: if one auxiliary source is unavailable or fails to read (a missing or corrupt
file, for example), a warning is raised and that source's variables are simply left out of the
returned dataset rather than aborting the whole read -- measurement variables and any other
auxiliary sources that did read successfully are unaffected.

For a Sentinel-2 L1C product a ``vars_sel`` and ``subset`` dictionary may look like::

    vars_sel = {
        "meas": ["B02", "B03", "B04"],
        "aux": ["viewing_zenith_angle_B03"],
    }

    subset = {
        "roi": geom,  # shapely geometry, GeoJSON dict, or polygon coordinates
        "roi_crs": "EPSG:4326",
    }

All available variable names and supported combinations for a given product can be inspected via the documentation api or using the ``eoio.product_options`` function::

    from eoio.interface import product_options

    product_path = "path/to/product"
    options = product_options(product_path)
    print(options["vars_sel"])

A more complete page with all available parameters and example usage of the options for the read function will be added in the future.

Processors (and their parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A number of processors are also available which can be run to add additional information to the returned dataset. 
Processors to apply to your dataset can be selected and passed through the read function using the ``processors`` input:

* ``processors`` - a dictionary with the names of the post processors to apply to the dataset along with any required inputs for those processors.

The format of these processing parameters looks something like this::
    
    processors = {
        "add_lat_lon": {},
        "interpolate": {
            "coords": ["x_5000m", "y_5000m"],
            "target_grid": ["x_10m", "y_10m"],
            "method": "linear",
            "data_vars": ['solar_zenith_angle'],
            "inplace": False,
            },
        "s2_rut": {
            "data_vars": ["B01", "B02", "B03"],
            "group_unc": True,  # set to True to compute group uncertainties (systematic and random)
            },
        }
    }

Here the first processor adds latitude and longitude coordinates to the dataset; the second processor interpolates the solar zenith angle from a 5000m grid to a 10m grid; and the third processor applies the Sentinel-2 Radiometric Uncertainty Tool (RUT) to compute systematic and random uncertainties for the selected bands (this does need the sza to be interpoalted to 10m).
All available processors and their inputs can be discovered through the documentation api or using the ``eoio.product_processors`` function::

    from eoio.interface import product_processors

    product_path = "path/to/product"
    processors = product_processors(product_path)
    print(processors)

A more complete page with all available parameters and example usage of the procesors will be added in the future.

Simple Example for Sentinel-2 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a simple example of reading a Sentinel-2 L2A product (including all bands and no further processing)::

    SAFE_PATH = "path/to/S2A_MSIL2A_...SAFE"

    # Read product
    ds = read(
        SAFE_PATH,
    )

Complete Example for Sentinel-2 with RUT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a complete example of reading a Sentinel-2 L1C product with subsetting, variable selection, and processors (the processing is significantly faster for smaller ROI)::

    from shapely import wkt
    from eoio import read

    SAFE_PATH = "path/to/S2A_MSIL1C_...SAFE"

    # Define region of interest
    roi_string = "POLYGON ((lon1 lat1, lon2 lat2, lon3 lat3, ...))"
    roi = wkt.loads(roi_string)

    # Read with all parameters
    ds = read(
        SAFE_PATH,
        vars_sel={
            "meas": ["B02", "B03", "B04"],
            "aux": ["solar_zenith_angle"],
        },
        subset={"roi": roi, "roi_crs": "EPSG:4326"},
        read_params={"use_chunks": True},
        processors={"add_lat_lon": {},
                    "interpolate": {
                        "coords": ["y_5000m", "x_5000m"],
                        "data_vars": ['solar_zenith_angle'],
                        "target_grid": ["y_10m", "x_10m"],
                        "method": "linear",
                        "inplace": False,
                    },
                    "s2_rut": {
                        "data_vars": ["B01", "B02", "B03"],
                        "group_unc": True,  # set to True to compute group uncertainties (systematic and random)
                    },
        }
    )

There are also example scripts available in the ``examples`` directory of the repository which illustrate the use of the read function for different satellite products.

Output Dataset
--------------
Example Sentinel-2 output dataset (with selecting bands, processors, uncertainties, ... - see the example above)::

    <xarray.Dataset> Size: 594MB
    Dimensions:                 (x_60m: 583, y_60m: 579, x_10m: 3502, y_10m: 3475,
                                x_5000m: 23, y_5000m: 23, y_5000m_interp_0: 23,
                                y_5000m_interp_1: 23, x_5000m_interp_0: 23,
                                x_5000m_interp_1: 23)
    Coordinates:
    * x_60m                   (x_60m) float64 5kB 6.375e+05 ... 6.724e+05
    * y_60m                   (y_60m) float64 5kB 5.962e+06 ... 5.927e+06
    * x_10m                   (x_10m) float64 28kB 6.374e+05 ... 6.724e+05
    * y_10m                   (y_10m) float64 28kB 5.962e+06 ... 5.927e+06
    * x_5000m                 (x_5000m) int64 184B 600000 605000 ... 705000 710000
    * y_5000m                 (y_5000m) int64 184B 6000000 5995000 ... 5890000
    * y_5000m_interp_0        (y_5000m_interp_0) int64 184B 6000000 ... 5890000
    * y_5000m_interp_1        (y_5000m_interp_1) int64 184B 6000000 ... 5890000
    * x_5000m_interp_0        (x_5000m_interp_0) int64 184B 600000 ... 710000
    * x_5000m_interp_1        (x_5000m_interp_1) int64 184B 600000 ... 710000
        spatial_ref             int64 8B 0
    Data variables:
        B01                     (y_60m, x_60m) float32 1MB dask.array<chunksize=(579, 583), meta=np.ndarray>
        B02                     (y_10m, x_10m) float32 49MB dask.array<chunksize=(1674, 1747), meta=np.ndarray>
        B03                     (y_10m, x_10m) float32 49MB dask.array<chunksize=(1674, 1747), meta=np.ndarray>
        solar_zenith_angle      (y_5000m, x_5000m) float64 4kB 75.96 75.95 ... 74.81
        solar_zenith_angle_60m  (y_60m, x_60m) float64 3MB 75.56 75.56 ... 75.2 75.2
        solar_zenith_angle_10m  (y_10m, x_10m) float64 97MB 75.56 75.56 ... 75.2
        u_systematic_B01        (y_60m, x_60m) float64 3MB 0.003413 0.003421 ... nan
        u_random_B01            (y_60m, x_60m) float64 3MB 0.000946 ... nan
        u_systematic_B02        (y_10m, x_10m) float64 97MB 0.002606 ... nan
        u_random_B02            (y_10m, x_10m) float64 97MB 0.00181 0.001425 ... nan
        u_systematic_B03        (y_10m, x_10m) float64 97MB 0.001929 ... nan
        u_random_B03            (y_10m, x_10m) float64 97MB 0.001194 0.00131 ... nan
    Attributes: (12/28)
        title:                     S2MSI1C
        summary:                   TBC
        institution:               TBC
        source:                    TBC
        references:                TBD
        date_created:              2026-04-11T14:47:26.164923+00:00
        ...                        ...
        eoio:version:              v0.1+625.g85cdac4.dirty
        eoio:path:                 T:\ECO\EOServer\data\unittest_datasets\S2MSIL1...
        history:                   2026-04-11T14:47:26.164923+00:00: S2A_MSIL1C_2...
        eoio:reader:               sentinel2
        eoio:subset:               (637430.357340333, 5927091.05886575, 672449.28...
        eoio:processing_steps:     [{'processor': 'interpolate', 'interpolated_co...

where band values, solar and viewing angles and auxiliary data are stored as variables; band specific metadata is
stored within its corresponding variables' attributes; general metadata is stored within the datasets attributes; and
masks are stored as obsarray flags.

Accessing Variables
-------------------
Variables and metadata can be accessed in the same way as you would access any xarray variable or attribute.

Selecting a specific band
^^^^^^^^^^^^^^^^^^^^^^^^^
::

    print(ds["B02"])

    <xarray.DataArray 'B02' (y_10m: 3375, x_10m: 2577)>
    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]])
    Coordinates:
        band           int32 1
      * x_10m          (x_10m) float64 3.57e+05 3.57e+05 ... 3.828e+05 3.828e+05
      * y_10m           (y_10m) float64 4.573e+06 4.573e+06 ... 4.54e+06 4.54e+06
        spatial_ref    int32 0
        longitude_10m         (y_10m, x_10m) float64 109.3 109.3 109.3 ... 109.6 109.6 109.6
        latitude_10m         (y_10m, x_10m) float64 41.3 41.3 41.3 41.3 ... 41.0 41.0 41.0
        number         int32 0
        time           datetime64[ns] 2022-03-18T12:00:00
        step           timedelta64[ns] 15:36:00
        surface        float64 0.0
        valid_time     datetime64[ns] 2022-03-19T03:36:00
        isobaricInhPa  float64 0.0
    Attributes:
        units:
        standard_name:                        toa_bidirectional_reflectance_B02
        long_name:                            Reflectance in band B02
        ABSOLUTE_CALIBRATION_ACCURACY:        5
        ALPHA:                                0.567
        BETA:                                 0.04696
        CROSS_BAND_CALIBRATION_ACCURACY:      3
        MULTI_TEMPORAL_CALIBRATION_ACCURACY:  1
        PHYSICAL_GAINS:                       FULL_RESOLUTION
        RADIO_ADD_OFFSET:                     -1000
        RESOLUTION:                           10
        SOLAR_IRRADIANCE:                     {'@unit': 'W/m²/µm', '#text': 1959.66}

