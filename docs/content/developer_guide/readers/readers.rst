.. currentmodule:: eoio

.. _add_readers:

#######
Readers
#######

In this guide, you will find detailed descriptions and
examples to describe how to add a satellite reader to *eoio*.

.. note::

   When adding a reader class, references to it **must** also be included in other parts of *eoio* for it
   to be used by higher level interface functions in *eoio*. See :ref:`add_readers.integrate` for more
   information on code wide requirements.


.. contents::
   :depth: 4

Useful API References
=====================

.. autosummary::
   :toctree: ../../api/
   :nosignatures:

    readers.factory.ReaderFactory


Creating a Reader Class
=======================

EO readers all inherit from `eoio.readers.base.BaseReader` and can make use of methods
within the class.

.. note::

   When creating a reader class make sure to place it in the `eoio.readers` folder and place respective test files in
   `eoio.readers.tests`. Examples of other reader implementations can similarly be found in those locations if you wish
   to familiarise yourself with current readers.

Base Classes
------------
Due to similarities between different readers, sometimes it makes sense to create more specialised base readers that
can simplify the implementation of child reader classes. Current classes and more specific notes on their
implementation requirements can be found by following the links below.

.. toctree::
    :caption: Current Base Classes
    :maxdepth: 2

    point
    raster


The rest of this page provides information on the `BaseReader` class that all readers must inherit from,
all information on this page is therefore still relevant regardless of whether or not your reader is inheriting
directly from `BaseReader`.

If creating a multi-spectral reader consider inheriting from `eoio.readers.msbase.MSBaseReader` for a simpler
implementation with a ready made format compatible with higher level interface methods.

Structure and Methods
---------------------

The way that *eoio* is structured allows developers flexibility in the implementation of their reader, **however**
there are still a number of requirements that **must** be met in the structure of the code to be compatible with the
the wider implementation of *eoio*.

*eoio* conventions
^^^^^^^^^^^^^^^^^^
Although there is no set way to organise the inner workings of a reader class, the convention to which a lot of the
current readers adhere to is indicated below::

    class YourReader(BaseReader):
        # class attributes here

        def __init__(...):
            ...

        # object properties
        # private methods
        # static methods
        # other methods

where `object properties`, `private methods` and `static methods` are organised alphabetically in each section and
`other methods` are organised in whichever way makes sense for the reader implementation.

.. _add_readers.required_implementation:
Required implementation
^^^^^^^^^^^^^^^^^^^^^^^
Readers all inherit from `eoio.readers.base.BaseReader` and must therefore
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
     - ``open_dataset(self) -> xr.Dataset``
     - ``Open selected data and metadata``
     -
   * - @property @abstractmethods
     -
     -
     -
   * -
     - ``all_options(self) -> dict``
     - ``Return dictionary of available subsetting parameters``
     - Dictionary of all subsetting parameter keys and all of the possible values for each of the keys
   * - class attributes
     -
     -
     -
   * -
     - ``_default_subset_dict: Dict[str, Any]``
     - ````
     - Default definition of desired subsetting parameters
   * -
     - ``_default_read_dict: Dict[str, Any]``
     - ````
     - Default definition of desired reading parameters


Suggested implementations for abstract methods
++++++++++++++++++++++++++++++++++++++++++++++

A number of similar methods have been implemented across different readers
whose format might be of use to those considering contributing their own reader to *eoio*.
The format of these methods and their purpose can be found in :ref:`add_readers.useful_methods`.


Input/Output Formats
--------------------

.. _add_readers.define_params:
Defining and setting subsetting parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned above `all_options` is an abstract property of all reader classes which returns all the valid subsetting
options available to a product. Defining the subsetting options available for a product is inherently product specific,
but there are a couple of conventions that you should follow when setting these in the interest of uniformity across
*eoio*.

Finding valid subsetting parameters
+++++++++++++++++++++++++++++++++++
Valid subsetting parameters can either be read from the product information

* product spectral meas_vars or other measurement variables
* auxiliary data variables (when names are able to be read from product)
* product masks (when names are able to be read from product)

or be set within the reader class.

* metadata
* auxiliary data variables (determined from a combination of product and product specification information)
* product masks (determined from a combination of product and product specification information)

Attributes inherent to a specified product, such as the available measurement variables or geographic bounds of a
product, are often set as `@property`'s of the class using private methods to parse product information, like so::

    @property
    def product_attribute(self):
        """
        Return the `product attribute` of the product
        """
        if self._product_attribute is None:
            self._product_parsing_method()  # a private method that parses some information from
                                            # the product and sets the property
        return self._product_attribute

which can then be used to determine the valid subsetting parameters.

Setting subsetting parameters
+++++++++++++++++++++++++++++
If any of your subsetting parameters are set to `True` this should select all available options for that parameter. A
mock example implementation for a multi-spectral reader can be seen below::

    # setting a property of an object
    self.selected_product_meas_vars = True

    # calls the setter
    @selected_product_meas_vars.setter
    def selected_product_meas_vars(self, meas_vars):
        if meas_vars is True:
            # set property to all available
            self._selected_product_meas_vars = self.available_meas_vars
        elif isinstance(meas_vars, list):
            ...

    # view selected product meas_vars
    print(self.selected_product_meas_vars)

    ["Band1", "Band2", "Band3", ..., "Band14"]

.. note::

    The `all_options` property implementation can make use of the property setter implementation described above,
    retrieving all available options by setting property values to `True`. The implementation used in the Sentinel-3
    OLCI Level 1B reader is::

        @property
        def all_options(self) -> dict:
            """
            Return dictionary of available subsetting parameters
            """
            if self._all_options is None:
                # retrieve values to set again
                mask, aux = self.mask, self.aux
                self.mask, self.aux = True, True
                # append True, False, None options to lists generated from available options using reader
                available_mask = (
                    self.mask + [True, False, None]
                    if isinstance(self.mask, list)
                    else self.mask
                )
                available_aux = (
                    self.aux + [True, False, None]
                    if isinstance(self.aux, list)
                    else self.aux
                )
                self.mask, self.aux = mask, aux
                self._all_options = {
                    "meas": self.available_meas + [True, False, None],
                    "read_img": [True, False, None],
                    "roi": [
                        list(v.exterior.coords) for k, v in self.bounds.items() if k != ""
                    ]
                    + [("(lon, lat)", "half-box width distance " "in meters")],
                    "roi_crs": [int(i[5:]) for i in self.bounds.keys() if i != ""],
                    "metadataLevel": self.meta_options,
                    "mask": available_mask,
                    "aux": available_aux,
                }
            return self._all_options

Insitu
++++++

MultiSpectral
+++++++++++++
Example `all_options` output from a Sentinel-2 Level 1 Product::

    {
     'meas': ['B01','B02’,..., True, False, None],
     'read_img': [True, False, None],
     'roi’: [product_coordinate_bounds, ('(lon, lat)', 'half-box width distance in meters')],
     'roi_crs': [4326, 32630],
     'metadataLevel': ['basic', 'partial’, ..., None],
     'mask': ['ancillary_lost’, ..., 'opaque’, ..., 'detector_footprint', True, False, None],
     'aux': ['tco3', 'tcwv’, ..., 'aod1240’, ..., None],
     }

Incorporating parameter variables into dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example application of code and required output (for dataset and each section too)

Measurement Variables
+++++++++++++++++++++

Auxiliary Variables
+++++++++++++++++++

Metadata
++++++++
All general product metadata (values that are either strings or lists qualify as such, any *variables* should be stored
in the manner described above in *Auxiliary Variables*) are stored in the attributes section of the dataset ::

    ds.attrs

this can easily be updated as you would any dictionary::

    ds.attrs.update({"my_key": "my_value", "my_dict": my_dict})


.. note::

    It is worth noting that `ds.attrs` can be a nested dictionary, although if your `ds.attrs` is a nested dictionary
    or contains data types that aren't strings, floats or lists, the standard way of saving a :py:class:`xarray.Dataset`
    as a netcdf file will not work.

Variable specific metadata should be stored in the attributes section of that specific variables, e.g. ::

    print(ds.Oa06.attrs)  # example from a Sentinel-3 OLCI L1B Product

    {'absolute_orbit_number': 38395,
     'ac_subsampling_factor': 64,
     'al_subsampling_factor': 1,
     'comment': '',
     'contact': 'eosupport@copernicus.esa.int',
     'creation_time': '2023-07-02T11:38:02Z',
     'history': '2023-07-02T11:38:02Z: PUGCoreProcessor joborder.22563601.xml',
     'institution': 'PS1',
     'netCDF_version': '4.2 of Mar 13 2018 10:14:33 $',
     'processing_baseline': 'OL__L1_.003.00.00',
     'product_name': 'S3A_OL_1_EFR____20230702T093544_20230702T093844_20230702T113802_0179_100_307_2160_PS1_O_NR_003.SEN3',
     'references': 'S3IPF PDS 004.1 - i2r5 - Product Data Format Specification - OLCI Level 1, S3IPF PDS 002 - i1r8 - Product Data Format Specification - Product Structures, S3IPF DPM 002 - i2r7 - Detailed Processing Model - OLCI Level 1',
     'resolution': '[ 270 294 ]',
     'source': 'IPF-OL-1-EO 06.13',
     'start_time': '2023-07-02T09:35:43.844237Z',
     'stop_time': '2023-07-02T09:38:43.809222Z',
     'title': 'OLCI Level 1b Product, Radiance Oa06 Data Set',
     'add_offset': 0.0,
     'ancillary_variables': 'Oa06_radiance_unc',
     'coordinates': 'time_stamp altitude latitude longitude',
     'long_name': 'TOA radiance for OLCI acquisition band Oa06',
     'standard_name': 'toa_upwelling_spectral_radiance_Oa06',
     'units': 'W/( m² * sr * μm)',
     'unc_comps': 'u_radiance_Oa06',
     'rsr_units': '1e-6 m',
     'response': ...
    }

and can be updated the same way you would your general metadata.

Required Metadata
"""""""""""""""""
Most of your metadata will be extracted from your EO Product and will retain the naming convention defined within,
however there are a number of metadata values which **must** be included in your output dataset.


.. list-table:: General
   :widths: 20 40 20
   :header-rows: 1

   * - key
     - value
     - example
   * - `product_name`
     - name of EO product
     - 'S3A_OL_1_EFR____20230702T093544_...SEN3'
   * - `platform`
     - name of the instrument platform used to acquire data
     - 'Sentinel-3A'
   * - `sensor_name`
     - name of the sensor used to acquire data
     - 'OLCI'
   * - `history`
     - logs an audit trail for modifications to the original data
     - '2023-09-07 07:09:21.250441: S3A_OL_1_EFR____20230702T093544...SEN3 read in using eoio version 0+untagged.415.g8c75b82.dirty'
   * - `processing_level`
     - processing level of the dataset
     - 'Level-1C'
   * - `product_date`
     - date of product observation, as a datetime object
     - datetime.datetime(2022, 1, 1, 3, 41, 41, 24000)
   * - `description`
     - description of the dataset
     - 'TBD'
   * - `collection_name`
     - name of collection product belongs to
     - 'S2MSI1C'

.. list-table:: Multispectral Raster Reader specific
   :widths: 20 40 20
   :header-rows: 1

   * - `spatial_resolution`
     - Spatial resolution of each band in the dataset, in metres by default
     - [60,10,10,10,20,20,20,10,60,60,20,20,20]
   * - `product_bounds`
     - Latitude/longitude bounds of the product
     - ''POLYGON ((108.6028789217982 41.52680855233116, 109.91844900373562 41.54671317678836, 109.93452476145569 40.55788159106108, 108.63847643330851 40.53865406413067, 108.6028789217982 41.52680855233116))'
   * - `geometry_ids`
     - List of geometry identifiers for each band in the dataset
     - ['60m','10m','10m','10m','20m','20m','20m','10m','60m','60m','20m','20m','20m']


.. list-table:: Variable Specific
   :widths: 20 40 20
   :header-rows: 1

   * - key
     - value
     - example
   * - `standard_name`
     - The name used to identify the physical quantity. A standard name contains no whitespace and is case sensitive.
     - 'toa_upwelling_spectral_radiance_Oa06'
   * - `long_name`
     - A long descriptive name which may, for example, be used for labeling plots.
     - 'TOA radiance for OLCI acquisition band Oa06'
   * - `units`
     - Representative units of the physical quantity. Unless it is dimensionless, a variable with a standard_name attribute must have units which are physically equivalent to `units`.
     - 'W/( m² * sr * μm)'

See `NetCDF Climate and Forecast (CF) Metadata Conventions <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_description_of_the_data>`_
for more complete descriptions of acceptable values.

.. note::

    The `history` attribute should use the `eoio.__version__` when adding information to the audit trail of a
    product. This can be obtained by placing ::

        from importlib.metadata import version

        __version__ = version("eoio")

    along with the other imports at the top of your module. This will then allows you to use `__version__` within your
    code to obtain the current `eoio.__version__`.

    An example use case can be seen below::

        if "history" in self.ds.attrs:
            self.ds.attrs["history"] = (
                self.ds.attrs["history"]
                + f"\n{dt.datetime.now()}: {os.path.split(self.path)[-1]} read in using eoio version {__version__}"
            )
        else:
            self.ds.attrs[
                "history"
            ] = f"{dt.datetime.now()}: {os.path.split(self.path)[-1]} read in using eoio version {__version__}"


Mask Variables
++++++++++++++
*eoio* makes use of
`obsarray <https://github.com/comet-toolkit/obsarray/blob/main/docs/content/user/flag_accessor.rst>`_ flag variables
"to define, store and interface with flag variables in :py:class:`xarray.Dataset`'s following the
`CF Convention <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#flags>`_
metadata standard." All mask variables such as *cloud*, or *quality* indicators should be stored in this format.

Accessing flags
"""""""""""""""
Flag variables can be created, viewed and accessed using the `.flag` accessor::

    print(ds.flag)  # example output for a Sentinel-3 L1B OLCI Product

    <FlagAccessor>
    Dataset Flags:
    * <FlagVariable>
    FlagVariable: 'quality_flags'
    ['saturated@Oa06', 'dubious', 'sun-glint_risk', 'duplicated', 'cosmetic', 'invalid', 'straylight_risk', 'bright', 'tidal_region', 'fresh_inland_water', 'coastline', 'land']

where `FlagVariable` lists all of the different flag variables there are, *quality*, *cloud* etc, and their individual
flag values can be accessed using::

    print(ds.flag.flag_vars)

    Data variables:
    quality_flags  (y_300m, x_300m) int64 2181038080 2181038080 ... 2181038080

    print(ds.flag["quality_flags"])

    <FlagVariable>
    FlagVariable: 'quality_flags'
    ['saturated@Oa06', 'dubious', 'sun-glint_risk', 'duplicated', 'cosmetic', 'invalid', 'straylight_risk', 'bright', 'tidal_region', 'fresh_inland_water', 'coastline', 'land']

    print(ds.flag["quality_flags"]["dubious"])

    <Flag>
    (y_300m: 4091, x_300m: 4865)>
    array([[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]])
    Dimensions without coordinates: y_300m, x_300m

To access a :py:class:`xarray.DataArray` version of a flag value, which allows for easier plotting etc., you can use
`.value`::

    print(ds.flag["quality_flags"]["dubious"].value)

    <xarray.DataArray (y_300m: 4091, x_300m: 4865)>
    array([[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]])
    Dimensions without coordinates: y_300m, x_300m

Each flag variable can also be accessed as a :py:class:`xarray.DataArray` within the dataset, like so::

    print(ds.quality_flags)

    <xarray.DataArray 'quality_flags' (y_300m: 4091, x_300m: 4865)>
    array([[2181038080, 2181038080, 2181038080, ..., 2181038080, 2181038080,
            2181038080],
           [2181038080, 2181038080, 2181038080, ..., 2181038080, 2181038080,
            2181038080],
           [2181038080, 2181038080, 2181038080, ..., 2181038080, 2181038080,
            2181038080],
           ...,
           [  33554432,   33554432,   33554432, ..., 2181038080, 2181038080,
            2181038080],
           [  33554432,   33554432,   33554432, ..., 2181038080, 2181038080,
            2181038080],
           [  33554432,   33554432,   33554432, ..., 2181038080, 2181038080,
            2181038080]], dtype=int64)
    Coordinates:
      * x_300m    (x_300m) float64 0.5 1.5 2.5 3.5 ... 4.862e+03 4.864e+03 4.864e+03
      * y_300m    (y_300m) float64 0.5 1.5 2.5 3.5 ... 4.088e+03 4.09e+03 4.09e+03
        lon_300m  (y_300m, x_300m) float64 1.289 1.292 1.295 ... 21.28 21.29 21.29
        latitude_300m  (y_300m, x_300m) float64 41.97 41.97 41.97 ... 49.89 49.89 49.89
    Attributes:
        flag_meanings:  saturated@Oa06 dubious sun-glint_risk duplicated cosmetic...
        flag_mask:     32768,2097152,4194304,8388608,16777216,33554432,67108864,...


Adding flags
""""""""""""""
There are (at least) two different ways to add new flag variables to your dataset. If you have boolean arrays for
your mask it is easiest to add these values in the conventional way (as shown in the
`obsarray <https://github.com/comet-toolkit/obsarray/blob/main/docs/content/user/flag_accessor.rst>`_ documentation).
This way requires you to set each of your mask values individually as shown below::

    ds.flag["new_flag_var"] = (["dims"], {"flag_meanings": ["flag_value_1", "flag_value_2"]})
    ds.flag["new_flag_var"]["flag_value_1"][:, :] = flag_data # boolean array
    ds.flag["new_flag_var"]["flag_value_2"][:, :] = flag_data # boolean array

*See the Sentinel-2 MSI Reader for an example implementation.*

However in the case where your masks are already combined into a single array using bit field notation, you can assign
all of the flag variable data in one go. ::

    # assign flags as flag variables
    self.ds["quality_flags"] = (["dims"], flag_data)

    # update attributes
    self.ds.quality_flags.attrs = {
        "flag_meanings": flag_meanings,
        "flag_masks": flag_masks,
    }

This creates an :py:class:`xarray.DataArray` in such a format that allows *obsarray* to recognise it as a flag. For this
to be the case you must know how the flag meanings (space separated flag value names) and flag masks (comma separated
flag value integers) relate to one another. ::

    flag_meanings = 'saturated@Oa06 dubious sun-glint_risk duplicated cosmetic invalid straylight_risk bright tidal_region fresh_inland_water coastline land'
    flag_masks = '32768,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912,1073741824,2147483648'

For the case where not all of the flags are requested by the user, the flag data must be filtered to remove the
unwanted flag values. This can be done using bitwise operations, by retrieving the integer corresponding to all the
unwanted flag values and then removing them from the data when assigning to the flag variable. ::

    # get flag value for all non-requested masks
    non_requested_masks = sum(
        [int(i[1]) for i in zip(flag_meanings, flag_masks) if i[0] not in masks]
    )  # where masks is the list of user requested masks

    # assign flags as flag variables
    ds["quality_flags"] = (
        ("y_300m", "x_300m"),
        ~non_requested_masks & flags_ds.data,  # this gets rid of the data from non_requested_masks
    )


*See Sentinel-3 L1B OLCI Reader and Landsat-8/9 Reader for example implementations.*

Uncertainty Components
++++++++++++++++++++++
unc

Coordinates
+++++++++++
coords

Product and Dataset Format Specification Documents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`eoio Output Dataset Format <https://npluk-my.sharepoint.com/:w:/g/personal/mattea_goalen_npl_co_uk/ETGXMT6ID_pFk0R2Z1-BnTsB53UlBgSF0glRYHnY3Y-bWg?e=FP3oYh>`_

List of links to EO Product Specification Documents
    * `Sentinel-2 MSI Level 0-2 <https://sentinel.esa.int/documents/247904/685211/S2-PDGS-TAS-DI-PSD-V14.9.pdf/>`_
    * `Sentinel-3 OLCI Level 1 <https://sentinel.esa.int/documents/247904/1872756/S3IPF+PDS+004.1+-+i2r6+-+Product+Data+Format+Specification+-+OLCI+Level+1.pdf/>`_
    * `Landsat-8/9 OLI TIRS Collection 2 Level 1 <https://earth.esa.int/eogateway/documents/20142/0/Landsat-8-9-OLI-TIRS-Collection-2-Level-1-Data-Format-Control-Book-DFCB.pdf>`_
    * `Landsat-8/9 OLI TIRS Collection 2 Level 2 <https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v5.pdf>`_
    * `RadCalNet <https://www.radcalnet.org/resources/RadCalNetQuickstartGuide_20180702.pdf/>`_
    * `Hypernets <https://hypernets-processor.readthedocs.io/en/latest/content/atbd/products.html>`_
    * `Sentinel-3 SLSTR Level 1 <https://sentinels.copernicus.eu/documents/247904/1872792/S3IPF+PDS+005.1+-+i2r10+-+Product+Data+Format+Specification+-+SLSTR+Level+1_20211015_online.pdf/59cdac94-f0d5-3a2e-3013-c524dbce9814?t=1648134076063>`_




.. _add_readers.useful_methods:
Useful class methods
--------------------

Initialising your reader class
++++++++++++++++++++++++++++++

When initialising your reader object it is *highly encouraged* to reference the `eoio.readers.base.BaseReader`
`__init__`, done using the `super()` call which can be seen below. This checks that all your subsetting keys are valid
and sets all non-requested values to their default values given by the `_default_subset_dict` attribute of the class.
You should also initialise (set to ``None``) any properties/attributes, both those inherent to the specified product
(those which can be parsed from the product directly) and any required subsetting attributes (values that can be
specified by the user) as shown below. ::

    def __init__(
       self,
       path,
       subset_info: Optional[Dict[str, Any]] = None,
       read_params: Optional[Dict[str, Any]] = None,
    ):
       """Initialise the reader"""
       super(YourReader, self).__init__(path, subset_info, read_params)

       # attributes inherent to the specified product
       self._available_meas = None
       self._bounds = None
       self._default_crs = None
       self._name = None
       self._instrument = None

       # subsetting attributes
       self._aux = None
       self._crs = None
       self._geometry_ids = None
       self._mask = None
       self._meta_level = None
       self._roi = None
       self._selected_product_meas_vars = None

       # initialise dataset
       self.ds = None


Abstract Methods
++++++++++++++++
There are only two `@abstractmethods` from `eoio.readers.base.BaseReader` that require implementation in your
reader (if you're inheriting from other base classes please refer to their respective documentation pages for more
information) as seen above in :ref:`add_readers.required_implementation`.

An example `all_options` implementation can be seen below. For tips on how to make use of `@property.setter` for easier
implementation see :ref:`add_readers.define_params`::

        @property
        def all_options(self) -> dict:
            """
            Return dictionary of available subsetting parameters
            """
            if self._all_options is None:
                # retrieve all possible options for each subsetting parameter
                available_masks = [...]
                available_aux = [...]
                available_meas_vars = [...]
                metadata_options = [...]

                # set all_options property
                self._all_options = {
                    "meas": available_meas_vars,
                    "read_img": [True, False, None],
                    "roi": [...],
                    "roi_crs": 4326,
                    "metadataLevel": metadata_options,
                    "mask": available_masks,
                    "aux": available_aux,
                }
            return self._all_options

Example `open_dataset` implementation for a multi-spectral reader::

    def open_dataset(self) -> xr.Dataset:
        """
        Open selected data and metadata
        :return: xr.Dataset of desired subset
        """
        # create Dataset to populate
        self.ds = xr.Dataset()

        # check and set subsetting inputs
        self.aux = self.subset_info["aux"]
        self.mask = self.subset_info["mask"]
        self.meta_level = self.subset_info["metadataLevel"]
        self.roi = self.subset_info["roi"], self.subset_info["roi_crs"]
        self.selected_product_meas = self.subset_info["meas"]

        # add variable names if required
        if (
            self.subset_info["read_img"] or self.subset_info["metadataLevel"]
        ):  # if meas_var DataArrays are required
            self.ds = self.ds.assign(
                dict(
                    zip(
                        self.selected_product_meas_vars,
                        [None] * len(self.selected_product_meas_vars),
                    )
                )
            )

        if self.subset_info["read_img"]:
            self.read_image()
        if self.meta_level:
            self.read_meta()
        if self.mask:
            self.read_masks()
        if self.aux:
            self.read_aux()

        return self.ds

where the methods `read_image`, `read_meta`, `read_masks` and `read_aux` are defined below::

    def read_image(self) -> None:
        """
        Read in image data

        Add image meas_var data to meas_vars present in the initialised xr.Dataset,
        subsetting in accordance with the subsetting parameters provided in subset_info.
        Resulting xr.Dataset has converted input coordinates from initial
        coordinate reference system to the World Geodetic System 1984 (WGS 84)
        """

    def read_meta(self) -> None:
        """
        Read metadata in and assign to dataset attributes
        """

    def read_masks(self) -> None:
        """
        Read in mask data and add values as flags to the xr.Dataset
        """

    def read_aux(self) -> None:
        """
        Read in auxiliary data and add to xr.Dataset
        """

.. note::

   These examples have been taken from `eoio.readers.msbase.MSBaseReader` and may not
   be directly applicable to your sensor, but their structure might be of use. To understand
   more about their implementation it is recommended to look at the source code from the
   `eoio.readers.msbase.MSBaseReader` API page.


.. _add_readers.integrate:
Integrate into *eoio*
=====================

When adding a reader it is crucial for it to be included in
``eoio.readers.factory.ReaderFactory.get_reader`` for it to be used by the ``read``
function. The reader required is inferred from the product filepath so a regular
expression for your reader is also required, for example
``re.compile(r"S3.?_OL_1.*.SEN3\Z")`` for Sentinel-3 OLCI Level 1B products.