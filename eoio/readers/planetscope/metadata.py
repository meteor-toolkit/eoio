"""eoio.readers.planetscope.metadata - extractor class for PlanetScope metadata."""

from __future__ import annotations
import json
from typing import Any
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.utils.dict_tools import (
    dict_merge,
)  # TODO get from processor_tools once implemented
from processor_tools.utils.dict_tools import get_value
import datetime as dt
from eoio.readers.footprint_utils import normalize_footprint

from importlib.metadata import version

__version__ = version("eoio")

min_basic_var_metadata_keys = [
    "long_name",
    "standard_name",
    "units",
]


class PlanetScopeMetadataExtractorError(ValueError):
    pass


class PlanetScopeMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata helper for PlanetScope products.

    Solely responsible for extracting metadata. No heavy dependencies.

    Can be full or basic level.
    """

    def __init__(self, reader):
        super().__init__(reader)
        self.metadata_filepaths = (
            reader.layout.metadata_files()
        )  # if metadata files exist, will be a list of 3 filepaths

    def read_metadata_files(self) -> dict:
        """
        Read metadata from XML and JSON files
        """
        import xmltodict

        metadata: dict = {}

        for metadata_filepath in self.metadata_filepaths:
            if metadata_filepath.endswith(".xml"):
                with open(metadata_filepath) as metadata_xml_file:
                    metadata = dict_merge([metadata, xmltodict.parse(metadata_xml_file.read())])
                    metadata_xml_file.close()
            elif metadata_filepath.endswith(".json"):
                with open(metadata_filepath) as metadata_json_file:
                    metadata = dict_merge([metadata, json.load(metadata_json_file)])
                    metadata_json_file.close()
            else:
                raise PlanetScopeMetadataExtractorError(f"Unsupported metadata file format: {metadata_filepath}")

        return metadata

    def get_product_metadata(self) -> dict:
        """
        Note: reflectance coeff and radiometric scale factor info (for processor) found in xml, everything else in the two jsons.
        """
        prod_md = self.get_basic_metadata()

        prod_md["geospatial_bounds_crs"] = "EPSG:" + str(get_value(self.read_metadata_files(), "proj:epsg"))
        prod_md["product_properties"] = get_value(self.read_metadata_files(), "properties")
        prod_md["cloud_area_fraction"] = get_value(self.read_metadata_files(), "cloud_percent")

        return prod_md

    def get_basic_metadata(self) -> dict:
        """
        Extract basic metadata from the product metadata, to be used in other tools in MetEOR.

        :returns: Basic metadata dictionary.
        """
        full_prod_metadata = self.read_metadata_files()

        date = dt.datetime.strptime(
            get_value(full_prod_metadata, "eop:acquisitionDate").split("+")[0],
            "%Y-%m-%dT%H:%M:%S",
        )
        basic_prod_md = {
            "collection_name": (
                get_value(full_prod_metadata, "constellation")
                if get_value(full_prod_metadata, "constellation") is not None
                else "PlanetScope"
            ),
            "product_name": get_value(full_prod_metadata, "eop:identifier"),
            "platform": get_value(full_prod_metadata, "platform"),
            "instrument": get_value(full_prod_metadata, "instruments"),
            "processing_level": get_value(full_prod_metadata, "eop:productType"),
            "satellite_id": get_value(full_prod_metadata, "platform"),
            "constellation": get_value(full_prod_metadata, "constellation"),
            "spatial_resolution": [self.reader.meas_var_res[meas_var] for meas_var in self.reader.meas_var_res.keys()],  # type: ignore[attr-defined]
            "geometry_ids": [
                str(self.reader.meas_var_res[meas_var]) + "m"  # type: ignore[attr-defined]
                for meas_var in self.reader.meas_var_res.keys()  # type: ignore[attr-defined]
            ],
            "product_geospatial_bounds": get_value(full_prod_metadata, "bbox"),  # gets EPSG:4326
            "product_datetime": date.isoformat(),
            "description": "TBD",
            "institution": "Planet",
            "keywords": [
                "planetscope",
                "superdove",
                "satellite",
                "remote sensing",
                "multispectral",
                "earth observation",
                "basic analytic",
                "visible",
            ],
            "source": "PlanetScope SuperDove Basic Analytic Satellite Product",
            "platform_type": "satellite",
            "footprint": normalize_footprint(
                geometry_input=get_value(full_prod_metadata, "bbox"),
                crs_input=4326,  # PlanetScope products use EPSG:4326
            ),
        }

        return basic_prod_md

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the product metadata.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
        """
        var_md: dict = {}

        # Observation geometry is typically at 3m resolution for PlanetScope
        if var == "observation_geometry":
            var_md = {
                "spatial_resolution": 3,
                "spatial_resolution_units": "m",
                "geometry_id": "3m",
            }
            return var_md

        band_rel_dict = {
            "B1": "Coastal Blue",
            "B2": "Blue",
            "B3": "Green I",
            "B4": "Green",
            "B5": "Yellow",
            "B6": "Red",
            "B7": "Red Edge",
            "B8": "Near-Infrared",
        }

        mtd_files = self.read_metadata_files()

        # get list of band info from prod metadata that has the "common_name" key
        band_metadata_dicts = get_value(mtd_files, "eo:bands", multiple=True)
        for bnd_tuple in band_metadata_dicts:
            if get_value(bnd_tuple[1], "common_name", multiple=True):
                band_metadata_list = bnd_tuple[1]

        for i in range(len(band_metadata_list)):
            if band_rel_dict[var] == band_metadata_list[i]["name"]:
                var_md = {
                    "band_id": var,
                    "band_name": band_rel_dict[var],
                    # STAC eo:bands gives wavelengths in micrometres; reported here in nm
                    # like every other eoio reader (round: 0.442 * 1000 is 442.00000000000006)
                    "band_central_wavelength": round(float(band_metadata_list[i]["center_wavelength"]) * 1000, 6),
                    "band_central_wavelength_units": "nm",
                    "band_full_width_half_max": round(float(band_metadata_list[i]["full_width_half_max"]) * 1000, 6),
                    "standard_name": "toa_radiance",
                    "long_name": "TOA radiance",
                    "units": "W/( m² * sr * μm)",
                    "spatial_resolution": self.reader.meas_var_res[var],  # type: ignore[attr-defined]
                    "spatial_resolution_units": "m",
                    "geometry_id": str(self.reader.meas_var_res[var]) + "m",  # type: ignore[attr-defined]
                    "ancillary_variables": [],
                    "measurand": "radiance",
                }
                if get_value(mtd_files, "ps:bandSpecificMetadata")[i]["ps:bandNumber"] == str(i + 1):
                    var_md["radiometric_scale_factor"] = get_value(mtd_files, "ps:bandSpecificMetadata")[i][
                        "ps:radiometricScaleFactor"
                    ]
                    var_md["reflectance_coefficient"] = get_value(mtd_files, "ps:bandSpecificMetadata")[i][
                        "ps:reflectanceCoefficient"
                    ]

        return var_md

    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the product metadata.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
        """
        basic_var_md: dict = {}

        # aux metadata done within aux module so pass here
        if var == "observation_geometry":
            return basic_var_md

        band_rel_dict = {
            "B1": "Coastal Blue",
            "B2": "Blue",
            "B3": "Green I",
            "B4": "Green",
            "B5": "Yellow",
            "B6": "Red",
            "B7": "Red Edge",
            "B8": "Near-Infrared",
        }

        # get list of band info from prod metadata that has the "common_name" key
        band_metadata_dicts = get_value(self.read_metadata_files(), "eo:bands", multiple=True)
        for bnd_tuple in band_metadata_dicts:
            if get_value(bnd_tuple[1], "common_name", multiple=True):
                band_metadata_list = bnd_tuple[1]

        for i in range(len(band_metadata_list)):
            if band_rel_dict[var] == band_metadata_list[i]["name"]:
                basic_var_md = {
                    "standard_name": "toa_radiance",
                    "long_name": f"TOA radiance in band {var}",
                    "units": "W/( m² * sr * μm)",
                    "measurand": "radiance",
                }

        return basic_var_md

    def get_angle_metadata(self) -> dict:
        """
        Return metadata for angle variables (solar/observer angles).

        :returns: Dictionary of angle variable metadata.
        """
        angle_attrs = {
            "solar_zenith_angle": {
                "units": "degrees",
                "long_name": "Solar Zenith Angle",
                "standard_name": "solar_zenith_angle",
                "measurand": "angle",
                "geometry_id": "3m",
                "spatial_resolution": 3,
                "spatial_resolution_units": "m",
                "description": "Solar zenith angle is the the angle between the line of sight to the sun and the local vertical.",
                "source_variable_name": "90 - sun_elevation",
            },
            "solar_azimuth_angle": {
                "units": "degrees",
                "long_name": "Solar Azimuth Angle",
                "standard_name": "solar_azimuth_angle",
                "measurand": "angle",
                "geometry_id": "3m",
                "spatial_resolution": 3,
                "spatial_resolution_units": "m",
                "description": "Solar azimuth angle is the horizontal angle between the line of sight to the sun and a reference direction which is often due north. The angle is measured clockwise.",
                "source_variable_name": "sun_azimuth",
            },
            "sensor_zenith_angle": {
                "units": "degrees",
                "long_name": "Viewing Zenith Angle",
                "standard_name": "sensor_zenith_angle",
                "measurand": "angle",
                "geometry_id": "3m",
                "spatial_resolution": 3,
                "spatial_resolution_units": "m",
                "description": "Viewing zenith angle is the angle between the line of sight to the sensor and the local zenith at the observation target. This angle is measured starting from directly overhead and its range is from zero (directly overhead the observation target) to 180 degrees (directly below the observation target). Local zenith is a line perpendicular to the Earth's surface at a given location. 'Observation target' means a location on the Earth defined by the sensor performing the observations.",
                "source_variable_name": "view_angle",
            },
            "sensor_azimuth_angle": {
                "units": "degrees",
                "long_name": "Viewing Azimuth Angle",
                "standard_name": "sensor_azimuth_angle",
                "measurand": "angle",
                "geometry_id": "3m",
                "spatial_resolution": 3,
                "spatial_resolution_units": "m",
                "description": "Viewing azimuth angle is the horizontal angle between the line of sight from the observation point to the sensor and a reference direction at the observation point, which is often due north. The angle is measured clockwise positive, starting from the reference direction.",
                "source_variable_name": "satellite_azimuth",
            },
        }

        return angle_attrs

    def get_aux_metadata(self) -> dict:
        """
        Return metadata for auxiliary data variables.
        TODO implement when masks are implemented - see Landsat reader for example.

        :returns: Dictionary of auxiliary data variable metadata.
        """
        aux_vars: dict[str, Any] = {}

        return aux_vars
