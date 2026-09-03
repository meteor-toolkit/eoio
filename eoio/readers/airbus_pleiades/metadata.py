"""eoio.readers.airbus_pleiades.metadata - extractor class for Airbus Pleiades metadata."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.utils.dict_tools import (
    dict_merge,
)  # TODO get from processor_tools once implemented
from processor_tools.utils.dict_tools import get_value
import datetime as dt
from eoio.readers.footprint_utils import normalize_footprint

min_basic_var_metadata_keys = [
    "long_name",
    "standard_name",
    "units",
]


class PleiadesMetadataExtractorError(ValueError):
    pass


class PleiadesMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata helper for Airbus Pleiades MS_ORT (L1) products.

    Solely responsible for extracting metadata. No heavy dependencies.

    Can be full or basic level.
    """

    def __init__(self, reader):
        super().__init__(reader)
        self.metadata_filepath = Path(reader.layout.metadata_file())

    def read_metadata_xml(self) -> dict:
        """
        Read metadata from XML files
        """
        metadata: dict = {}
        import xmltodict

        with open(self.metadata_filepath) as metadata_xml_file:
            metadata_dict = xmltodict.parse(metadata_xml_file.read())
            metadata_xml_file.close()

        metadata = dict_merge([metadata, metadata_dict])

        return metadata

    def get_product_metadata(self) -> dict:
        """
        Read metadata from XML files
        """
        prod_md = self.get_basic_metadata()

        prod_md["cloud_area_fraction"] = get_value(self.read_metadata_xml(), "CLOUD_COVERAGE")["#text"]

        return prod_md

    def get_basic_metadata(self) -> dict:
        """
        Extract basic metadata from the dataset, to be used in other tools in MetEOR.

        :returns: Basic metadata dictionary.
        """
        full_extracted_metadata = self.read_metadata_xml()

        date = dt.datetime.strptime(
            get_value(full_extracted_metadata, "IMAGING_DATE")
            + "T"
            + get_value(full_extracted_metadata, "IMAGING_TIME"),
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )

        # Extract product_name from DATA_FILE_PATH, handling dict with @href
        data_file_path = get_value(full_extracted_metadata, "DATA_FILE_PATH")
        if isinstance(data_file_path, dict) and "@href" in data_file_path:
            product_name = Path(data_file_path["@href"]).name
        else:
            product_name = str(data_file_path) if data_file_path else ""

        basic_md = {
            "collection_name": (
                get_value(full_extracted_metadata, "PRODUCT_CODE")
                if get_value(full_extracted_metadata, "PRODUCT_CODE") is not None
                else "Pleiades"
            ),
            "product_name": product_name,
            "platform": (
                get_value(full_extracted_metadata, "MISSION")
                + "_"
                + get_value(full_extracted_metadata, "MISSION_INDEX")
            ),
            "instrument": (
                get_value(full_extracted_metadata, "INSTRUMENT")
                + "_"
                + get_value(full_extracted_metadata, "INSTRUMENT_INDEX")
            ),
            "processing_level": (
                get_value(full_extracted_metadata, "PROCESSING_LEVEL")
                + "_"
                + get_value(full_extracted_metadata, "SPECTRAL_PROCESSING")
            ),
            "satellite_id": (
                get_value(full_extracted_metadata, "MISSION")
                + "_"
                + get_value(full_extracted_metadata, "MISSION_INDEX")
            ),
            "constellation": (
                get_value(full_extracted_metadata, "PRODUCT_CODE")
                if get_value(full_extracted_metadata, "PRODUCT_CODE") is not None
                else ""
            ),
            "spatial_resolution": [self.reader.meas_var_res[meas_var] for meas_var in self.reader.meas_var_res.keys()],  # type: ignore[attr-defined]
            "geometry_ids": [
                str(self.reader.meas_var_res[meas_var]) + "m"  # type: ignore[attr-defined]
                for meas_var in self.reader.meas_var_res.keys()  # type: ignore[attr-defined]
            ],
            "product_geospatial_bounds": get_value(full_extracted_metadata, "Dataset_Extent"),
            "located_geometric_values": get_value(full_extracted_metadata, "Located_Geometric_Values"),
            "product_datetime": date.isoformat(),
            "description": "TBD",
            "institution": "Airbus",
            "keywords": [
                "airbus",
                "pleiades",
                "satellite",
                "remote sensing",
                "multispectral",
                "earth observation",
                "ms ort",
                "visible",
            ],
            "source": "Airbus Pleiades MS ORT Satellite Product",
            "platform_type": "satellite",
        }
        # need geospatial_bounds_crs for subsetting
        proj_crs = ""
        for crs_val in get_value(self.read_metadata_xml(), "PROJECTED_CRS_CODE").split(":")[4:]:
            if crs_val:
                proj_crs += crs_val + ":"
        basic_md["geospatial_bounds_crs"] = proj_crs[:-1]

        # Add canonical footprint from product_geospatial_bounds
        # Extract EPSG code (last numeric part) from CRS string
        epsg_code = None
        crs_parts = (proj_crs[:-1] if proj_crs else "").split(":")
        if crs_parts:
            try:
                epsg_code = int(crs_parts[-1])  # Get the last numeric part
            except (ValueError, IndexError):
                epsg_code = None

        basic_md["footprint"] = normalize_footprint(
            geometry_input=basic_md["product_geospatial_bounds"],
            crs_input=epsg_code,
        )

        return basic_md

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the dataset.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
        """
        var_md: dict = {}

        # Observation geometry is typically at 2m resolution
        if var == "observation_geometry":
            var_md = {
                "spatial_resolution": 2,
                "spatial_resolution_units": "m",
                "geometry_id": "2m",
            }
            return var_md

        band_rel_dict = {  # B1 is B0 in metadata, etc
            "B1": "B0",
            "B2": "B1",
            "B3": "B2",
            "B4": "B3",
        }

        band_names_dict = {
            "B1": "Blue",  # BLUE_CHANNEL
            "B2": "Green",  # GREEN_CHANNEL
            "B3": "Red",  # RED_CHANNEL
            "B4": "Near-Infrared",  # NIR / ALPHA_CHANNEL
        }

        band_metadata_dicts = get_value(self.read_metadata_xml(), "Band_Measurement_List")

        for i in range(len(get_value(band_metadata_dicts, "Band_Spectral_Range"))):
            if band_rel_dict[var] == get_value(band_metadata_dicts, "Band_Spectral_Range")[i].get("BAND_ID"):
                var_md = {
                    "band_id": var,
                    "band_name": band_names_dict[var],
                    "band_spectral_range": (
                        band_metadata_dicts["Band_Spectral_Range"][i]["MIN"],
                        band_metadata_dicts["Band_Spectral_Range"][i]["MAX"],
                    ),
                    "band_gain": band_metadata_dicts["Band_Radiance"][i]["GAIN"],
                    "band_bias": band_metadata_dicts["Band_Radiance"][i]["BIAS"],
                    "band_solar_irradiance": band_metadata_dicts["Band_Solar_Irradiance"][i]["VALUE"],
                    "standard_name": "toa_radiance",
                    "long_name": f"TOA radiance in band {var}",
                    "units": "W/( m² * sr * μm)",
                    "band_central_wavelength": (
                        float(band_metadata_dicts["Band_Spectral_Range"][i]["MAX"])
                        + float(band_metadata_dicts["Band_Spectral_Range"][i]["MIN"]) / 2
                    )
                    * 1000,
                    "band_central_wavelength_units": "nm",
                    "spatial_resolution": self.reader.meas_var_res[var],  # type: ignore[attr-defined]
                    "spatial_resolution_units": "m",
                    "geometry_id": str(self.reader.meas_var_res[var]) + "m",  # type: ignore[attr-defined]
                    "ancillary_variables": [],
                    "measurand": "radiance",
                }

        return var_md

    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Extract variable metadata from the dataset.

        :param var: Variable name.
        :returns: Variable metadata dictionary.
        """
        basic_var_md: dict = {}

        # aux metadata done within aux module so pass here
        if var == "observation_geometry":
            return basic_var_md

        band_rel_dict = {  # B1 is B0 in metadata, etc
            "B1": "B0",
            "B2": "B1",
            "B3": "B2",
            "B4": "B3",
        }

        band_metadata_dicts = get_value(self.read_metadata_xml(), "Band_Measurement_List")

        for i in range(len(get_value(band_metadata_dicts, "Band_Spectral_Range"))):
            if band_rel_dict[var] == get_value(band_metadata_dicts, "Band_Spectral_Range")[i].get("BAND_ID"):
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
                "geometry_id": "2m",
                "spatial_resolution": 2,
                "spatial_resolution_units": "m",
                "description": "Solar zenith angle is the the angle between the line of sight to the sun and the local vertical.",
                "source_variable_name": "90 - SUN_ELEVATION",
            },
            "solar_azimuth_angle": {
                "units": "degrees",
                "long_name": "Solar Azimuth Angle",
                "standard_name": "solar_azimuth_angle",
                "measurand": "angle",
                "geometry_id": "2m",
                "spatial_resolution": 2,
                "spatial_resolution_units": "m",
                "description": "Solar azimuth angle is the horizontal angle between the line of sight to the sun and a reference direction which is often due north. The angle is measured clockwise.",
                "source_variable_name": "SUN_AZIMUTH",
            },
            "sensor_zenith_angle": {
                "units": "degrees",
                "long_name": "Viewing Zenith Angle",
                "standard_name": "sensor_zenith_angle",
                "measurand": "angle",
                "geometry_id": "2m",
                "spatial_resolution": 2,
                "spatial_resolution_units": "m",
                "description": "Viewing zenith angle is the angle between the line of sight to the sensor and the local zenith at the observation target. This angle is measured starting from directly overhead and its range is from zero (directly overhead the observation target) to 180 degrees (directly below the observation target). Local zenith is a line perpendicular to the Earth's surface at a given location. 'Observation target' means a location on the Earth defined by the sensor performing the observations.",
                "source_variable_name": "VIEWING_ANGLE",
            },
            "sensor_azimuth_angle": {
                "units": "degrees",
                "long_name": "Viewing Azimuth Angle",
                "standard_name": "sensor_azimuth_angle",
                "measurand": "angle",
                "geometry_id": "2m",
                "spatial_resolution": 2,
                "spatial_resolution_units": "m",
                "description": "Viewing azimuth angle is the horizontal angle between the line of sight from the observation point to the sensor and a reference direction at the observation point, which is often due north. The angle is measured clockwise positive, starting from the reference direction.",
                "source_variable_name": "AZIMUTH_ANGLE",
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
