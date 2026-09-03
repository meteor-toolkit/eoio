"""
eoio.readers.landsat.metadata.extractor - Metadata extraction for Landsat products.
"""

from __future__ import annotations

from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.landsat.metadata.ls_mtd_xml import LSL1ProdXMLReader
from eoio.readers.landsat.metadata.ls_mtd_json import LSL1ProdJSONReader
from eoio.readers.footprint_utils import normalize_footprint

MEAS_VAR_BAND_IDS = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B9": 8,
    "B10": 9,
    "B11": 10,
}


class LSMetadataExtractor(BaseMetadataExtractor):
    """
    High-level metadata extractor for Landsat 8/9 products.

    This class extracts metadata from Landsat MTL XML file via LSProdXMLReader
    and exposes structured dictionaries suitable for downstream use.
    """

    def __init__(self, reader):
        """
        Initialise the Landsat metadata extractor.

        :param reader: eoio Landsat reader instance providing layout/config.
        """
        super().__init__(reader)

        # define attributes
        self.layout = reader.layout

        # Find and initialize the XML reader
        self.xml_reader = LSL1ProdXMLReader(self.layout.product_metadata_xml())

        # Find and initialize the JSON reader (STAC preferred)
        self.json_reader = LSL1ProdJSONReader(self.layout.product_metadata_json())

        # Cache JSON-derived band metadata to minimize repeated lookup
        self._band_wavelengths = self.json_reader.find_all_band_central_wavelengths()
        self._band_gsds = self.json_reader.find_all_band_gsds()

    def get_basic_metadata(self) -> dict:
        """
        Return high-level, product-wide metadata shared across variables.

        :returns: Dictionary of basic product metadata.
        """
        res = [
            int(self._band_gsds.get(band)) for band in MEAS_VAR_BAND_IDS.keys() if self._band_gsds.get(band) is not None
        ]
        geoms = [f"{r}m" for r in res]

        basic_md = {
            "collection_name": "LANDSAT_C2L1",
            "product_name": self.xml_reader.find_product_id(),
            "platform": self.xml_reader.find_spacecraft_name(),
            "instrument": self.xml_reader.find_sensor_id(),
            "processing_level": self.xml_reader.find_processing_level(),
            "processing_version": "TBD",  # self.xml_reader.find_collection_number(),
            "spatial_resolution": res,
            "spatial_resolution_units": "m",
            "geometry_ids": geoms,
            "product_bounds": self.xml_reader.find_bounds(),
            "product_datetime": self.xml_reader.find_acquisition_datetime().isoformat(),
            "description": "TBD",
            "epsg": self.json_reader.find_epsg(),
            "institution": "NASA / USGS",
            "keywords": [
                "Landsat",
                "oli",
                "tirs",
                "satellite",
                "remote sensing",
                "multispectral",
                "earth observation",
                "level 1",
                "visible",
                "thermal",
            ],
            "source": "Landsat 8/9 Collection 2 Level-1 Satellite Product",
            "platform_type": "satellite",
            "footprint": normalize_footprint(
                geometry_input=self.xml_reader.find_bounds(),
                crs_input=self.json_reader.find_epsg(),
            ),
        }

        return basic_md

    def get_product_metadata(self) -> dict:
        """
        Return detailed product-level metadata (acquisition, quality, calibration).

        :returns: Dictionary of detailed product metadata.
        """
        prod_md = {
            "collection_number": self.xml_reader.find_collection_number(),
            "collection_category": self.xml_reader.find_collection_category(),
            "scene_center_time": self.xml_reader.find_value("scene_center_time", default=""),
            "sun_elevation": self.xml_reader.find_value("sun_elevation", default=None),
            "sun_azimuth": self.xml_reader.find_value("sun_azimuth", default=None),
            "cloud_area_fraction": self.xml_reader.find_value("cloud_cover", default=None),
            "cloud_area_fraction_land": self.xml_reader.find_value("cloud_cover_land", default=None),
            "earth_sun_distance": self.xml_reader.find_value("earth_sun_distance", default=None),
            "wrs_path": self.xml_reader.find_wrs_path(),
            "wrs_row": self.xml_reader.find_wrs_row(),
            "map_projection": self.xml_reader.find_value("map_projection", default=""),
            "utm_zone": self.xml_reader.find_value("utm_zone", default=None),
            "date_product_generated": self.xml_reader.find_value("date_product_generated", default=""),
            "acquisition_datetime": self.xml_reader.find_acquisition_datetime(),
            "processing_software_version": self.xml_reader.find_value("processing_software_version", default=""),
            "image_quality_oli": self.xml_reader.find_value("image_quality_oli", default=None),
            "image_quality_tirs": self.xml_reader.find_value("image_quality_tirs", default=None),
            # Optional aggregation of projection parameters (XML)
            "projection_parameters": self.xml_reader.find_projection_parameters(),
            # JSON-derived projection metadata (STAC)
            "proj_shape": self.json_reader.find_proj_shape(),
            "proj_transform": self.json_reader.find_proj_transform(),
        }

        return prod_md

    # ---------------------------------------------------------------------
    # Variable-level metadata
    # ---------------------------------------------------------------------
    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Return basic descriptive metadata for a dataset variable (band).

        :param var: Variable name for which metadata should be extracted.
        :returns: Dictionary containing basic variable metadata.
        """
        if var in MEAS_VAR_BAND_IDS:
            band_id = MEAS_VAR_BAND_IDS[var]
            return self._get_band_variable_basic_metadata(var, band_id)
        elif var in self.layout.get_angle_files().keys():
            return self.get_angle_metadata().get(var, {})
        else:
            raise KeyError(f"Unknown Landsat variable: {var}")

    def _get_band_variable_basic_metadata(self, var: str, band_id: int) -> dict:
        # Simple units convention: reflective bands -> DN, thermal -> radiance units
        reflective = {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"}
        units = "1" if var in reflective else "W/(m^2 sr μm)"
        return {
            "units": units,
            "long_name": (
                "Top of atmosphere hemispherical conical reflectance"
                if var in reflective
                else "Top of atmosphere radiance"
            ),
            "standard_name": ("toa_reflectance" if var in reflective else "toa_radiance"),
            "measurand": "reflectance" if var in reflective else "radiance",
        }

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Return detailed metadata for a specific Landsat band variable:
        spectral characteristics, radiometric calibration coefficients,
        and thermal constants (for B10/B11).
        """
        if var in MEAS_VAR_BAND_IDS:
            band_id = MEAS_VAR_BAND_IDS[var]
            return self._get_band_variable_metadata(var, band_id)
        elif var in self.layout.get_angle_files().keys():
            return self.get_angle_metadata().get(var, {})
        elif var in self.get_aux_metadata().keys():
            return self.get_aux_metadata().get(var, {})
        else:
            raise KeyError(f"Unknown Landsat variable: {var}")

    def _get_band_variable_metadata(self, var: str, band_id: int) -> dict:
        # Radiometric rescaling for this band (XML)
        rescaling = self.xml_reader.get_radiometric_rescaling()
        band_rescaling = rescaling.get(var, {})

        # GSD per band (JSON) — convert to int so resolution values and dimension
        # names are integers everywhere ("30m" not "30.0m", 30 not 30.0)
        res = int(self._band_gsds.get(var))
        geom = f"{res}m"

        # Thermal constants if applicable (XML)
        thermal_constants = {}
        if var in ["B10", "B11"]:
            thermal = self.xml_reader.get_thermal_constants()
            thermal_constants = thermal.get(var, {})

        variable_mtd = {
            "band_name": self._get_band_name(var),
            "band_central_wavelength": self._band_wavelengths.get(var),
            "band_central_wavelength_units": "nm",
            "band_id": band_id,
            "spatial_resolution": res,  # e.g. 30
            "spatial_resolution_units": "m",
            "geometry_id": geom,  # e.g. "30m"
            "radiance_mult": band_rescaling.get("radiance_mult"),
            "radiance_add": band_rescaling.get("radiance_add"),
            "reflectance_mult": band_rescaling.get("reflectance_mult"),
            "reflectance_add": band_rescaling.get("reflectance_add"),
            "ancillary_variables": "",
        }

        if thermal_constants:
            variable_mtd.update(thermal_constants)

        return variable_mtd

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _get_band_name(self, var: str) -> str:
        """
        Get the descriptive name for a Landsat band.
        """
        band_names = {
            "B1": "Coastal/Aerosol",
            "B2": "Blue",
            "B3": "Green",
            "B4": "Red",
            "B5": "Near Infrared",
            "B6": "SWIR 1",
            "B7": "SWIR 2",
            "B8": "Panchromatic",
            "B9": "Cirrus",
            "B10": "Thermal Infrared 1",
            "B11": "Thermal Infrared 2",
        }
        return band_names.get(var, var)

    def get_angle_metadata(self) -> dict:
        """
        Return metadata for angle variables (solar/observer angles).

        :returns: Dictionary of angle variable metadata.
        """
        # Each angle has a unique measurand so the stack processor never groups
        # multiple angles together (group size < 2 → skipped).  Using a shared
        # "angle" measurand would cause all four to be stacked into a single cube
        # on the same band_30m dimension as the reflectance bands, which corrupts
        # the reflectance cube via xarray coordinate alignment.
        angle_vars = {
            "solar_zenith_angle": {
                "units": "degrees",
                "long_name": "Solar Zenith Angle",
                "standard_name": "solar_zenith_angle",
                "measurand": "solar_zenith_angle",
                "geometry_id": "30m",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "description": "Solar zenith angle is the the angle between the line of sight to the sun and the local vertical.",
                "source_variable_name": "SZA",
            },
            "solar_azimuth_angle": {
                "units": "degrees",
                "long_name": "Solar Azimuth Angle",
                "standard_name": "solar_azimuth_angle",
                "measurand": "solar_azimuth_angle",
                "geometry_id": "30m",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "description": "Solar azimuth angle is the horizontal angle between the line of sight to the sun and a reference direction which is often due north. The angle is measured clockwise.",
                "source_variable_name": "SAA",
            },
            "viewing_zenith_angle": {
                "units": "degrees",
                "long_name": "Viewing Zenith Angle",
                "standard_name": "sensor_zenith_angle",
                "measurand": "viewing_zenith_angle",
                "geometry_id": "30m",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "description": "Viewing zenith angle is the angle between the line of sight to the sensor and the local zenith at the observation target. This angle is measured starting from directly overhead and its range is from zero (directly overhead the observation target) to 180 degrees (directly below the observation target). Local zenith is a line perpendicular to the Earth's surface at a given location. 'Observation target' means a location on the Earth defined by the sensor performing the observations.",
                "source_variable_name": "VZA",
            },
            "viewing_azimuth_angle": {
                "units": "degrees",
                "long_name": "Viewing Azimuth Angle",
                "standard_name": "sensor_azimuth_angle",
                "measurand": "viewing_azimuth_angle",
                "geometry_id": "30m",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "description": "Viewing azimuth angle is the horizontal angle between the line of sight from the observation point to the sensor and a reference direction at the observation point, which is often due north. The angle is measured clockwise positive, starting from the reference direction.",
                "source_variable_name": "VAA",
            },
        }

        return angle_vars

    def get_aux_metadata(self) -> dict:
        """
        Return metadata for auxiliary data variables.

        :returns: Dictionary of auxiliary data variable metadata.
        """
        # Landsat auxiliary products are typically 30m resolution
        aux_vars = {
            "ATRAN": {
                "units": None,
                "standard_name": "Atmospheric Transmittance Layer",
                "description": "The ratio of the transmitted radiation to the total radiation incident upon the medium (atmosphere).",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "geometry_id": "30m",
            },
            "CDIST": {
                "units": "km",
                "standard_name": "Distance to Cloud",
                "description": "The distance that a pixel is from the nearest cloud pixel.",
                "notes": "Infrequently the pixel distance will be greater than the maximum allowed value. This layer "
                "is used with emissivity standard deviation to create surface temperature QA.",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "geometry_id": "30m",
            },
            "DRAD": {
                "units": "W/(m^2 sr µm)/DN",
                "standard_name": "Downwelled Radiance Layer",
                "description": "The thermal energy emitted by the atmosphere that reaches the Earth's surface and is "
                "then reflected toward the sensor.",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "geometry_id": "30m",
            },
            "EMIS": {
                "units": "emissivity coefficient",
                "standard_name": "Emissivity Layer",
                "description": "the ratio of the energy radiated from a material's surface to the energy radiated "
                "from a blackbody.",
                "notes": "Landsat emissivity values that are greater than the water emissivity constance are adjusted "
                "to be the water constant (0.988). Negative values for emissivity are replaced with the fill "
                "value instead.",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "geometry_id": "30m",
            },
            "EMSD": {
                "units": "emissivity coefficient",
                "standard_name": "Emissivity Standard Deviation",
                "description": "The extent of variation for the emissivity product.",
                "notes": "This layer is used with CDIST to create ST_QA.",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "geometry_id": "30m",
            },
            "TRAD": {
                "units": "W/(m^2 sr µm)/DN",
                "standard_name": "Thermal Radiance Layer",
                "description": "The values produced when L1's TIR Band 10 is converted to radiance.",
                "notes": "The maximum value for thermal radiance, 22000 Wm-2sr-1µm-1, may be exceeded (e.g.,"
                "over volcanoes and fires). TRAD is generated so all the radiance layers share the same "
                "units. ",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "geometry_id": "30m",
            },
            "URAD": {
                "units": "W/(m^2 sr µm)/DN",
                "standard_name": "Upwelled Radiance Layer",
                "description": "The amount of energy emitted from the atmosphere and scattered toward the sensor.",
                "spatial_resolution": 30,
                "spatial_resolution_units": "m",
                "geometry_id": "30m",
            },
        }

        return aux_vars


if __name__ == "__main__":
    pass
