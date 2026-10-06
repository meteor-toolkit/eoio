"""eoio.readers.planetscope.tests.test_metadata - unit tests for eoio.readers.planetscope.metadata"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import json

from processor_tools.utils.dict_tools import get_value

from eoio.readers.planetscope.metadata import (
    PlanetScopeMetadataExtractor,
    PlanetScopeMetadataExtractorError,
)


class TestPlanetScopeMetadataExtractor(unittest.TestCase):
    """Unit tests for PlanetScopeMetadataExtractor class."""

    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        self._create_test_metadata_files()
        self._create_mock_reader()

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def _create_test_metadata_files(self) -> None:
        """Create minimal test metadata files."""
        # Create xml metadata file
        xml_content = """<?xml version="1.0"?>
<root>
    <eop:acquisitionDate>2022-04-01T08:47:07+00:00</eop:acquisitionDate>
    <eop:productType>AnalyticMS</eop:productType>
    <ps:bandSpecificMetadata>
        <ps:bandNumber>1</ps:bandNumber>
        <!-- Multiply by radiometricScaleFactor to convert DNs to TOA Radiance (watts per steradian per square metre -->
        <ps:radiometricScaleFactor>0.01</ps:radiometricScaleFactor>
        <!-- Multiply by reflectanceCoefficient to convert DNs to TOA Reflectance -->
        <ps:reflectanceCoefficient>2.2782100354566955e-05</ps:reflectanceCoefficient>
      </ps:bandSpecificMetadata>
    <ps:bandSpecificMetadata>
        <ps:bandNumber>2</ps:bandNumber>
        <!-- Multiply by radiometricScaleFactor to convert DNs to TOA Radiance (watts per steradian per square metre -->
        <ps:radiometricScaleFactor>0.01</ps:radiometricScaleFactor>
        <!-- Multiply by reflectanceCoefficient to convert DNs to TOA Reflectance -->
        <ps:reflectanceCoefficient>2.2002063959589947e-05</ps:reflectanceCoefficient>
      </ps:bandSpecificMetadata>
    <ps:bandSpecificMetadata>
        <ps:bandNumber>3</ps:bandNumber>
        <!-- Multiply by radiometricScaleFactor to convert DNs to TOA Radiance (watts per steradian per square metre -->
        <ps:radiometricScaleFactor>0.01</ps:radiometricScaleFactor>
        <!-- Multiply by reflectanceCoefficient to convert DNs to TOA Reflectance -->
        <ps:reflectanceCoefficient>2.3515125126176514e-05</ps:reflectanceCoefficient>
      </ps:bandSpecificMetadata>
    <ps:bandSpecificMetadata>
        <ps:bandNumber>4</ps:bandNumber>
      </ps:bandSpecificMetadata>
    <ps:bandSpecificMetadata>
        <ps:bandNumber>5</ps:bandNumber>
      </ps:bandSpecificMetadata>
    <ps:bandSpecificMetadata>
        <ps:bandNumber>7</ps:bandNumber>
      </ps:bandSpecificMetadata>
    <ps:bandSpecificMetadata>
        <ps:bandNumber>8</ps:bandNumber>
      </ps:bandSpecificMetadata>

</root>
"""
        xml_file = self.temp_dir / "metadata.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        # Create json metadata file 1 (main metadata)
        json_data_1 = {
            "id": "20220401_084707_00_2477",
            "type": "Feature",
            "constellation": "planetscope",
            "platform": "240a",
            "instruments": ["PSB.SD"],
            "bbox": [15.10274, -23.60723694, 15.13462891, -23.59451068],
            "eo:bands": [
                {
                    "name": "Coastal Blue",
                    "common_name": "coastal",
                    "center_wavelength": 0.442,
                    "full_width_half_max": 0.021,
                },
                {
                    "name": "Blue",
                    "common_name": "blue",
                    "center_wavelength": 0.49,
                    "full_width_half_max": 0.05,
                },
                {
                    "name": "Green I",
                    "center_wavelength": 0.531,
                    "full_width_half_max": 0.036,
                },
                {
                    "name": "Green",
                    "common_name": "green",
                    "center_wavelength": 0.565,
                    "full_width_half_max": 0.036,
                },
                {
                    "name": "Yellow",
                    "common_name": "yellow",
                    "center_wavelength": 0.61,
                    "full_width_half_max": 0.02,
                },
                {
                    "name": "Red",
                    "common_name": "red",
                    "center_wavelength": 0.665,
                    "full_width_half_max": 0.03,
                },
                {
                    "name": "Red Edge",
                    "common_name": "rededge",
                    "center_wavelength": 0.705,
                    "full_width_half_max": 0.016,
                },
                {
                    "name": "Near-Infrared",
                    "common_name": "nir",
                    "center_wavelength": 0.865,
                    "full_width_half_max": 0.04,
                },
            ],
            "proj:epsg": 32733,
        }
        json_file_1 = self.temp_dir / "metadata_1.json"
        json_file_1.write_text(json.dumps(json_data_1), encoding="utf-8")

        # Create json metadata file 2 (properties)
        json_data_2 = {
            "type": "Feature",
            "properties": {
                "acquired": "2022-04-01T08:47:07.003186Z",
                "instrument": "PSB.SD",
                "satellite_azimuth": 282.2,
                "satellite_id": "2477",
                "sun_azimuth": 54.2,
                "sun_elevation": 46.4,
            },
        }
        json_file_2 = self.temp_dir / "metadata_2.json"
        json_file_2.write_text(json.dumps(json_data_2), encoding="utf-8")

        self.xml_file = str(xml_file)
        self.json_file_1 = str(json_file_1)
        self.json_file_2 = str(json_file_2)

    def _create_mock_reader(self) -> None:
        """Create a mock reader object."""
        self.reader = Mock(name="reader")
        self.reader.layout = Mock(name="layout")
        self.reader.layout.metadata_files = Mock(return_value=[self.xml_file, self.json_file_1, self.json_file_2])
        self.reader.meas_var_res = {
            "B1": 3,
            "B2": 3,
            "B3": 3,
            "B4": 3,
            "B5": 3,
            "B6": 3,
            "B7": 3,
            "B8": 3,
        }
        self.reader.resolved_config = Mock(name="resolved_config")
        self.reader.resolved_config.vars_sel = {"meas": ["B1", "B2"]}

    def test_init_stores_metadata_filepaths(self) -> None:
        """Test that __init__ stores metadata filepaths from layout."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        self.assertIsNotNone(extractor.metadata_filepaths)
        self.assertEqual(len(extractor.metadata_filepaths), 3)

    def test_read_metadata_files_returns_dict_including_both_types_of_metadata_file(
        self,
    ) -> None:
        """Test that read_metadata_files returns a dictionary with both json and xml files."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.read_metadata_files()
        self.assertIsInstance(metadata, dict)
        # xml data should be in metadata
        self.assertIn("2022-04-01T08:47:07+00:00", get_value(metadata, "eop:acquisitionDate"))
        # json data should be in metadata
        self.assertIn("2477", get_value(metadata, "id"))
        self.assertIsInstance(get_value(metadata, "properties"), dict)

    def test_read_metadata_files_raises_on_invalid_format(self) -> None:
        """Test that read_metadata_files raises on unsupported format."""
        # Create invalid metadata file
        invalid_file = self.temp_dir / "invalid.txt"
        invalid_file.write_text("invalid content", encoding="utf-8")

        self.reader.layout.metadata_files = Mock(return_value=[str(invalid_file)])

        extractor = PlanetScopeMetadataExtractor(self.reader)

        with self.assertRaises(PlanetScopeMetadataExtractorError):
            extractor.read_metadata_files()

    def test_get_basic_metadata_returns_dict(self) -> None:
        """Test that get_basic_metadata returns a dictionary."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_basic_metadata()
        self.assertIsInstance(metadata, dict)

    def test_get_basic_metadata_contains_collection_name(self) -> None:
        """Test that basic metadata contains collection_name correctly and contains other basic attrs."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_basic_metadata()
        self.assertIn("collection_name", metadata)
        self.assertEqual(metadata["collection_name"], "planetscope")
        self.assertIn("product_name", metadata)
        self.assertIn("platform", metadata)
        self.assertIn("processing_level", metadata)
        self.assertIn("product_datetime", metadata)
        self.assertEqual(metadata["product_datetime"], "2022-04-01T08:47:07")
        self.assertIn("footprint", metadata)
        self.assertIsInstance(metadata["footprint"], dict)
        self.assertTrue(metadata["footprint"]["crs"].startswith("EPSG:"))

    def test_get_product_metadata_returns_dict(self) -> None:
        """Test that get_product_metadata returns a dictionary."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_product_metadata()
        self.assertIsInstance(metadata, dict)

    def test_get_product_metadata_contains_crs(self) -> None:
        """Test that product metadata contains geospatial_bounds_crs and product_properties."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_product_metadata()
        self.assertIn("geospatial_bounds_crs", metadata)
        self.assertTrue(metadata["geospatial_bounds_crs"].startswith("EPSG:"))
        self.assertIn("product_properties", metadata)

    def test_get_variable_basic_metadata_returns_dict(self) -> None:
        """Test that get_variable_basic_metadata returns a dictionary."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_variable_basic_metadata("B1")
        self.assertIsInstance(metadata, dict)

    def test_get_variable_basic_metadata_contains_band_info(self) -> None:
        """Test that variable metadata contains band information and other attrs."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_variable_basic_metadata("B1")
        self.assertIn("long_name", metadata)
        self.assertIn("units", metadata)
        self.assertIn("standard_name", metadata)

    def test_variable_product_metadata_reports_wavelengths_in_nm(self) -> None:
        """The STAC eo:bands wavelengths are micrometres; they must be reported in nm."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_variable_product_metadata("B1")
        self.assertEqual(metadata["band_central_wavelength"], 442.0)
        self.assertEqual(metadata["band_central_wavelength_units"], "nm")
        self.assertEqual(metadata["band_full_width_half_max"], 21.0)

    def test_get_variable_product_metadata_returns_dict(self) -> None:
        """Test that get_variable_product_metadata returns a dictionary."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_variable_product_metadata("B1")
        self.assertIsInstance(metadata, dict)

    def test_get_variable_product_metadata_observation_geometry_has_spatial_metadata(self) -> None:
        """Test that observation_geometry auxiliary variable includes spatial metadata."""
        extractor = PlanetScopeMetadataExtractor(self.reader)
        metadata = extractor.get_variable_product_metadata("observation_geometry")
        self.assertIn("spatial_resolution", metadata)
        self.assertEqual(metadata["spatial_resolution"], 3, "observation_geometry should have 3m resolution")
        self.assertEqual(metadata["spatial_resolution_units"], "m")
        self.assertEqual(metadata["geometry_id"], "3m")


if __name__ == "__main__":
    unittest.main()
