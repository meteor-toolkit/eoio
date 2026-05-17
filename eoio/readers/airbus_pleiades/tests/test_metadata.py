"""
eoio.readers.airbus_pleiades.tests.test_metadata - unit tests for eoio.readers.airbus_pleiades.metadata
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from eoio.readers.airbus_pleiades.metadata import PleiadesMetadataExtractor


class TestPleiadesMetadataExtractor(unittest.TestCase):
    """Unit tests for PleiadesMetadataExtractor class."""

    def _make_minimal_xml(self, root: Path) -> Path:
        """Create a minimal valid Pleiades metadata XML file."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Dimap_Document>
    <Metadata_Identification>
        <MISSION>PHR</MISSION>
        <MISSION_INDEX>1A</MISSION_INDEX>
        <INSTRUMENT>PHR</INSTRUMENT>
        <INSTRUMENT_INDEX>1A</INSTRUMENT_INDEX>
        <PROCESSING_LEVEL>ORT</PROCESSING_LEVEL>
        <SPECTRAL_PROCESSING>MS</SPECTRAL_PROCESSING>
    </Metadata_Identification>
    <Product_Characteristics>
        <PRODUCT_CODE>PHR</PRODUCT_CODE>
        <IMAGING_DATE>2023-01-15</IMAGING_DATE>
        <IMAGING_TIME>10:30:45.123456Z</IMAGING_TIME>
        <DATA_FILE_PATH>/path/to/data</DATA_FILE_PATH>
    </Product_Characteristics>
    <Geometric_Information>
        <PROJECTED_CRS_CODE>TEST:TEST:TEST:EPSG::32633</PROJECTED_CRS_CODE>
    </Geometric_Information>
    <Quality_Assessment>
        <CLOUD_COVERAGE unit="percent">0</CLOUD_COVERAGE>
    </Quality_Assessment>
    <Located_Geometric_Values>
        <Solar_Incidences>
            <SUN_AZIMUTH unit="deg">95.07627858043004</SUN_AZIMUTH>
            <SUN_ELEVATION unit="deg">66.85648966176281</SUN_ELEVATION>
        </Solar_Incidences>
    </Located_Geometric_Values>
    <Band_Measurement_List>
        <Band_Spectral_Range>
            <BAND_ID>B0</BAND_ID>
            <MIN>450</MIN>
            <MAX>520</MAX>
        </Band_Spectral_Range>
        <Band_Spectral_Range>
            <BAND_ID>B1</BAND_ID>
            <MIN>520</MIN>
            <MAX>590</MAX>
        </Band_Spectral_Range>
        <Band_Spectral_Range>
            <BAND_ID>B2</BAND_ID>
            <MIN>630</MIN>
            <MAX>690</MAX>
        </Band_Spectral_Range>
        <Band_Spectral_Range>
            <BAND_ID>B3</BAND_ID>
            <MIN>760</MIN>
            <MAX>890</MAX>
        </Band_Spectral_Range>
        <Band_Radiance>
            <GAIN>0.1</GAIN>
            <BIAS>0.0</BIAS>
        </Band_Radiance>
        <Band_Radiance>
            <GAIN>0.1</GAIN>
            <BIAS>0.0</BIAS>
        </Band_Radiance>
        <Band_Radiance>
            <GAIN>0.1</GAIN>
            <BIAS>0.0</BIAS>
        </Band_Radiance>
        <Band_Radiance>
            <GAIN>0.1</GAIN>
            <BIAS>0.0</BIAS>
        </Band_Radiance>
        <Band_Solar_Irradiance>
            <VALUE>1920</VALUE>
        </Band_Solar_Irradiance>
        <Band_Solar_Irradiance>
            <VALUE>1820</VALUE>
        </Band_Solar_Irradiance>
        <Band_Solar_Irradiance>
            <VALUE>1610</VALUE>
        </Band_Solar_Irradiance>
        <Band_Solar_Irradiance>
            <VALUE>970</VALUE>
        </Band_Solar_Irradiance>
    </Band_Measurement_List>
    <Dataset_Extent>
        <ULX>0</ULX>
        <ULY>0</ULY>
    </Dataset_Extent>
</Dimap_Document>
"""
        xml_path = root / "DIM_metadata.xml"
        xml_path.write_text(xml_content, encoding="utf-8")
        return xml_path

    def _make_mock_reader(self, xml_path: Path):
        """Create a mock reader object with required attributes."""
        reader = Mock()
        reader.layout = Mock()
        reader.layout.metadata_file = Mock(return_value=str(xml_path))
        reader.meas_var_res = {"B1": 2, "B2": 2, "B3": 2, "B4": 2}
        reader.resolved_config = Mock()
        reader.resolved_config.vars_sel = {"meas": ["B1", "B2", "B3", "B4"]}
        return reader

    def test_read_metadata_xml(self):
        """Test that read_metadata_xml returns the correct dict."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.read_metadata_xml()

            self.assertIsInstance(result, dict)
            self.assertIn("Dimap_Document", result)

    def test_get_basic_metadata(self):
        """Test that get_basic_metadata returns a dict with expected keys."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_basic_metadata()

            self.assertIsInstance(result, dict)
            expected_keys = [
                "collection_name",
                "product_name",
                "platform",
                "instrument",
                "processing_level",
                "satellite_id",
                "constellation",
                "spatial_resolution",
                "geometry_ids",
                "product_geospatial_bounds",
                "product_date",
                "description",
                "footprint",
            ]
            for key in expected_keys:
                self.assertIn(key, result)
            self.assertIsInstance(result["footprint"], dict)
            self.assertEqual(result["footprint"].get("crs"), "EPSG:32633")

    def test_get_basic_metadata_correct_values(self):
        """Test that get_basic_metadata extracts correct values."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_basic_metadata()

            self.assertEqual(result["collection_name"], "PHR")
            self.assertEqual(result["platform"], "PHR_1A")
            self.assertEqual(result["instrument"], "PHR_1A")
            self.assertEqual(result["processing_level"], "ORT_MS")
            self.assertEqual(result["satellite_id"], "PHR_1A")

    def test_get_product_metadata(self):
        """Test that get_product_metadata returns a dict."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_product_metadata()

            self.assertIsInstance(result, dict)

    def test_get_product_metadata_contains_basic_metadata(self):
        """Test that get_product_metadata contains basic metadata keys."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_product_metadata()

            self.assertIn("collection_name", result)
            self.assertIn("platform", result)

    def test_get_product_metadata_contains_cloud_area_fraction(self):
        """Test that get_product_metadata contains cloud_area_fraction."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_product_metadata()

            self.assertIn("cloud_area_fraction", result)

    def test_get_product_metadata_contains_geospatial_bounds_crs(self):
        """Test that get_product_metadata contains geospatial_bounds_crs."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_product_metadata()

            self.assertIn("geospatial_bounds_crs", result)
            self.assertEqual(result["geospatial_bounds_crs"], "32633")

    def test_get_variable_product_metadata_band_info(self):
        """Test that band metadata returns a dict containing expected keys."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_variable_product_metadata("B1")

            self.assertIsInstance(result, dict)

            self.assertIn("band_id", result)
            self.assertIn("band_name", result)
            self.assertIn("band_spectral_range", result)
            self.assertIn("band_gain", result)
            self.assertIn("band_bias", result)
            self.assertIn("band_solar_irradiance", result)

    def test_get_variable_product_metadata_band_names(self):
        """Test that band names are correctly assigned."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)

            self.assertEqual(extractor.get_variable_product_metadata("B1")["band_name"], "Blue")
            self.assertEqual(extractor.get_variable_product_metadata("B2")["band_name"], "Green")
            self.assertEqual(extractor.get_variable_product_metadata("B3")["band_name"], "Red")
            self.assertEqual(
                extractor.get_variable_product_metadata("B4")["band_name"],
                "Near-Infrared",
            )

    def test_get_variable_product_metadata_observation_geometry_has_spatial_metadata(self):
        """Test that observation_geometry auxiliary variable includes spatial metadata."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            md = extractor.get_variable_product_metadata("observation_geometry")

            self.assertIn("spatial_resolution", md)
            self.assertEqual(md["spatial_resolution"], 2, "observation_geometry should have 2m resolution")
            self.assertEqual(md["spatial_resolution_units"], "m")
            self.assertEqual(md["geometry_id"], "2m")

    def test_get_variable_basic_metadata_contains_required_keys(self):
        """Test that get_variable_basic_metadata returns a dict containing required keys."""
        with TemporaryDirectory() as td:
            root = Path(td)
            xml_path = self._make_minimal_xml(root)
            reader = self._make_mock_reader(xml_path)

            extractor = PleiadesMetadataExtractor(reader)
            result = extractor.get_variable_basic_metadata("B1")

            self.assertIsInstance(result, dict)

            self.assertIn("long_name", result)
            self.assertIn("standard_name", result)
            self.assertIn("units", result)


if __name__ == "__main__":
    unittest.main()
