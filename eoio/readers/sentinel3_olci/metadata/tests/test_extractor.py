"""eoio.readers.sentinel3_olci.metadata.tests.test_extractor - tests for S3OLCIMetadataExtractor."""

from __future__ import annotations
import json
import unittest
from unittest.mock import MagicMock
from pathlib import Path
import datetime as dt
from types import SimpleNamespace
import tempfile

from eoio.readers.sentinel3_olci.metadata.extractor import (
    S3OLCIMetadataExtractor,
    OLCI_BAND_IDS,
)


class TestS3OLCIMetadataExtractor(unittest.TestCase):
    """Test cases for S3OLCIMetadataExtractor."""

    def _make_reader_with_layout(self, manifest_xml_path):
        """
        Create a minimal fake EOIO Sentinel-3 OLCI reader with layout
        that returns the path to xfdumanifest.xml.
        """
        layout = SimpleNamespace(manifest_path=MagicMock(return_value=manifest_xml_path))
        resolved_config = SimpleNamespace(subset=None)
        reader = SimpleNamespace(
            layout=layout,
            resolved_config=resolved_config,
            path=Path("."),
            uncertainty_vars=[],
        )
        return reader

    def setUp(self):
        """Set up temporary directory and XML fixture."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Create a minimal xfdumanifest.xml for testing
        self.xml_text = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu"
    xmlns:safe="http://www.esa.int/safe/sentinel/1.1"
    xmlns:sentinel3="http://www.esa.int/safe/sentinel/sentinel-3/1.0"
    xmlns:olci="http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0"
    xmlns:gml="http://www.opengis.net/gml"
    version="1.0">

  <safe:acquisitionPeriod>
    <safe:startTime>2023-06-15T10:30:45.123456Z</safe:startTime>
    <safe:stopTime>2023-06-15T10:40:15.654321Z</safe:stopTime>
  </safe:acquisitionPeriod>

  <safe:platform>
    <safe:familyName>Sentinel-3</safe:familyName>
    <safe:number>A</safe:number>
  </safe:platform>

  <safe:orbitReference>
    <safe:orbitNumber type="start">10234</safe:orbitNumber>
    <safe:relativeOrbitNumber type="start">137</safe:relativeOrbitNumber>
    <safe:cycleNumber>123</safe:cycleNumber>
  </safe:orbitReference>

  <safe:frameSet>
    <safe:footPrint>
      <gml:posList>
        60.5 10.0  60.5 20.0  50.5 20.0  50.5 10.0  60.5 10.0
      </gml:posList>
    </safe:footPrint>
  </safe:frameSet>

  <sentinel3:generalProductInformation>
    <sentinel3:productName>S3A_OL_1_EFR____20230615T103045_20230615T104015_0350_078_OLCI_OBC_____LN3_O_ST_002.SEN3</sentinel3:productName>
    <sentinel3:productType>OL_1_EFR____</sentinel3:productType>
    <sentinel3:processingBaseline>02.30</sentinel3:processingBaseline>
  </sentinel3:generalProductInformation>

  <olci:olciProductInformation>
    <olci:measurementAccuracy>
      <sentinel3:relativeMeasurementAccuracy>0.015</sentinel3:relativeMeasurementAccuracy>
    </olci:measurementAccuracy>
    <olci:imageSize>
      <sentinel3:rows>3754</sentinel3:rows>
      <sentinel3:columns>5120</sentinel3:columns>
    </olci:imageSize>
  </olci:olciProductInformation>

  <olci:bandDescriptions>
    <sentinel3:band name="Oa01">
      <sentinel3:centralWavelength>400.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>15.0</sentinel3:bandwidth>
    </sentinel3:band>
    <sentinel3:band name="Oa02">
      <sentinel3:centralWavelength>410.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <sentinel3:band name="Oa03">
      <sentinel3:centralWavelength>443.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <sentinel3:band name="Oa04">
      <sentinel3:centralWavelength>490.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <sentinel3:band name="Oa05">
      <sentinel3:centralWavelength>510.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <sentinel3:band name="Oa21">
      <sentinel3:centralWavelength>1020.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>40.0</sentinel3:bandwidth>
    </sentinel3:band>
  </olci:bandDescriptions>

</xfdu:XFDU>
"""
        self.xml_file = self.tmp_path / "xfdumanifest.xml"
        self.xml_file.write_text(self.xml_text, encoding="utf-8")

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()

    # ---- Initialization and basic structure --------------------------------

    def test_init_creates_xml_reader(self):
        """Test that initialization creates an XML reader."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)
        self.assertIsNotNone(extractor.xml_reader)

    def test_init_caches_wavelengths_and_bandwidths(self):
        """Test that initialization caches spectral data."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        self.assertIsInstance(extractor._central_wavelengths, dict)
        self.assertIsInstance(extractor._bandwidths, dict)
        self.assertGreater(len(extractor._central_wavelengths), 0)
        self.assertGreater(len(extractor._bandwidths), 0)

    # ---- get_basic_metadata tests -------------------------------------------

    def test_get_basic_metadata_structure(self):
        """Test that basic metadata returns expected structure."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_basic_metadata()

        # Check that expected keys are present
        self.assertIn("collection_name", metadata)
        self.assertIn("product_name", metadata)
        self.assertIn("platform", metadata)
        self.assertIn("instrument", metadata)
        self.assertIn("processing_level", metadata)
        self.assertIn("processing_version", metadata)
        self.assertIn("spatial_resolution", metadata)
        self.assertIn("geometry_ids", metadata)
        self.assertIn("product_bounds", metadata)
        self.assertIn("product_date", metadata)
        self.assertIn("footprint", metadata)
        self.assertIsInstance(metadata["footprint"], dict)

    def test_get_basic_metadata_values(self):
        """Test that basic metadata contains correct values."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_basic_metadata()

        self.assertEqual(metadata["collection_name"], "OL_1_EFR____")
        self.assertEqual(metadata["instrument"], "OLCI")
        self.assertEqual(metadata["processing_level"], "L1")
        self.assertEqual(metadata["processing_version"], "02.30")
        self.assertEqual(metadata["spatial_resolution"], 300)
        self.assertIn("S3A_OL_1_EFR", metadata["product_name"])
        self.assertTrue(metadata["product_bounds"].startswith("POLYGON"))
        self.assertEqual(metadata["product_date"], dt.date(2023, 6, 15))
        self.assertEqual(
            metadata["product_datetime"], dt.datetime(2023, 6, 15, 10, 30, 45, 123456, tzinfo=dt.timezone.utc)
        )

    def test_get_basic_metadata_includes_reader_info(self):
        """Test that basic metadata includes reader information."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_basic_metadata()

        self.assertIn("eoio:reader", metadata)
        self.assertEqual(metadata["eoio:reader"], "sentinel3_olci")

    def test_eoio_subset_omitted_when_no_subset(self):
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_basic_metadata()

        self.assertNotIn("eoio:subset", metadata)

    def test_eoio_subset_is_valid_json_when_subset_present(self):
        """Regression test: eoio:subset used to be the raw subset dict, which isn't
        netCDF-attribute-safe (a raw dict can't be written to netCDF) and, unlike
        every other reader's eoio:subset, wasn't serialised at all."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        reader.resolved_config.subset = {"roi_crs_epsg": "EPSG:4326"}
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_basic_metadata()

        parsed = json.loads(metadata["eoio:subset"])
        self.assertEqual(parsed["roi_crs_epsg"], "EPSG:4326")

    # ---- get_product_metadata tests -----------------------------------------

    def test_get_product_metadata_structure(self):
        """Test that product metadata returns expected structure."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_product_metadata()

        # Check that expected keys are present
        self.assertIn("orbit_number", metadata)
        self.assertIn("relative_orbit_number", metadata)
        self.assertIn("orbit_direction", metadata)
        self.assertIn("cycle_number", metadata)
        self.assertIn("acquisition_start_time", metadata)
        self.assertIn("acquisition_stop_time", metadata)
        self.assertIn("measurement_accuracy", metadata)

    def test_get_product_metadata_values(self):
        """Test that product metadata contains correct values."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_product_metadata()

        self.assertEqual(metadata["orbit_number"], 10234)
        self.assertEqual(metadata["relative_orbit_number"], 137)
        self.assertEqual(metadata["cycle_number"], 123)
        self.assertIsInstance(metadata["acquisition_start_time"], dt.datetime)
        self.assertIsInstance(metadata["acquisition_stop_time"], dt.datetime)
        self.assertAlmostEqual(metadata["measurement_accuracy"], 0.015)

    # ---- get_variable_basic_metadata tests ----------------------------------

    def test_get_variable_basic_metadata_for_valid_band(self):
        """Test variable basic metadata for a valid OLCI band."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_variable_basic_metadata("Oa01")

        self.assertIn("units", metadata)
        self.assertIn("long_name", metadata)
        self.assertIn("standard_name", metadata)
        self.assertIn("description", metadata)

        self.assertEqual(metadata["units"], "W/m²/sr/nm")
        self.assertEqual(metadata["standard_name"], "toa_radiance")
        self.assertIn("Oa01", metadata["description"])

    def test_get_variable_basic_metadata_for_all_bands(self):
        """Test variable basic metadata for all 21 OLCI bands."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        for band_id in range(1, 22):
            band_name = f"Oa{band_id:02d}"
            metadata = extractor.get_variable_basic_metadata(band_name)

            self.assertEqual(metadata["units"], "W/m²/sr/nm")
            self.assertEqual(metadata["standard_name"], "toa_radiance")
            self.assertIn(f"band {band_id}", metadata["long_name"])

    def test_get_variable_basic_metadata_for_unknown_variable(self):
        """Test variable basic metadata for unknown variable."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_variable_basic_metadata("unknown_variable")

        self.assertIn("units", metadata)
        self.assertIn("long_name", metadata)
        self.assertIn("standard_name", metadata)

    # ---- get_variable_product_metadata tests --------------------------------

    def test_get_variable_product_metadata_for_valid_band(self):
        """Test variable product metadata for a valid OLCI band."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        metadata = extractor.get_variable_product_metadata("Oa02")

        self.assertIn("band_id", metadata)
        self.assertIn("band_name", metadata)
        self.assertIn("spatial_resolution", metadata)
        self.assertIn("band_central_wavelength", metadata)
        self.assertIn("band_bandwidth", metadata)

        self.assertEqual(metadata["band_id"], 2)
        self.assertEqual(metadata["band_name"], "Oa02")
        self.assertEqual(metadata["spatial_resolution"], 300)
        self.assertAlmostEqual(metadata["band_central_wavelength"], 410.0)
        self.assertAlmostEqual(metadata["band_bandwidth"], 10.0)

    def test_get_variable_product_metadata_for_invalid_band(self):
        """Test that invalid band raises KeyError."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        with self.assertRaises(KeyError):
            extractor.get_variable_product_metadata("invalid_band")

    def test_get_variable_product_metadata_multiple_bands(self):
        """Test variable product metadata for multiple OLCI bands."""
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3OLCIMetadataExtractor(reader)

        # Test bands 1, 5, and 21
        bands_to_test = [1, 5, 21]
        expected_wavelengths = [400.0, 510.0, 1020.0]
        expected_bandwidths = [15.0, 10.0, 40.0]

        for band_id, expected_wl, expected_bw in zip(bands_to_test, expected_wavelengths, expected_bandwidths):
            band_name = f"Oa{band_id:02d}"
            metadata = extractor.get_variable_product_metadata(band_name)

            self.assertEqual(metadata["band_id"], band_id)
            self.assertAlmostEqual(metadata["band_central_wavelength"], expected_wl)
            self.assertAlmostEqual(metadata["band_bandwidth"], expected_bw)

    # ---- OLCI_BAND_IDS tests ------------------------------------------------

    def test_olci_band_ids_mapping(self):
        """Test that OLCI_BAND_IDS mapping is correct."""
        # Should have entries for bands 1-21
        self.assertEqual(len(OLCI_BAND_IDS), 21)

        # Test a few specific mappings
        self.assertEqual(OLCI_BAND_IDS["Oa01"], 1)
        self.assertEqual(OLCI_BAND_IDS["Oa02"], 2)
        self.assertEqual(OLCI_BAND_IDS["Oa21"], 21)

        # Test all bands are present
        for band_id in range(1, 22):
            band_name = f"Oa{band_id:02d}"
            self.assertIn(band_name, OLCI_BAND_IDS)
            self.assertEqual(OLCI_BAND_IDS[band_name], band_id)


if __name__ == "__main__":
    unittest.main()
