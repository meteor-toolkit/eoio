"""eoio.readers.sentinel3_olci.metadata.tests.test_s3_olci_mtd - tests for S3OLCIXMLReader."""

from __future__ import annotations
import unittest
from unittest import mock
import tempfile
from pathlib import Path
import datetime as dt
from eoio.readers.sentinel3_olci.metadata.s3_olci_mtd import S3OLCIXMLReader


class TestS3OLCIXMLReader(unittest.TestCase):
    """Test cases for S3OLCIXMLReader metadata extraction."""

    def setUp(self):
        """Set up temporary directory and XML fixture for tests."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Representative Sentinel-3 OLCI xfdumanifest.xml structure
        # This includes temporal, identification, orbit, geometry, and OLCI-specific metadata
        self.xml_text = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu"
    xmlns:safe="http://www.esa.int/safe/sentinel/1.1"
    xmlns:sentinel3="http://www.esa.int/safe/sentinel/sentinel-3/1.0"
    xmlns:olci="http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0"
    xmlns:gml="http://www.opengis.net/gml"
    version="1.0">

  <!-- Temporal metadata -->
  <safe:acquisitionPeriod>
    <safe:startTime>2023-06-15T10:30:45.123456Z</safe:startTime>
    <safe:stopTime>2023-06-15T10:40:15.654321Z</safe:stopTime>
  </safe:acquisitionPeriod>

  <!-- Platform and identification information -->
  <safe:platform>
    <safe:familyName>Sentinel-3</safe:familyName>
    <safe:number>A</safe:number>
  </safe:platform>

  <!-- Orbit information -->
  <safe:orbitReference>
    <safe:orbitNumber type="start">10234</safe:orbitNumber>
    <safe:orbitNumber type="stop">10234</safe:orbitNumber>
    <safe:relativeOrbitNumber type="start">137</safe:relativeOrbitNumber>
    <safe:relativeOrbitNumber type="stop">137</safe:relativeOrbitNumber>
    <safe:cycleNumber>123</safe:cycleNumber>
    <safe:inclinationAngle>
      <safe:ascendingNodeDate>2023-06-15T10:35:30Z</safe:ascendingNodeDate>
    </safe:inclinationAngle>
  </safe:orbitReference>

  <!-- Geometry - footprint -->
  <safe:frameSet>
    <safe:footPrint>
      <gml:posList>
        60.5 10.0  60.5 20.0  50.5 20.0  50.5 10.0  60.5 10.0
      </gml:posList>
    </safe:footPrint>
  </safe:frameSet>

  <!-- Sentinel-3 general product information -->
  <sentinel3:generalProductInformation>
    <sentinel3:productName>S3A_OL_1_EFR____20230615T103045_20230615T104015_0350_078_OLCI_OBC_____LN3_O_ST_002.SEN3</sentinel3:productName>
    <sentinel3:productType>OL_1_EFR____</sentinel3:productType>
    <sentinel3:processingBaseline>02.30</sentinel3:processingBaseline>
  </sentinel3:generalProductInformation>

  <!-- OLCI-specific product information -->
  <olci:olciProductInformation>
    <olci:measurementAccuracy>
      <sentinel3:relativeMeasurementAccuracy>0.015</sentinel3:relativeMeasurementAccuracy>
    </olci:measurementAccuracy>
    <olci:imageSize>
      <sentinel3:rows>3754</sentinel3:rows>
      <sentinel3:columns>5120</sentinel3:columns>
    </olci:imageSize>
  </olci:olciProductInformation>

  <!-- OLCI Band descriptions (central wavelengths and bandwidths) -->
  <olci:bandDescriptions>
    <!-- Band 1: Aerosol 400nm -->
    <sentinel3:band name="Oa01">
      <sentinel3:centralWavelength>400.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>15.0</sentinel3:bandwidth>
    </sentinel3:band>
    <!-- Band 2: Aerosol 410nm -->
    <sentinel3:band name="Oa02">
      <sentinel3:centralWavelength>410.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <!-- Band 3: Aerosol 443nm -->
    <sentinel3:band name="Oa03">
      <sentinel3:centralWavelength>443.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <!-- Band 4: Water absorption 490nm -->
    <sentinel3:band name="Oa04">
      <sentinel3:centralWavelength>490.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <!-- Band 5: Chlorophyll 510nm -->
    <sentinel3:band name="Oa05">
      <sentinel3:centralWavelength>510.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>10.0</sentinel3:bandwidth>
    </sentinel3:band>
    <!-- Additional bands abbreviated for clarity -->
    <sentinel3:band name="Oa21">
      <sentinel3:centralWavelength>1020.0</sentinel3:centralWavelength>
      <sentinel3:bandwidth>40.0</sentinel3:bandwidth>
    </sentinel3:band>
  </olci:bandDescriptions>

</xfdu:XFDU>
"""
        self.xml_file = self.tmp_path / "xfdumanifest.xml"
        self.xml_file.write_text(self.xml_text, encoding="utf-8")

        self.reader = S3OLCIXMLReader(self.xml_file)

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()

    # ---- Temporal metadata --------------------------------------------------

    def test_find_acquisition_start_datetime(self):
        """Test parsing of acquisition start datetime."""
        dt_obj = self.reader.find_acquisition_start_datetime()
        self.assertIsInstance(dt_obj, dt.datetime)
        self.assertEqual(dt_obj.tzinfo, dt.timezone.utc)
        self.assertEqual(dt_obj.year, 2023)
        self.assertEqual(dt_obj.month, 6)
        self.assertEqual(dt_obj.day, 15)
        self.assertEqual(dt_obj.hour, 10)
        self.assertEqual(dt_obj.minute, 30)

    def test_find_acquisition_stop_datetime(self):
        """Test parsing of acquisition stop datetime."""
        dt_obj = self.reader.find_acquisition_stop_datetime()
        self.assertIsInstance(dt_obj, dt.datetime)
        self.assertEqual(dt_obj.tzinfo, dt.timezone.utc)
        self.assertEqual(dt_obj.year, 2023)
        self.assertEqual(dt_obj.month, 6)
        self.assertEqual(dt_obj.day, 15)
        self.assertEqual(dt_obj.hour, 10)
        self.assertEqual(dt_obj.minute, 40)

    def test_find_acquisition_start_date(self):
        """Test extraction of acquisition start date."""
        d = self.reader.find_acquisition_start_date()
        self.assertEqual(d, dt.date(2023, 6, 15))

    # ---- Identification / provenance ----------------------------------------

    def test_find_product_name(self):
        """Test extraction of product name."""
        name = self.reader.find_product_name()
        self.assertEqual(
            name,
            "S3A_OL_1_EFR____20230615T103045_20230615T104015_0350_078_OLCI_OBC_____LN3_O_ST_002.SEN3",
        )

    def test_find_product_type(self):
        """Test extraction of product type."""
        ptype = self.reader.find_product_type()
        self.assertEqual(ptype, "OL_1_EFR____")

    def test_find_processing_baseline(self):
        """Test extraction of processing baseline."""
        baseline = self.reader.find_processing_baseline()
        self.assertEqual(baseline, "02.30")

    def test_find_spacecraft_name(self):
        """Test extraction of spacecraft name."""
        spacecraft = self.reader.find_spacecraft_name()
        self.assertEqual(spacecraft, "Sentinel-3A")

    # ---- Orbit information --------------------------------------------------

    def test_find_orbit_number(self):
        """Test extraction of orbit number."""
        orbit = self.reader.find_orbit_number()
        self.assertIsInstance(orbit, int)
        self.assertEqual(orbit, 10234)

    def test_find_relative_orbit_number(self):
        """Test extraction of relative orbit number."""
        rel_orbit = self.reader.find_relative_orbit_number()
        self.assertIsInstance(rel_orbit, int)
        self.assertEqual(rel_orbit, 137)

    def test_find_cycle_number(self):
        """Test extraction of cycle number."""
        cycle = self.reader.find_cycle_number()
        self.assertIsInstance(cycle, int)
        self.assertEqual(cycle, 123)

    def test_find_orbit_direction(self):
        """Test extraction of orbit direction."""
        direction = self.reader.find_orbit_direction()
        self.assertIsInstance(direction, str)
        # Default from test fixture
        self.assertEqual(direction, "UNKNOWN")

    # ---- Geometry / footprint -----------------------------------------------

    def test_find_bounds_polygon(self):
        """Test extraction and construction of footprint polygon."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
        except ImportError:
            self.skipTest("Shapely not available; skipping footprint polygon test")

        # Patch lazy_shapely in the module where S3OLCIXMLReader is defined
        with mock.patch(
            "eoio.readers.sentinel3_olci.metadata.s3_olci_mtd.lazy_shapely",
            autospec=True,
        ) as m:
            m.return_value = (ShapelyPolygon, None, None, None, None)

            poly = self.reader.find_bounds()
            self.assertTrue(poly.is_valid)

            # The input posList is in lat/lon pairs: 60.5 10.0  60.5 20.0  50.5 20.0  50.5 10.0
            # After conversion to lon/lat: (10.0, 60.5), (20.0, 60.5), (20.0, 50.5), (10.0, 50.5)
            minx, miny, maxx, maxy = poly.bounds
            self.assertAlmostEqual(minx, 10.0)
            self.assertAlmostEqual(maxx, 20.0)
            self.assertAlmostEqual(miny, 50.5)
            self.assertAlmostEqual(maxy, 60.5)

    # ---- OLCI-specific metadata ---------------------------------------------

    def test_find_image_size(self):
        """Test extraction of image dimensions."""
        size = self.reader.find_image_size()
        self.assertIsInstance(size, dict)
        self.assertEqual(size["rows"], 3754)
        self.assertEqual(size["columns"], 5120)

    def test_find_measurement_accuracy(self):
        """Test extraction of measurement accuracy."""
        accuracy = self.reader.find_measurement_accuracy()
        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(accuracy, 0.015)

    # ---- Spectral metadata (band-specific) ----------------------------------

    def test_find_band_central_wavelength(self):
        """Test extraction of central wavelength for specific bands."""
        # Test Band 1 (400 nm)
        wl1 = self.reader.find_band_central_wavelength(1)
        self.assertAlmostEqual(wl1, 400.0)

        # Test Band 2 (410 nm)
        wl2 = self.reader.find_band_central_wavelength(2)
        self.assertAlmostEqual(wl2, 410.0)

        # Test Band 5 (510 nm)
        wl5 = self.reader.find_band_central_wavelength(5)
        self.assertAlmostEqual(wl5, 510.0)

        # Test Band 21 (1020 nm)
        wl21 = self.reader.find_band_central_wavelength(21)
        self.assertAlmostEqual(wl21, 1020.0)

    def test_find_band_bandwidth(self):
        """Test extraction of bandwidth for specific bands."""
        # Test Band 1 (15 nm)
        bw1 = self.reader.find_band_bandwidth(1)
        self.assertAlmostEqual(bw1, 15.0)

        # Test Band 2 (10 nm)
        bw2 = self.reader.find_band_bandwidth(2)
        self.assertAlmostEqual(bw2, 10.0)

        # Test Band 21 (40 nm)
        bw21 = self.reader.find_band_bandwidth(21)
        self.assertAlmostEqual(bw21, 40.0)

    def test_find_all_band_central_wavelengths(self):
        """Test aggregation of all band central wavelengths."""
        wavelengths = self.reader.find_all_band_central_wavelengths()
        self.assertIsInstance(wavelengths, dict)
        # We defined bands 1-5 and 21 in fixture
        self.assertIn(1, wavelengths)
        self.assertIn(5, wavelengths)
        self.assertIn(21, wavelengths)
        self.assertEqual(wavelengths[1], 400.0)
        self.assertEqual(wavelengths[5], 510.0)
        self.assertEqual(wavelengths[21], 1020.0)

    def test_find_all_band_bandwidths(self):
        """Test aggregation of all band bandwidths."""
        bandwidths = self.reader.find_all_band_bandwidths()
        self.assertIsInstance(bandwidths, dict)
        # We defined bands 1-5 and 21 in fixture
        self.assertIn(1, bandwidths)
        self.assertIn(5, bandwidths)
        self.assertIn(21, bandwidths)
        self.assertEqual(bandwidths[1], 15.0)
        self.assertEqual(bandwidths[21], 40.0)


if __name__ == "__main__":
    unittest.main()
