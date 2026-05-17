"""Tests for eoio.readers.sentinel3_slstr.metadata.s3_slstr_mtd.S3SLSTRXMLReader."""

from __future__ import annotations
import unittest
from pathlib import Path
import tempfile
import datetime as dt

from eoio.readers.sentinel3_slstr.metadata.s3_slstr_mtd import S3SLSTRXMLReader


class TestS3SLSTRXMLReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Minimal xfdumanifest.xml covering the fields used by the reader
        self.xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu"
		xmlns:safe="http://www.esa.int/safe/sentinel/1.1"
		xmlns:sentinel3="http://www.esa.int/safe/sentinel/sentinel-3/1.0"
		xmlns:slstr="http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0"
		xmlns:gml="http://www.opengis.net/gml"
		version="1.0">

	<safe:acquisitionPeriod>
		<safe:startTime>2024-01-02T03:04:05.000000Z</safe:startTime>
		<safe:stopTime>2024-01-02T03:14:05.000000Z</safe:stopTime>
	</safe:acquisitionPeriod>

	<safe:platform>
		<safe:familyName>Sentinel-3</safe:familyName>
		<safe:number>A</safe:number>
	</safe:platform>

	<safe:orbitReference>
		<safe:orbitNumber type="start">12345</safe:orbitNumber>
		<safe:relativeOrbitNumber type="start">67</safe:relativeOrbitNumber>
		<safe:cycleNumber>89</safe:cycleNumber>
	</safe:orbitReference>

	<safe:frameSet>
		<safe:footPrint srsName="http://www.opengis.net/def/crs/EPSG/0/4326">
			<gml:posList>60.0 10.0 60.0 20.0 50.0 20.0 50.0 10.0 60.0 10.0</gml:posList>
		</safe:footPrint>
	</safe:frameSet>

	<sentinel3:generalProductInformation>
		<sentinel3:productName>S3A_SL_1_RBT____20240102T030405_20240102T031405_0012_034_XXX___SLSTR</sentinel3:productName>
		<sentinel3:productType>SL_1_RBT____</sentinel3:productType>
		<sentinel3:processingBaseline>03.10</sentinel3:processingBaseline>
	</sentinel3:generalProductInformation>

	<slstr:slstrProductInformation>
		<slstr:measurementAccuracy>
			<sentinel3:relativeMeasurementAccuracy>0.020</sentinel3:relativeMeasurementAccuracy>
		</slstr:measurementAccuracy>
		<slstr:nadirImageSize grid="1 km">
			<sentinel3:rows>1500</sentinel3:rows>
			<sentinel3:columns>2000</sentinel3:columns>
		</slstr:nadirImageSize>
		<slstr:bandDescriptions>
			<sentinel3:band name="S1">
				<sentinel3:centralWavelength>400.5</sentinel3:centralWavelength>
				<sentinel3:bandwidth>10.0</sentinel3:bandwidth>
			</sentinel3:band>
			<sentinel3:band name="F1">
				<sentinel3:centralWavelength>1234.5</sentinel3:centralWavelength>
				<sentinel3:bandwidth>20.0</sentinel3:bandwidth>
			</sentinel3:band>
		</slstr:bandDescriptions>
	</slstr:slstrProductInformation>

</xfdu:XFDU>
"""

        self.xml_file = self.tmp_path / "xfdumanifest.xml"
        self.xml_file.write_text(self.xml_text, encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_basic_parsing_methods(self):
        r = S3SLSTRXMLReader(str(self.xml_file))
        self.assertEqual(
            r.find_product_name(),
            "S3A_SL_1_RBT____20240102T030405_20240102T031405_0012_034_XXX___SLSTR",
        )
        self.assertEqual(r.find_product_type(), "SL_1_RBT____")
        self.assertEqual(r.find_processing_baseline(), "03.10")
        self.assertEqual(r.find_spacecraft_name(), "Sentinel-3A")
        self.assertEqual(r.find_orbit_number(), 12345)
        self.assertEqual(r.find_relative_orbit_number(), 67)
        self.assertEqual(r.find_cycle_number(), 89)

        start_dt = r.find_acquisition_start_datetime()
        stop_dt = r.find_acquisition_stop_datetime()
        self.assertIsInstance(start_dt, dt.datetime)
        self.assertIsInstance(stop_dt, dt.datetime)

    def test_find_bounds_and_crs_and_image_size(self):
        r = S3SLSTRXMLReader(str(self.xml_file))
        poly = r.find_bounds()
        # shapely polygon WKT expected
        self.assertTrue(poly.wkt.startswith("POLYGON"))

        crs = r.find_crs_code()
        self.assertEqual(crs, "EPSG:4326")

        img = r.find_image_size()
        self.assertEqual(img.get("rows"), 1500)
        self.assertEqual(img.get("columns"), 2000)

    def test_spectral_helpers(self):
        r = S3SLSTRXMLReader(str(self.xml_file))
        self.assertAlmostEqual(r.find_band_central_wavelength("S1"), 400.5)
        self.assertAlmostEqual(r.find_band_bandwidth("S1"), 10.0)

        all_cw = r.find_all_band_central_wavelengths()
        all_bw = r.find_all_band_bandwidths()
        self.assertIn("S1", all_cw)
        self.assertIn("F1", all_cw)
        self.assertAlmostEqual(all_cw["S1"], 400.5)
        self.assertAlmostEqual(all_bw["F1"], 20.0)


if __name__ == "__main__":
    unittest.main()
