"""eoio.readers.sentinel2.metadata.tests.test_s2_prod_mtd- tests for eoio.readers.sentinel2.metadata.test_s2_prod_mtd"""

from __future__ import annotations
import unittest
from unittest import mock
import tempfile
from pathlib import Path
import datetime as dt
from eoio.readers.sentinel2.metadata.s2_prod_mtd import S2ProdXMLReader


class TestS2ProdXMLReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Small but representative S2 product metadata fragment
        self.xml_text = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<n1:Level-1C_User_Product
    xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd">

  <n1:General_Info>
    <Product_Info>
      <PRODUCT_START_TIME>2025-11-28T11:14:31.024Z</PRODUCT_START_TIME>
      <PRODUCT_URI>S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE</PRODUCT_URI>
      <PROCESSING_LEVEL>Level-1C</PROCESSING_LEVEL>
      <PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>

      <Datatake datatakeIdentifier="GS2A_20251128T111431_054503_N05.11">
        <SPACECRAFT_NAME>Sentinel-2A</SPACECRAFT_NAME>
        <SENSING_ORBIT_NUMBER>137</SENSING_ORBIT_NUMBER>
        <SENSING_ORBIT_DIRECTION>DESCENDING</SENSING_ORBIT_DIRECTION>
      </Datatake>
    </Product_Info>

    <Product_Image_Characteristics>
      <QUANTIFICATION_VALUE unit="none">10000</QUANTIFICATION_VALUE>

      <Radiometric_Offset_List>
        <RADIO_ADD_OFFSET band_id="0">-1000</RADIO_ADD_OFFSET>
        <RADIO_ADD_OFFSET band_id="1">-1000</RADIO_ADD_OFFSET>
      </Radiometric_Offset_List>

      <Reflectance_Conversion>
        <Solar_Irradiance_List>
          <SOLAR_IRRADIANCE bandId="0" unit="W/m²/µm">1884.69</SOLAR_IRRADIANCE>
          <SOLAR_IRRADIANCE bandId="1" unit="W/m²/µm">1959.66</SOLAR_IRRADIANCE>
        </Solar_Irradiance_List>
      </Reflectance_Conversion>

      <Spectral_Information_List>
        <Spectral_Information bandId="0" physicalBand="B1">
          <RESOLUTION>60</RESOLUTION>
          <Wavelength>
            <CENTRAL unit="nm">442.7</CENTRAL>
          </Wavelength>
          <Spectral_Response>
            <VALUES>0.1 0.2 0.3</VALUES>
          </Spectral_Response>
        </Spectral_Information>

        <Spectral_Information bandId="1" physicalBand="B2">
          <RESOLUTION>10</RESOLUTION>
          <Wavelength>
            <CENTRAL unit="nm">492.7</CENTRAL>
          </Wavelength>
          <Spectral_Response>
            <VALUES>1 2 3</VALUES>
          </Spectral_Response>
        </Spectral_Information>
      </Spectral_Information_List>

      <PHYSICAL_GAINS bandId="0">4.0867035</PHYSICAL_GAINS>
      <PHYSICAL_GAINS bandId="1">3.74665902</PHYSICAL_GAINS>
    </Product_Image_Characteristics>
  </n1:General_Info>

  <n1:Geometric_Info>
    <Product_Footprint>
      <Product_Footprint>
        <Global_Footprint>
          <EXT_POS_LIST>
            54.0 -1.0  54.0 0.0  53.0 0.0  53.0 -1.0  54.0 -1.0
          </EXT_POS_LIST>
        </Global_Footprint>
      </Product_Footprint>
    </Product_Footprint>
  </n1:Geometric_Info>

</n1:Level-1C_User_Product>
"""
        self.xml_file = self.tmp_path / "MTD_MSIL1C.xml"
        self.xml_file.write_text(self.xml_text, encoding="utf-8")

        self.reader = S2ProdXMLReader(self.xml_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    # ---- Time ----------------------------------------------------------------

    def test_find_product_start_datetime(self):
        dt_obj = self.reader.find_product_start_datetime()
        self.assertEqual(dt_obj.tzinfo, dt.timezone.utc)
        self.assertEqual(dt_obj.year, 2025)
        self.assertEqual(dt_obj.month, 11)
        self.assertEqual(dt_obj.day, 28)
        self.assertEqual(dt_obj.hour, 11)
        self.assertEqual(dt_obj.minute, 14)

    def test_find_product_start_date(self):
        d = self.reader.find_product_start_date()
        self.assertEqual(d, dt.date(2025, 11, 28))

    # ---- Simple scalar fields ------------------------------------------------

    def test_find_product_uri_processing_fields(self):
        self.assertIn("S2A_MSIL1C_", self.reader.find_value("product_uri"))
        self.assertEqual(self.reader.find_value("processing_level"), "Level-1C")
        self.assertEqual(self.reader.find_value("processing_baseline"), "05.11")

    def test_find_orbit_fields(self):
        self.assertEqual(self.reader.find_value("spacecraft_name"), "Sentinel-2A")
        self.assertEqual(self.reader.find_value("sensing_orbit_number"), 137)
        self.assertEqual(self.reader.find_value("sensing_orbit_direction"), "DESCENDING")

    # ---- Radiometry ----------------------------------------------------------

    def test_find_quantification_values(self):
        q = self.reader.find_quantification_values()
        self.assertIsInstance(q["reflectance"], int)
        self.assertEqual(q["reflectance"], 10000)

    def test_find_reflectance_offsets(self):
        offsets = self.reader.find_reflectance_offsets()
        self.assertEqual(offsets, {"B01": -1000, "B02": -1000})

    def test_find_physical_gains(self):
        gains = self.reader.find_physical_gains()
        self.assertEqual(gains, {0: 4.0867035, 1: 3.74665902})

    def test_find_solar_irradiance(self):
        irr = self.reader.find_solar_irradiance()
        self.assertEqual(irr, {0: 1884.69, 1: 1959.66})

    def test_find_solar_irradiance_unit(self):
        irr = self.reader.find_solar_irradiance_unit()
        self.assertEqual(irr, "W/m²/µm")

    # ---- Spectral ------------------------------------------------------------

    def test_find_band_resolution(self):
        self.assertEqual(self.reader.find_band_resolution(0), 60)
        self.assertEqual(self.reader.find_band_resolution(1), 10)

    def test_find_band_central_wavelength(self):
        self.assertAlmostEqual(self.reader.find_band_central_wavelength(0), 442.7)
        self.assertAlmostEqual(self.reader.find_band_central_wavelength(1), 492.7)

    def test_find_spectral_response(self):
        # band 0 are floats
        v0 = self.reader.find_spectral_response(0)
        self.assertEqual(v0, [0.1, 0.2, 0.3])

        # band 1 are ints in the XML; XMLReader will cast them to int
        v1 = self.reader.find_spectral_response(1)
        self.assertEqual(v1, [1, 2, 3])

    # ---- Footprint polygon ---------------------------------------------------
    #
    # We patch lazy_shapely to avoid forcing Shapely as a hard dependency in tests.
    # If your project already depends on Shapely, you can remove the patch and
    # import Polygon directly.

    def test_find_bounds_polygon(self):
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
        except Exception:
            self.skipTest("Shapely not available; skipping footprint polygon test")

        # Patch the lazy_shapely in the module where S2ProdXMLReader.find_bounds imports it.
        # Update this target to your real module path.
        with mock.patch("eoio.readers.sentinel2.metadata.s2_prod_mtd.lazy_shapely", autospec=True) as m:
            m.return_value = (ShapelyPolygon, None, None, None, None)

            poly = self.reader.find_bounds()
            self.assertTrue(poly.is_valid)

            # The input is lat lon pairs; polygon should be built lon/lat
            # Expected lon/lat coords: (-1,54), (0,53), (-1,54)
            # (closure may be enforced by your implementation)
            minx, miny, maxx, maxy = poly.bounds
            self.assertAlmostEqual(minx, -1.0)
            self.assertAlmostEqual(maxx, 0.0)
            self.assertAlmostEqual(miny, 53.0)
            self.assertAlmostEqual(maxy, 54.0)


if __name__ == "__main__":
    unittest.main()
