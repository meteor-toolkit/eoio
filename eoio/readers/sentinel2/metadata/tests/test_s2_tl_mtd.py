"""eoio.readers.sentinel2.metadata.tests.test_s2_tl_mtd- tests for eoio.readers.sentinel2.metadata.test_s2_tl_mtd"""

from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import datetime as dt
import numpy as np
from eoio.readers.sentinel2.metadata.s2_tl_mtd import S2TLXMLReader


class TestS2TLXMLReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.xml_file = self.tmp_path / "MTD_TL.xml"

        # Minimal but representative MTD_TL.xml structure:
        # - root uses n1 prefix and namespace
        # - n1:General_Info and n1:Geometric_Info and n1:Quality_Indicators_Info are prefixed
        # - most children are unprefixed
        self.xml_file.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-1C_Tile
  xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd">

  <n1:General_Info>
    <TILE_ID>S2A_OPER_MSI_L1C_TL_TEST_A000000_T30UXE_N05.11</TILE_ID>
    <DATASTRIP_ID>S2A_OPER_MSI_L1C_DS_TEST_S20251128T111434_N05.11</DATASTRIP_ID>
    <DOWNLINK_PRIORITY>NOMINAL</DOWNLINK_PRIORITY>
    <SENSING_TIME>2025-11-28T11:14:31.024Z</SENSING_TIME>
    <Archiving_Info>
      <ARCHIVING_TIME>2025-11-28T13:18:48.109865Z</ARCHIVING_TIME>
      <ARCHIVING_CENTRE>ESRIN</ARCHIVING_CENTRE>
    </Archiving_Info>
  </n1:General_Info>

  <n1:Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_NAME>WGS84 / UTM zone 30N</HORIZONTAL_CS_NAME>
      <HORIZONTAL_CS_CODE>EPSG:32630</HORIZONTAL_CS_CODE>

      <Size resolution="10">
        <NROWS>4</NROWS><NCOLS>5</NCOLS>
      </Size>
      <Size resolution="20">
        <NROWS>2</NROWS><NCOLS>3</NCOLS>
      </Size>
      <Size resolution="60">
        <NROWS>1</NROWS><NCOLS>2</NCOLS>
      </Size>

      <Geoposition resolution="10">
        <ULX>600000</ULX><ULY>5700000</ULY><XDIM>10</XDIM><YDIM>-10</YDIM>
      </Geoposition>
      <Geoposition resolution="20">
        <ULX>600000</ULX><ULY>5700000</ULY><XDIM>20</XDIM><YDIM>-20</YDIM>
      </Geoposition>
      <Geoposition resolution="60">
        <ULX>600000</ULX><ULY>5700000</ULY><XDIM>60</XDIM><YDIM>-60</YDIM>
      </Geoposition>
    </Tile_Geocoding>

    <Tile_Angles>
      <Mean_Sun_Angle>
        <ZENITH_ANGLE>70.0</ZENITH_ANGLE>
        <AZIMUTH_ANGLE>150.0</AZIMUTH_ANGLE>
      </Mean_Sun_Angle>

      <Sun_Angles_Grid>
        <Zenith>
          <COL_STEP unit="m">5000</COL_STEP>
          <ROW_STEP unit="m">5000</ROW_STEP>
          <Values_List>
            <VALUES>1 2 3</VALUES>
            <VALUES>4 5 6</VALUES>
          </Values_List>
        </Zenith>
        <Azimuth>
          <COL_STEP unit="m">5000</COL_STEP>
          <ROW_STEP unit="m">5000</ROW_STEP>
          <Values_List>
            <VALUES>10 20 30</VALUES>
            <VALUES>40 50 60</VALUES>
          </Values_List>
        </Azimuth>
      </Sun_Angles_Grid>

      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="7">
        <Zenith>
          <Values_List>
            <VALUES>1 1</VALUES>
            <VALUES>1 1</VALUES>
          </Values_List>
        </Zenith>
        <Azimuth>
          <Values_List>
            <VALUES>10 10</VALUES>
            <VALUES>10 10</VALUES>
          </Values_List>
        </Azimuth>
      </Viewing_Incidence_Angles_Grids>

      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="8">
        <Zenith>
          <Values_List>
            <VALUES>3 3</VALUES>
            <VALUES>3 3</VALUES>
          </Values_List>
        </Zenith>
        <Azimuth>
          <Values_List>
            <VALUES>30 30</VALUES>
            <VALUES>30 30</VALUES>
          </Values_List>
        </Azimuth>
      </Viewing_Incidence_Angles_Grids>

    </Tile_Angles>
  </n1:Geometric_Info>

  <n1:Quality_Indicators_Info>
    <Image_Content_QI>
      <CLOUDY_PIXEL_PERCENTAGE>0.5</CLOUDY_PIXEL_PERCENTAGE>
      <SNOW_PIXEL_PERCENTAGE>1.25</SNOW_PIXEL_PERCENTAGE>
      <DEGRADED_MSI_DATA_PERCENTAGE>0</DEGRADED_MSI_DATA_PERCENTAGE>
    </Image_Content_QI>

    <Pixel_Level_QI>
      <MASK_FILENAME type="MSK_CLASSI">QI_DATA/MSK_CLASSI_B00.gml</MASK_FILENAME>
      <MASK_FILENAME type="MSK_QUALIT" bandId="1">QI_DATA/MSK_QUALIT_B01.gml</MASK_FILENAME>
    </Pixel_Level_QI>

    <PVI_FILENAME>GRANULE/xxx/preview.jpg</PVI_FILENAME>
  </n1:Quality_Indicators_Info>

</n1:Level-1C_Tile>
""",
            encoding="utf-8",
        )

        self.reader = S2TLXMLReader(self.xml_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Scalars / thin wrappers
    # ------------------------------------------------------------------

    def test_basic_scalars(self):
        self.assertTrue(self.reader.find_tile_id().startswith("S2A_OPER_MSI_L1C_TL_TEST"))
        self.assertTrue(self.reader.find_datastrip_id().startswith("S2A_OPER_MSI_L1C_DS_TEST"))
        self.assertEqual(self.reader.find_downlink_priority(), "NOMINAL")
        self.assertEqual(self.reader.find_horizontal_cs_name(), "WGS84 / UTM zone 30N")
        self.assertEqual(self.reader.find_horizontal_cs_code(), "EPSG:32630")

    def test_sensing_time_and_datetime(self):
        s = self.reader.find_sensing_time_str()
        self.assertEqual(s, "2025-11-28T11:14:31.024Z")

        d = self.reader.find_sensing_datetime()
        self.assertEqual(d.tzinfo, dt.timezone.utc)
        self.assertEqual(d.date(), dt.date(2025, 11, 28))
        self.assertEqual(d.hour, 11)
        self.assertEqual(d.minute, 14)

    # ------------------------------------------------------------------
    # Geocoding
    # ------------------------------------------------------------------

    def test_tile_shape_and_geoposition(self):
        self.assertEqual(self.reader.find_tile_shape(10), (4, 5))
        self.assertEqual(self.reader.find_tile_shape(20), (2, 3))
        self.assertEqual(self.reader.find_tile_shape(60), (1, 2))

        geo10 = self.reader.find_geoposition(10)
        self.assertEqual(geo10["ulx"], 600000.0)
        self.assertEqual(geo10["uly"], 5700000.0)
        self.assertEqual(geo10["xdim"], 10.0)
        self.assertEqual(geo10["ydim"], -10.0)

    def test_build_xy_coords(self):
        x, y = self.reader.build_xy_coords(10)
        self.assertEqual(x.shape, (5,))
        self.assertEqual(y.shape, (4,))

        # Pixel centres:
        # x0 = 600000 + 0.5*10 = 600005
        self.assertAlmostEqual(x[0], 600005.0)
        # y0 = 5700000 + 0.5*(-10) = 5699995
        self.assertAlmostEqual(y[0], 5699995.0)

    # ------------------------------------------------------------------
    # Quality indicators
    # ------------------------------------------------------------------

    def test_quality_indicators(self):
        self.assertAlmostEqual(self.reader.find_cloudy_pixel_percentage(), 0.5)
        self.assertAlmostEqual(self.reader.find_snow_pixel_percentage(), 1.25)
        self.assertAlmostEqual(self.reader.find_degraded_msi_data_percentage(), 0.0)
        self.assertEqual(self.reader.find_pvi_filename(), "GRANULE/xxx/preview.jpg")

    def test_mask_filenames(self):
        masks = self.reader.find_mask_filenames()
        self.assertEqual(masks[("MSK_CLASSI", None)], "QI_DATA/MSK_CLASSI_B00.gml")
        self.assertEqual(masks[("MSK_QUALIT", 1)], "QI_DATA/MSK_QUALIT_B01.gml")

    # ------------------------------------------------------------------
    # Sun angles
    # ------------------------------------------------------------------

    def test_mean_sun_angles(self):
        z, a = self.reader.find_mean_sun_angles()
        self.assertAlmostEqual(z, 70.0)
        self.assertAlmostEqual(a, 150.0)

    def test_sun_grids(self):
        col_step, row_step = self.reader.find_sun_angle_steps()
        self.assertAlmostEqual(col_step, 5000.0)
        self.assertAlmostEqual(row_step, 5000.0)

        zen = self.reader.find_sun_angle_grid("zenith")
        azi = self.reader.find_sun_angle_grid("azimuth")

        self.assertEqual(zen.shape, (2, 3))
        self.assertEqual(azi.shape, (2, 3))

        np.testing.assert_allclose(zen, np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
        np.testing.assert_allclose(azi, np.array([[10, 20, 30], [40, 50, 60]], dtype=float))

    # ------------------------------------------------------------------
    # Viewing incidence angles
    # ------------------------------------------------------------------

    def test_list_viewing_grids(self):
        pairs = self.reader.list_viewing_grids()
        self.assertEqual(pairs, [(1, 7), (1, 8)])

    def test_viewing_grids(self):
        zen_stack, dets = self.reader.find_viewing_angle_grid(band_id=1, direction="zenith")
        azi_stack, dets2 = self.reader.find_viewing_angle_grid(band_id=1, direction="azimuth")

        self.assertEqual(dets, [7, 8])
        self.assertEqual(dets2, [7, 8])

        # Shape is (ndetectors, nrows, ncols)
        self.assertEqual(zen_stack.shape, (2, 2, 2))
        self.assertEqual(azi_stack.shape, (2, 2, 2))

        # Detector 7
        np.testing.assert_allclose(zen_stack[0], np.ones((2, 2), dtype=float))
        np.testing.assert_allclose(azi_stack[0], np.full((2, 2), 10.0))

        # Detector 8
        np.testing.assert_allclose(zen_stack[1], np.full((2, 2), 3.0))
        np.testing.assert_allclose(azi_stack[1], np.full((2, 2), 30.0))

    def test_band_mean_viewing_grids(self):
        zen, azi = self.reader.find_band_mean_viewing_grids(band_id=1)

        # Mean of detector 7 and 8:
        # zen mean: (1 + 3) / 2 = 2
        # azi mean: (10 + 30) / 2 = 20
        np.testing.assert_allclose(zen, np.full((2, 2), 2.0))
        np.testing.assert_allclose(azi, np.full((2, 2), 20.0))

    # ------------------------------------------------------------------
    # Internal grid parsing behaviour (optional)
    # ------------------------------------------------------------------

    def test_read_values_rows_non_rectangular_raises(self):
        # Create a deliberately non-rectangular grid by calling the helper on a custom xpath.
        # We do this by writing an extra VALUES row with different length into the existing file.
        bad_xml = self.xml_file.read_text(encoding="utf-8").replace(
            "<VALUES>4 5 6</VALUES>",
            "<VALUES>4 5</VALUES>",
        )
        self.xml_file.write_text(bad_xml, encoding="utf-8")

        reader2 = S2TLXMLReader(self.xml_file)
        with self.assertRaises(ValueError):
            _ = reader2.find_sun_angle_grid("zenith")


if __name__ == "__main__":
    unittest.main()
