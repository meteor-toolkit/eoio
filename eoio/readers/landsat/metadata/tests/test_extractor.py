"""eoio.readers.landsat.tests.test_extractor - tests for eoio.readers.landsat.metadata.extractor"""

from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch
import datetime as dt
import xarray as xr

from eoio.readers.landsat.metadata.extractor import LSMetadataExtractor


class TestLSMetadataExtractor(unittest.TestCase):
    @patch("eoio.readers.landsat.metadata.extractor.LSL1ProdJSONReader")
    @patch("eoio.readers.landsat.metadata.extractor.LSL1ProdXMLReader")
    def setUp(self, mock_xml_reader_cls, mock_json_reader_cls):
        # ---- fake reader + layout ----
        self.reader = MagicMock()
        self.reader.config.subset = {"roi": "dummy"}
        self.reader.layout = MagicMock()
        self.reader.layout.product_metadata_xml.return_value = "/tmp/LC08_L1TP_MTL.xml"
        self.reader.layout.product_metadata_json.return_value = "/tmp/LC08_L1TP_STAC.json"
        self.ds = xr.Dataset()

        # ---- mock XML reader instance ----
        self.mock_xml = MagicMock()
        mock_xml_reader_cls.return_value = self.mock_xml

        # ---- mock JSON reader instance ----
        self.mock_json = MagicMock()
        mock_json_reader_cls.return_value = self.mock_json

        # ---- XML reader methods used by extractor ----
        self.mock_xml.find_product_id.return_value = "LC08_L1TP_190030_20250519_20250519_02_T1"
        self.mock_xml.find_spacecraft_name.return_value = "LANDSAT_8"
        self.mock_xml.find_sensor_id.return_value = "OLI_TIRS"
        self.mock_xml.find_processing_level.return_value = "L1TP"
        self.mock_xml.find_collection_number.return_value = "02"
        self.mock_xml.find_acquisition_datetime.return_value = dt.datetime(
            2025, 5, 19, 10, 42, 12, tzinfo=dt.timezone.utc
        )

        # Product-level metadata
        self.mock_xml.find_collection_category.return_value = "TIER_1"
        self.mock_xml.find_value.side_effect = lambda key, default=None: {
            "scene_center_time": "10:42:12.3456780Z",
            "sun_elevation": 45.3,
            "sun_azimuth": 123.4,
            "cloud_cover": 12.0,
            "cloud_cover_land": 10.0,
            "earth_sun_distance": 0.99,
            "map_projection": "UTM",
            "utm_zone": 30,
            "date_product_generated": "2025-05-19T13:25:00Z",
            "processing_software_version": "LPGS_15.4.0",
        }.get(key, default)
        self.mock_xml.find_wrs_path.return_value = 190
        self.mock_xml.find_wrs_row.return_value = 30
        self.mock_xml.find_projection_parameters.return_value = {
            "datum": "WGS84",
            "grid_cell_size_reflective": 30,
            "grid_cell_size_pan": 15,
            "grid_cell_size_thermal": 100,
        }

        # Variable-level metadata
        self.mock_xml.get_radiometric_rescaling.return_value = {
            "B2": {
                "radiance_mult": 0.0123,
                "radiance_add": 0.123,
                "reflectance_mult": 0.0001,
                "reflectance_add": -0.1,
            },
            "B10": {
                "radiance_mult": 0.1001,
                "radiance_add": 0.001,
                "reflectance_mult": None,
                "reflectance_add": None,
            },
        }

        # ---- JSON reader methods ----
        # band metadata
        self.mock_json.find_all_band_central_wavelengths.return_value = {
            "B1": 443.0,
            "B2": 482.0,
            "B3": 561.0,
            "B4": 655.0,
            "B5": 865.0,
            "B6": 1609.0,
            "B7": 2201.0,
            "B8": 590.0,
            "B9": 1373.0,
            "B10": 10895.0,
            "B11": 12005.0,
        }
        self.mock_json.find_all_band_gsds.return_value = {
            "B1": 30.0,
            "B2": 30.0,
            "B3": 30.0,
            "B4": 30.0,
            "B5": 30.0,
            "B6": 30.0,
            "B7": 30.0,
            "B8": 15.0,
            "B9": 30.0,
            "B10": 100.0,
            "B11": 100.0,
        }
        self.mock_json.find_epsg.return_value = 32630

        self.extractor = LSMetadataExtractor(self.reader)

    def test_get_basic_metadata_ReturnsCorrectStructure(self):
        basic = self.extractor.get_basic_metadata()
        self.assertEqual(basic["product_name"], "LC08_L1TP_190030_20250519_20250519_02_T1")
        self.assertEqual(basic["platform"], "LANDSAT_8")
        self.assertEqual(basic["epsg"], 32630)
        self.assertIn(30, basic["spatial_resolution"])
        self.assertIn("30m", basic["geometry_ids"])
        self.assertIn("footprint", basic)
        self.assertIsInstance(basic["footprint"], dict)
        self.assertEqual(basic["footprint"]["crs"], "EPSG:32630")
        self.assertEqual(basic["product_datetime"], "2025-05-19T10:42:12+00:00")

    def test_get_product_metadata_ReturnsCorrectStructure(self):
        pm = self.extractor.get_product_metadata()
        self.assertEqual(pm["collection_number"], "02")
        self.assertEqual(pm["sun_elevation"], 45.3)
        self.assertEqual(pm["wrs_path"], 190)
        self.assertEqual(pm["cloud_area_fraction"], 12.0)

    def test_get_variable_product_metadata_ReturnsCorrectStructure(self):
        # B2: Reflective
        vm_b2 = self.extractor.get_variable_product_metadata("B2")
        self.assertEqual(vm_b2["radiance_mult"], 0.0123)
        self.assertEqual(vm_b2["reflectance_mult"], 0.0001)
        self.assertEqual(vm_b2["band_central_wavelength"], 482.0)
        self.assertEqual(vm_b2["spatial_resolution"], 30)

        # B10: Thermal
        vm_b10 = self.extractor.get_variable_product_metadata("B10")
        self.assertEqual(vm_b10["radiance_mult"], 0.1001)

    def test_get_variable_product_metadata_aux_vars_have_spatial_metadata(self):
        """Test that auxiliary variables include spatial_resolution and geometry_id metadata."""
        aux_vars = ["ATRAN", "CDIST", "DRAD", "EMIS", "EMSD", "TRAD", "URAD"]
        for var in aux_vars:
            vm_aux = self.extractor.get_variable_product_metadata(var)
            self.assertIn("spatial_resolution", vm_aux, f"{var} missing spatial_resolution")
            self.assertEqual(vm_aux["spatial_resolution"], 30, f"{var} should have 30m resolution")
            self.assertEqual(vm_aux["spatial_resolution_units"], "m", f"{var} should have units=m")
            self.assertEqual(vm_aux["geometry_id"], "30m", f"{var} should have geometry_id=30m")
            # Auxiliary vars should still have their descriptive metadata
            self.assertIn("standard_name", vm_aux, f"{var} missing standard_name")
            self.assertIn("description", vm_aux, f"{var} missing description")

    def test_attach_metadata_AddsAttrsToDataset(self):
        self.extractor.get_basic_metadata = MagicMock(return_value={"basic": "info"})
        self.extractor.get_product_metadata = MagicMock(return_value={"prod": "info"})

        ds = self.extractor.attach_metadata(self.ds, level="all")

        self.assertIn("basic", ds.attrs)
        self.assertEqual(ds.attrs["basic"], "info")
        self.assertIn("product_metadata", ds.attrs)
        self.assertEqual(ds.attrs["product_metadata"]["prod"], "info")


if __name__ == "__main__":
    unittest.main()
