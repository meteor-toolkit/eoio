"""eoio.readers.sentinel2.tests.test_metadata - tests for eoio.readers.sentinel2.metadata"""

from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import datetime as dt
from types import SimpleNamespace
from eoio.readers.sentinel2.reader import S2MSIReader
from eoio.readers.sentinel2.metadata.extractor import S2MSIMetadataExtractor

# Path to a real Sentinel-2 product XML for integration testing.
# If you run the tests, ensure this path points to a valid SAFE product on your system.
# Otherwise, the integration test will be skipped.
# SAFE_PATH = "C:\Users\seh2\OneDrive - National Physical Laboratory\Data\Archive\S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"
SAFE_PATH = "/Users/seh2/Library/CloudStorage/OneDrive-NationalPhysicalLaboratory/Data/archive/sentinel2/S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"


class TestS2MSIMetadataExtractor(unittest.TestCase):
    def _make_reader_with_layout(self):
        """
        Create a minimal fake eoio Sentinel-2 reader with a layout that returns
        paths for the three metadata XMLs.
        """
        layout = SimpleNamespace(
            product_metadata_xml=MagicMock(return_value="/tmp/MTD_MSIL1C.xml"),
            tl_metadata_xml=MagicMock(return_value="/tmp/MTD_TL.xml"),
            ds_metadata_xml=MagicMock(return_value="/tmp/MTD_DS.xml"),
        )
        reader = SimpleNamespace(layout=layout)
        return reader

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_init_sets_radiometric_offsets_to_zero_when_missing(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """
        If radiometric offsets are missing (PSD < v15), extractor should fill
        offsets with 0 for each band present in solar_irradiance.
        """
        reader = self._make_reader_with_layout()

        prod = mock_prod_cls.return_value
        _tl = mock_tl_cls.return_value
        ds = mock_ds_cls.return_value

        # Cached dicts
        prod.find_solar_irradiance.return_value = {0: 100.0, 1: 200.0}
        prod.find_physical_gains.return_value = {0: 1.0, 1: 1.1}

        ds.find_absolute_calibration_accuracy.return_value = {0: 0.01, 1: 0.02}
        ds.find_cross_band_calibration_accuracy.return_value = {0: 0.03, 1: 0.04}
        ds.find_multi_temporal_calibration_accuracy.return_value = {0: 0.05, 1: 0.06}
        ds.find_noise_model_alpha.return_value = {0: 7.0, 1: 8.0}
        ds.find_noise_model_beta.return_value = {0: 9.0, 1: 10.0}

        # Radiometric offsets missing in old PSD
        prod.find_reflectance_offsets.return_value = None

        ex = S2MSIMetadataExtractor(reader)

        # Expect zero-filled offsets for the solar irradiance keys
        self.assertEqual(ex._radiometric_offsets, {0: 0, 1: 0})

        # Sanity: layout methods used to create XML readers
        reader.layout.product_metadata_xml.assert_called_once()
        reader.layout.tl_metadata_xml.assert_called_once()
        reader.layout.ds_metadata_xml.assert_called_once()

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_basic_metadata(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        reader = self._make_reader_with_layout()

        prod = mock_prod_cls.return_value
        tl = mock_tl_cls.return_value
        ds = mock_ds_cls.return_value

        # Minimal init caches
        prod.find_solar_irradiance.return_value = {0: 100.0}
        prod.find_physical_gains.return_value = {0: 1.0}
        ds.find_absolute_calibration_accuracy.return_value = {0: 0.01}
        ds.find_cross_band_calibration_accuracy.return_value = {0: 0.02}
        ds.find_multi_temporal_calibration_accuracy.return_value = {0: 0.03}
        ds.find_noise_model_alpha.return_value = {0: 1.0}
        ds.find_noise_model_beta.return_value = {0: 2.0}
        prod.find_reflectance_offsets.return_value = {0: 0}

        # get_basic_metadata deps
        prod.find_all_band_resolutions.return_value = {0: 10, 1: 20, 2: 60}
        prod.find_product_type.return_value = "MSIL1C"
        prod.find_product_uri.return_value = "S2A_TEST_PRODUCT"
        prod.find_spacecraft_name.return_value = "Sentinel-2A"
        prod.find_processing_level.return_value = "Level-1C"
        prod.find_processing_baseline.return_value = "05.11"

        bounds_obj = SimpleNamespace(wkt="POLYGON((...))")
        prod.find_bounds.return_value = bounds_obj
        tl.find_horizontal_cs_code.return_value = "EPSG:32630"

        prod.find_product_start_date.return_value = dt.date(2025, 11, 28)

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_basic_metadata()

        self.assertEqual(md["collection_name"], "MSIL1C")
        self.assertEqual(md["product_name"], "S2A_TEST_PRODUCT")
        self.assertEqual(md["platform"], "Sentinel-2A")
        self.assertEqual(md["instrument"], "MSI")
        self.assertEqual(md["processing_level"], "Level-1C")
        self.assertEqual(md["processing_version"], "05.11")

        self.assertEqual(md["spatial_resolution"], [10, 20, 60])
        self.assertEqual(md["geometry_id"], ["10m", "20m", "60m"])

        self.assertEqual(md["geospatial_bounds"], "POLYGON((...))")
        self.assertEqual(md["product_date"], dt.date(2025, 11, 28))
        self.assertIn("description", md)
        self.assertIn("footprint", md)
        self.assertIsInstance(md["footprint"], dict)
        self.assertEqual(md["footprint"]["crs"], "EPSG:32630")

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_product_metadata(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        reader = self._make_reader_with_layout()

        prod = mock_prod_cls.return_value
        tl = mock_tl_cls.return_value
        ds = mock_ds_cls.return_value

        # Minimal init caches
        prod.find_solar_irradiance.return_value = {0: 100.0}
        prod.find_physical_gains.return_value = {0: 1.0}
        ds.find_absolute_calibration_accuracy.return_value = {0: 0.01}
        ds.find_cross_band_calibration_accuracy.return_value = {0: 0.02}
        ds.find_multi_temporal_calibration_accuracy.return_value = {0: 0.03}
        ds.find_noise_model_alpha.return_value = {0: 1.0}
        ds.find_noise_model_beta.return_value = {0: 2.0}
        prod.find_reflectance_offsets.return_value = {0: 0}

        # get_product_metadata deps
        prod.find_orbit_number.return_value = 137
        prod.find_orbit_direction.return_value = "ASCENDING"
        prod.find_quantification_values.return_value = 10000
        prod.find_reflectance_conversion_u.return_value = 0.987

        tl.find_horizontal_cs_code.return_value = "EPSG:32630"
        tl.find_tile_id.return_value = "T30UXE"
        tl.find_datastrip_id.return_value = "DS_ABC"
        tl.find_downlink_priority.return_value = "NOMINAL"
        tl.find_sensing_datetime.return_value = dt.datetime(2025, 11, 28, 11, 14, 31, tzinfo=dt.timezone.utc)
        tl.find_cloudy_pixel_percentage.return_value = 12.3
        tl.find_snow_pixel_percentage.return_value = 0.0
        tl.find_degraded_msi_data_percentage.return_value = 0.5

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_product_metadata()

        self.assertEqual(md["orbit_number"], 137)
        self.assertEqual(md["orbit_direction"], "ASCENDING")
        self.assertEqual(md["quantification_level"], 10000)
        self.assertEqual(md["horizontal_cs_code"], "EPSG:32630")
        self.assertEqual(md["tile_id"], "T30UXE")
        self.assertEqual(md["datastrip_id"], "DS_ABC")
        self.assertEqual(md["downlink_priority"], "NOMINAL")
        self.assertAlmostEqual(md["cloudy_pixel_percentage"], 12.3)
        self.assertAlmostEqual(md["snow_pixel_percentage"], 0.0)
        self.assertAlmostEqual(md["degraded_msi_data_percentage"], 0.5)
        self.assertAlmostEqual(md["reflectance_conversion_u"], 0.987)

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_product_metadata_valid_band(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        reader = self._make_reader_with_layout()

        prod = mock_prod_cls.return_value
        tl = mock_tl_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {1: 200.0}  # B02 => band_id 1
        prod.find_physical_gains.return_value = {1: 1.23}
        ds.find_absolute_calibration_accuracy.return_value = {1: 0.01}
        ds.find_cross_band_calibration_accuracy.return_value = {1: 0.02}
        ds.find_multi_temporal_calibration_accuracy.return_value = {1: 0.03}
        ds.find_noise_model_alpha.return_value = {1: 7.0}
        ds.find_noise_model_beta.return_value = {1: 8.0}
        prod.find_reflectance_offsets.return_value = {"B02": -1000}

        # Per-band methods
        prod.find_band_resolution.return_value = 10
        prod.find_band_central_wavelength.return_value = 492.4
        prod.find_solar_irradiance_unit.return_value = "W/m²/µm"

        tl.find_tile_shape.return_value = (10980, 10980)
        tl.find_geoposition.return_value = {"x": 600000, "y": 5400000}

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_variable_product_metadata("B02")

        self.assertEqual(md["band_id"], 1)
        self.assertEqual(md["spatial_resolution"], 10)
        self.assertEqual(md["geometry_id"], "10m")
        self.assertEqual(md["band_central_wavelength"], 492.4)
        self.assertEqual(md["solar_irradiance"], 200.0)
        self.assertEqual(md["solar_irradiance_unit"], "W/m²/µm")
        self.assertEqual(md["radiometric_offset"], -1000)
        self.assertEqual(md["physical_gains"], 1.23)
        self.assertEqual(md["tile_shape"], (10980, 10980))
        self.assertEqual(md["geoposition"], {"x": 600000, "y": 5400000})
        self.assertEqual(md["absolute_calibration_accuracy"], 0.01)
        self.assertEqual(md["cross_band_calibration_accuracy"], 0.02)
        self.assertEqual(md["multi_temporal_calibration_accuracy"], 0.03)
        self.assertEqual(md["noise_model_alpha"], 7.0)
        self.assertEqual(md["noise_model_beta"], 8.0)

        # Ensure the correct band_id was used for resolution + TL lookups
        prod.find_band_resolution.assert_called_with(1)
        prod.find_band_central_wavelength.assert_called_with(1)
        tl.find_tile_shape.assert_called_with(10)
        tl.find_geoposition.assert_called_with(10)

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_basic_metadata_l1c_band(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """Test get_variable_basic_metadata for L1C measurement band with ancillary vars."""
        reader = self._make_reader_with_layout()
        reader.aux_def = ["solar_zenith_angle_B02", "view_zenith_angle"]

        prod = mock_prod_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {1: 200.0}
        prod.find_physical_gains.return_value = {1: 1.0}
        ds.find_absolute_calibration_accuracy.return_value = {1: 0.01}
        ds.find_cross_band_calibration_accuracy.return_value = {1: 0.02}
        ds.find_multi_temporal_calibration_accuracy.return_value = {1: 0.03}
        ds.find_noise_model_alpha.return_value = {1: 7.0}
        ds.find_noise_model_beta.return_value = {1: 8.0}
        prod.find_reflectance_offsets.return_value = {"B02": 0}

        prod.find_product_type.return_value = "S2MSI1C"

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_variable_basic_metadata("B02")

        self.assertEqual(md["standard_name"], "toa_reflectance")
        self.assertEqual(md["long_name"], "TOA HCRF")
        self.assertEqual(md["units"], "1")
        self.assertIn("ancillary_variables", md)
        # Should include variables that match "B02" or contain "solar"
        self.assertIn("solar_zenith_angle_B02", md["ancillary_variables"])

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_basic_metadata_l2a_band(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """Test get_variable_basic_metadata for L2A measurement band."""
        reader = self._make_reader_with_layout()
        reader.aux_def = []

        prod = mock_prod_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {1: 200.0}
        prod.find_physical_gains.return_value = {1: 1.0}
        ds.find_absolute_calibration_accuracy.return_value = {1: 0.01}
        ds.find_cross_band_calibration_accuracy.return_value = {1: 0.02}
        ds.find_multi_temporal_calibration_accuracy.return_value = {1: 0.03}
        ds.find_noise_model_alpha.return_value = {1: 7.0}
        ds.find_noise_model_beta.return_value = {1: 8.0}
        prod.find_reflectance_offsets.return_value = {"B02": 0}

        prod.find_product_type.return_value = "S2MSI2A"

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_variable_basic_metadata("B02")

        self.assertEqual(md["standard_name"], "boa_reflectance")
        self.assertEqual(md["long_name"], "BOA HCRF")
        self.assertEqual(md["units"], "1")
        # No ancillary vars for this test
        self.assertNotIn("ancillary_variables", md)

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_basic_metadata_non_measurement_var(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """Test get_variable_basic_metadata for non-measurement variable (AOT, WVP, etc)."""
        reader = self._make_reader_with_layout()
        reader.aux_def = []

        prod = mock_prod_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {}
        prod.find_physical_gains.return_value = {}
        ds.find_absolute_calibration_accuracy.return_value = {}
        ds.find_cross_band_calibration_accuracy.return_value = {}
        ds.find_multi_temporal_calibration_accuracy.return_value = {}
        ds.find_noise_model_alpha.return_value = {}
        ds.find_noise_model_beta.return_value = {}
        prod.find_reflectance_offsets.return_value = {}

        prod.find_product_type.return_value = "S2MSI2A"

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_variable_basic_metadata("AOT")

        # Non-measurement variables return basic empty metadata
        self.assertEqual(md["units"], "")
        self.assertEqual(md["long_name"], "")
        self.assertEqual(md["standard_name"], "")
        self.assertNotIn("ancillary_variables", md)

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_basic_metadata_angle_var(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """Angle variables should expose populated standard metadata."""
        reader = self._make_reader_with_layout()
        reader.aux_def = []

        prod = mock_prod_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {}
        prod.find_physical_gains.return_value = {}
        ds.find_absolute_calibration_accuracy.return_value = {}
        ds.find_cross_band_calibration_accuracy.return_value = {}
        ds.find_multi_temporal_calibration_accuracy.return_value = {}
        ds.find_noise_model_alpha.return_value = {}
        ds.find_noise_model_beta.return_value = {}
        prod.find_reflectance_offsets.return_value = {}

        ex = S2MSIMetadataExtractor(reader)

        md = ex.get_variable_basic_metadata("solar_zenith_angle")
        self.assertEqual(md["units"], "degrees")
        self.assertEqual(md["standard_name"], "solar_zenith_angle")
        self.assertEqual(md["measurand"], "angle")
        self.assertEqual(md["long_name"], "solar zenith angle")

        md = ex.get_variable_basic_metadata("viewing_zenith_angle_B02")
        self.assertEqual(md["measurand"], "viewing_zenith_angle")

        md = ex.get_variable_basic_metadata("viewing_azimuth_angle_B02")
        self.assertEqual(md["measurand"], "viewing_azimuth_angle")

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_basic_metadata_tcwv_passthrough(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """AUX GRIB vars should not be overwritten by empty metadata placeholders."""
        reader = self._make_reader_with_layout()
        reader.aux_def = ["tcwv"]

        prod = mock_prod_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {}
        prod.find_physical_gains.return_value = {}
        ds.find_absolute_calibration_accuracy.return_value = {}
        ds.find_cross_band_calibration_accuracy.return_value = {}
        ds.find_multi_temporal_calibration_accuracy.return_value = {}
        ds.find_noise_model_alpha.return_value = {}
        ds.find_noise_model_beta.return_value = {}
        prod.find_reflectance_offsets.return_value = {}

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_variable_basic_metadata("tcwv")

        # Empty dict means attach_metadata leaves existing GRIB attrs intact
        self.assertEqual(md, {})

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_product_metadata_aux_ecmwf_vars(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """AUX ECMWF variables should have standard 5000m resolution metadata."""
        reader = self._make_reader_with_layout()

        prod = mock_prod_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {}
        prod.find_physical_gains.return_value = {}
        ds.find_absolute_calibration_accuracy.return_value = {}
        ds.find_cross_band_calibration_accuracy.return_value = {}
        ds.find_multi_temporal_calibration_accuracy.return_value = {}
        ds.find_noise_model_alpha.return_value = {}
        ds.find_noise_model_beta.return_value = {}
        prod.find_reflectance_offsets.return_value = {}

        ex = S2MSIMetadataExtractor(reader)

        # Test ECMWF variables (old and new baselines)
        for var in ["tcwv", "tco3", "msl", "u10", "v10", "r"]:
            md = ex.get_variable_product_metadata(var)
            self.assertEqual(md["spatial_resolution"], 5000)
            self.assertEqual(md["spatial_resolution_unit"], "m")
            self.assertEqual(md["geometry_id"], "5000m")

        # Test CAMS variables
        for var in ["aod550", "z", "bcaod550"]:
            md = ex.get_variable_product_metadata(var)
            self.assertEqual(md["spatial_resolution"], 5000)
            self.assertEqual(md["spatial_resolution_unit"], "m")
            self.assertEqual(md["geometry_id"], "5000m")

    @patch(
        "eoio.readers.sentinel2.metadata.extractor.BaseMetadataExtractor.__init__",
        autospec=True,
        return_value=None,
    )
    @patch("eoio.readers.sentinel2.metadata.extractor.S2DSXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2TLXMLReader", autospec=True)
    @patch("eoio.readers.sentinel2.metadata.extractor.S2ProdXMLReader", autospec=True)
    def test_get_variable_basic_metadata_multiple_ancillary_vars(
        self,
        mock_prod_cls,
        mock_tl_cls,
        mock_ds_cls,
        _mock_base_init,
    ):
        """Test that ancillary variables are correctly matched for a band."""
        reader = self._make_reader_with_layout()
        # Multiple ancillary vars, some matching B03, some with "solar"
        reader.aux_def = [
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "view_zenith_angle_B03",
            "view_azimuth_angle_B03",
            "wind_speed",
        ]

        prod = mock_prod_cls.return_value
        ds = mock_ds_cls.return_value

        # Init caches
        prod.find_solar_irradiance.return_value = {2: 250.0}
        prod.find_physical_gains.return_value = {2: 1.0}
        ds.find_absolute_calibration_accuracy.return_value = {2: 0.01}
        ds.find_cross_band_calibration_accuracy.return_value = {2: 0.02}
        ds.find_multi_temporal_calibration_accuracy.return_value = {2: 0.03}
        ds.find_noise_model_alpha.return_value = {2: 7.0}
        ds.find_noise_model_beta.return_value = {2: 8.0}
        prod.find_reflectance_offsets.return_value = {"B03": 0}

        prod.find_product_type.return_value = "S2MSI1C"

        ex = S2MSIMetadataExtractor(reader)
        md = ex.get_variable_basic_metadata("B03")

        self.assertIn("ancillary_variables", md)
        anc_vars = md["ancillary_variables"]
        # Should contain solar vars and B03-specific vars
        self.assertIn("solar_zenith_angle", anc_vars)
        self.assertIn("solar_azimuth_angle", anc_vars)
        self.assertIn("view_zenith_angle_B03", anc_vars)
        self.assertIn("view_azimuth_angle_B03", anc_vars)
        # Should NOT contain wind_speed
        self.assertNotIn("wind_speed", anc_vars)


class TestS2MetadataExtractorData(unittest.TestCase):
    def test_real_safe(self):
        # Note: This test requires a real Sentinel-2 SAFE product at the specified path.
        # Update the SAFE_PATH to point to a valid SAFE product on your system for testing.
        safe_path = Path(SAFE_PATH)
        if not safe_path.exists():
            self.skipTest(f"Test SAFE product not found at {safe_path}")

        reader = S2MSIReader(safe_path)

        metadata_extractor = S2MSIMetadataExtractor(reader)

        _basic_prod_mtd = metadata_extractor.get_basic_metadata()
        _prod_mtd = metadata_extractor.get_product_metadata()
        _basic_var_mtd = metadata_extractor.get_variable_basic_metadata("B02")
        _var_mtd = metadata_extractor.get_variable_product_metadata("B02")
        pass


if __name__ == "__main__":
    unittest.main()
