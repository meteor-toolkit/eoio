"""eoio.readers.modis.metadata.tests.test_metadata - tests for eoio.readers.modis.metadata"""

import unittest
from unittest.mock import MagicMock, patch
from eoio.readers.modis.metadata.extractor import MODISMetadataExtractor


class TestMODISMetadataExtractor(unittest.TestCase):
    def setUp(self) -> None:
        # Create a minimal reader mock
        self.reader = MagicMock()
        self.reader.layout = MagicMock()
        self.reader.layout.processing_level = "L1B"
        self.reader.layout.file_res_key = "H"

    def test_init_creates_extractor(self):
        """Test that MODISMetadataExtractor initializes correctly."""
        with patch("eoio.readers.modis.metadata.extractor.modis_prod_mtd_reader_factory"):
            extractor = MODISMetadataExtractor(self.reader)

            self.assertIsNotNone(extractor)
            self.assertEqual(extractor.layout, self.reader.layout)
            self.assertEqual(extractor.reader, self.reader)

    def test_get_basic_metadata_returns_dict(self):
        """Test that get_basic_metadata returns a dictionary."""
        with patch("eoio.readers.modis.metadata.extractor.modis_prod_mtd_reader_factory") as mock_factory:
            mock_reader = MagicMock()
            mock_reader.attrs = {
                "LONGNAME": "Test Product",
                "SHORTNAME": "TEST",
                "PRODUCT_NAME": "MOD02HDF",
                "ASSOCIATEDPLATFORMSHORTNAME.1": "Terra",
                "ASSOCIATEDINSTRUMENTSHORTNAME.1": "MODIS",
                "RANGEBEGINNINGDATE": "2020-01-01",
            }
            mock_factory.return_value = mock_reader

            extractor = MODISMetadataExtractor(self.reader)
            metadata = extractor.get_basic_metadata()

            self.assertIsInstance(metadata, dict)
            self.assertIn("title", metadata)
            self.assertIn("processing_level", metadata)

    def test_get_basic_metadata_includes_processing_level(self):
        """Test that basic metadata includes processing level."""
        with patch("eoio.readers.modis.metadata.extractor.modis_prod_mtd_reader_factory") as mock_factory:
            mock_reader = MagicMock()
            mock_reader.attrs = {
                "LONGNAME": "Test",
                "SHORTNAME": "TST",
                "PRODUCT_NAME": "MOD02",
                "ASSOCIATEDPLATFORMSHORTNAME.1": "Terra",
                "ASSOCIATEDINSTRUMENTSHORTNAME.1": "MODIS",
                "RANGEBEGINNINGDATE": "2020-01-01",
            }
            mock_factory.return_value = mock_reader

            self.reader.layout.processing_level = "L1B"
            extractor = MODISMetadataExtractor(self.reader)
            metadata = extractor.get_basic_metadata()

            self.assertEqual(metadata["processing_level"], "L1B")

    def test_get_basic_metadata_includes_eoio_reader(self):
        """Every other reader (hypernets, radcalnet, emit, generic_netcdf, landsat,
        planetscope, sentinel2/3_olci/slstr, airbus_pleiades) sets "eoio:reader" in
        get_basic_metadata; MODIS was missing it entirely."""
        with patch("eoio.readers.modis.metadata.extractor.modis_prod_mtd_reader_factory") as mock_factory:
            mock_reader = MagicMock()
            mock_reader.attrs = {
                "LONGNAME": "Test",
                "SHORTNAME": "TST",
                "PRODUCT_NAME": "MOD02",
                "ASSOCIATEDPLATFORMSHORTNAME.1": "Terra",
                "ASSOCIATEDINSTRUMENTSHORTNAME.1": "MODIS",
                "RANGEBEGINNINGDATE": "2020-01-01",
                "RANGEBEGINNINGTIME": "00:00:00",
            }
            mock_factory.return_value = mock_reader

            extractor = MODISMetadataExtractor(self.reader)
            metadata = extractor.get_basic_metadata()

            self.assertEqual(metadata["eoio:reader"], "modis")

    def test_get_basic_metadata_product_datetime(self):
        """product_datetime should combine date and time."""
        with patch("eoio.readers.modis.metadata.extractor.modis_prod_mtd_reader_factory") as mock_factory:
            mock_reader = MagicMock()
            mock_reader.attrs = {
                "LONGNAME": "Test",
                "SHORTNAME": "TST",
                "PRODUCT_NAME": "MOD02",
                "ASSOCIATEDPLATFORMSHORTNAME.1": "Terra",
                "ASSOCIATEDINSTRUMENTSHORTNAME.1": "MODIS",
                "RANGEBEGINNINGDATE": "2020-01-01",
                "RANGEBEGINNINGTIME": "03:04:05",
            }
            mock_factory.return_value = mock_reader

            extractor = MODISMetadataExtractor(self.reader)
            metadata = extractor.get_basic_metadata()

            self.assertEqual(metadata["product_datetime"], "2020-01-01T03:04:05")

    def test_meas_var_band_ids_set_correctly(self):
        """Test that measurement variable band IDs are set during initialization."""
        with patch("eoio.readers.modis.metadata.extractor.modis_prod_mtd_reader_factory"):
            extractor = MODISMetadataExtractor(self.reader)

            # Should have meas_var_band_ids attribute
            self.assertIsNotNone(extractor.meas_var_band_ids)


if __name__ == "__main__":
    unittest.main()
