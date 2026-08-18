import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import xarray as xr

from eoio.readers.ESAWorldCover.reader import ESAWorldCoverReader


class TestESAWorldCoverReader(unittest.TestCase):
    def setUp(self):
        self.reader = ESAWorldCoverReader.__new__(ESAWorldCoverReader)
        self.reader.path = Path("test.tif")

    def test_get_extension(self):
        self.assertEqual(ESAWorldCoverReader.get_extension(), ".tif")

    @patch("eoio.readers.ESAWorldCover.reader.ESAWorldCoverMetadataExtractor")
    @patch("eoio.readers.ESAWorldCover.reader.append_data_vars")
    @patch("eoio.readers.ESAWorldCover.reader.xr.open_dataset")
    def test_open_dataset_with_measurements_and_metadata(
        self,
        mock_open_dataset,
        mock_append_data_vars,
        mock_metadata_extractor,
    ):
        mock_open_dataset.return_value = xr.Dataset()

        ds_after_append = xr.Dataset({"landcover_map": (("y", "x"), [[1, 2], [3, 4]])})
        mock_append_data_vars.return_value = ds_after_append

        metadata_instance = MagicMock()
        metadata_instance.attach_metadata.return_value = ds_after_append
        mock_metadata_extractor.return_value = metadata_instance

        self.reader.resolved_config = MagicMock()
        self.reader.resolved_config.vars_sel = {"meas": ["landcover_map"]}
        self.reader.resolved_config.subset = None
        self.reader.resolved_config.read_params = {
            "metadata_level": "all",
            "use_chunks": False,
            "chunks": None,
        }

        result = self.reader.open_dataset()

        mock_append_data_vars.assert_called_once()
        metadata_instance.attach_metadata.assert_called_once_with(
            ds_after_append,
            level="all",
        )
        self.assertIs(result, ds_after_append)

    @patch("eoio.readers.ESAWorldCover.reader.ESAWorldCoverMetadataExtractor")
    @patch("eoio.readers.ESAWorldCover.reader.xr.open_dataset")
    def test_open_dataset_without_measurements(
        self,
        mock_open_dataset,
        mock_metadata_extractor,
    ):
        mock_open_dataset.return_value = xr.Dataset()

        _ = xr.Dataset()


if __name__ == "__main__":
    unittest.main()
