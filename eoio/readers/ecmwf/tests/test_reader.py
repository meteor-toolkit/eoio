"""eoio.readers.ecmwf.tests.test_reader - unit tests for eoio.readers.ecmwf.reader"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import xarray as xr

from eoio.readers.ecmwf.reader import ECMWFReader


class TestECMWFReader(unittest.TestCase):
    def _make_reader(self, metadata_level=None):
        reader = object.__new__(ECMWFReader)

        reader.ds_src = xr.Dataset(
            {
                "valid_time": ("x", [1, 2, 3]),
                "data": ("x", [4, 5, 6]),
            }
        )
        reader.config = SimpleNamespace(
            vars_sel={"meas": "all", "aux": None, "mask": None},
            subset={"roi": None, "roi_crs": 4326},
            read_params={"metadata_level": metadata_level},
        )
        reader.path = Path("/tmp/fake_ecmwf.nc")

        return reader

    @patch("eoio.readers.ecmwf.reader.read_dataset")
    @patch("eoio.readers.ecmwf.reader.GenericNetCDFMetadataExtractor")
    def test_open_dataset_renames_valid_time_and_returns_original_metadata(
        self, mock_metadata_extractor, mock_read_dataset
    ):
        reader = self._make_reader(metadata_level="original")

        ds_after_rename = xr.Dataset(
            {
                "datetime": ("x", [1, 2, 3]),
                "data": ("x", [4, 5, 6]),
            }
        )
        mock_read_dataset.return_value = ds_after_rename

        result = reader.open_dataset()

        self.assertIs(result, ds_after_rename)
        mock_read_dataset.assert_called_once()
        read_call_kwargs = mock_read_dataset.call_args.kwargs
        self.assertIsNone(read_call_kwargs["subset"].wavelength_indices)
        self.assertIsNone(read_call_kwargs["subset"].datetime_indices)
        self.assertIsNone(read_call_kwargs["subset"].roi_subset)
        self.assertEqual(read_call_kwargs["include_vars"], None)
        self.assertNotIn("valid_time", read_call_kwargs["ds"].variables)
        self.assertIn("datetime", read_call_kwargs["ds"].variables)
        mock_metadata_extractor.assert_not_called()

    @patch("eoio.readers.ecmwf.reader.read_dataset")
    @patch("eoio.readers.ecmwf.reader.GenericNetCDFMetadataExtractor")
    def test_open_dataset_attaches_metadata_when_requested(self, mock_metadata_extractor, mock_read_dataset):
        reader = self._make_reader(metadata_level="all")

        ds_after_rename = xr.Dataset(
            {
                "datetime": ("x", [1, 2, 3]),
                "data": ("x", [4, 5, 6]),
            }
        )
        mock_read_dataset.return_value = ds_after_rename

        metadata_instance = mock_metadata_extractor.return_value
        metadata_instance.clear_metadata.return_value = ds_after_rename
        metadata_instance.attach_metadata.return_value = xr.Dataset(
            {
                "datetime": ("x", [1, 2, 3]),
                "data": ("x", [4, 5, 6]),
                "meta": ("x", [7, 8, 9]),
            }
        )

        result = reader.open_dataset()

        self.assertIs(result, metadata_instance.attach_metadata.return_value)
        mock_metadata_extractor.assert_called_once_with(reader, ds_after_rename)
        metadata_instance.clear_metadata.assert_called_once_with(ds_after_rename)
        metadata_instance.attach_metadata.assert_called_once_with(
            metadata_instance.clear_metadata.return_value,
            level="all",
        )

    @patch("eoio.readers.ecmwf.reader.xr.open_dataset")
    def test_init_uses_fallback_filename_when_original_path_missing(self, mock_open_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "data_stream-oper_stepType-instant.nc"
            fallback_path.write_text("")

            requested_path = Path(tmpdir) / "missing_product.nc"
            reader = ECMWFReader(str(requested_path))

            self.assertEqual(reader.path, fallback_path)
            mock_open_dataset.assert_called_once_with(str(fallback_path))


if __name__ == "__main__":
    unittest.main()
