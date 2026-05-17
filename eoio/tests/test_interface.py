"""eoio.read - tests for eoio.read"""

import unittest
import unittest.mock as mock

import xarray as xr

from eoio.interface import (
    # mid_lon_lat,
    # product_bounds,
    read,
    process,
    # write,
)
from eoio.readers.base import BaseReader

__author__ = [
    "Sam Hunt <sam.hunt@npl.co.uk>",
    "Mattea Goalen <mattea.goalen@npl.co.uk>",
    "Maddie Stedman <maddie.stedman@npl.co.uk>",
]
__all__ = []


class MockReaderFactory:
    def get_reader(self, path):
        return


class MockReader(BaseReader):
    def __init__(self, path: str):
        self._all_options = None

    @staticmethod
    def get_extension() -> str:
        return ""

    def extract_metadata(self):
        pass

    @property
    def all_options(self):
        if self._all_options is None:
            self._all_options = {"options": "values"}
        return self._all_options

    def open_dataset(self) -> xr.Dataset:
        return xr.Dataset()


class TestInterface(unittest.TestCase):
    @mock.patch("eoio.interface.os.path.exists")
    @mock.patch("eoio.utils.read_utils.extract_file")
    @mock.patch("eoio.interface.ReaderFactory")
    def test_read(self, reader_factory_mock, mock_extract_file, mock_path_exists):
        mock_extract_file.return_value = "path_string", {}, False
        mock_path_exists.return_value = True

        reader_mock = mock.MagicMock()
        reader_mock.open.return_value = "test_ds"
        get_reader_mock = mock.MagicMock(return_value=reader_mock)
        reader_factory_mock().get_reader.return_value = get_reader_mock

        # reader_factory_mock().get_reader()().open_dataset.return_value = "test_ds"
        test_ds = read("path_string")
        self.assertEqual(test_ds, "test_ds")

        mock_extract_file.assert_called_once_with("path_string", None)
        mock_path_exists.assert_called_once_with("path_string")

    @mock.patch("eoio.interface.ProcessorPipeline")
    def test_process(self, processor_pipeline_mock):
        """
        process() should instantiate ProcessorPipeline, run it,
        and return the processed dataset.
        """
        ds = xr.Dataset({"a": ("x", [1, 2, 3])})
        processed_ds = xr.Dataset({"a": ("x", [2, 4, 6])})

        processors = {"dummy_processor": {"param": 1}}
        context = {"ctx": "value"}

        pipeline_instance = mock.MagicMock()
        pipeline_instance.run.return_value = processed_ds
        processor_pipeline_mock.return_value = pipeline_instance

        result = process(ds, processors, context)

        processor_pipeline_mock.assert_called_once_with(
            processor_params=processors,
            context=context,
        )
        pipeline_instance.run.assert_called_once_with(ds)
        self.assertIs(result, processed_ds)


if __name__ == "__main__":
    unittest.main()
