"""eoio.readers.generic_netcdf.tests.test_metadata - unit tests for eoio.readers.generic_netcdf.metadata"""

import unittest
import xarray as xr
from eoio.readers.generic_netcdf.metadata import GenericNetCDFMetadataExtractor


class TestGenericNetCDFMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset(
            {
                "var1": ("series", [1, 2]),
            },
            attrs={"title": "Test Product"},
        )
        self.reader = type(
            "DummyReader",
            (),
            {"config": type("Config", (), {"subset": {}}), "path": "test_path"},
        )()

    def test_get_product_metadata(self):
        extractor = GenericNetCDFMetadataExtractor(self.reader, self.ds)
        metadata = extractor.get_product_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertIn("title", self.ds.attrs)


if __name__ == "__main__":
    unittest.main()
