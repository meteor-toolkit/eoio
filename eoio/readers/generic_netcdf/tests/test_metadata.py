"""eoio.readers.generic_netcdf.tests.test_metadata - unit tests for eoio.readers.generic_netcdf.metadata"""

import json
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

    def test_product_datetime_passes_through_from_source_attrs(self):
        ds = xr.Dataset(
            {"var1": ("series", [1, 2])},
            attrs={"product_date": "2023-01-15", "product_datetime": "2023-01-15T10:30:45"},
        )
        extractor = GenericNetCDFMetadataExtractor(self.reader, ds)
        md = extractor.get_basic_metadata()
        self.assertEqual(md["product_date"], "2023-01-15")
        self.assertEqual(md["product_datetime"], "2023-01-15T10:30:45")

    def test_product_datetime_defaults_to_empty_string_when_absent(self):
        extractor = GenericNetCDFMetadataExtractor(self.reader, self.ds)
        md = extractor.get_basic_metadata()
        self.assertEqual(md["product_datetime"], "")

    def test_eoio_subset_is_empty_string_when_no_subset(self):
        # "" (not None) -- an attr value of None can't be written to netCDF.
        extractor = GenericNetCDFMetadataExtractor(self.reader, self.ds)
        md = extractor.get_basic_metadata()
        self.assertEqual(md["eoio:subset"], "")

    def test_eoio_subset_is_valid_json_when_subset_present(self):
        """Regression test: eoio:subset used to be repr(self.subset), Python-only
        syntax (e.g. single quotes) that isn't parseable JSON."""
        reader = type(
            "DummyReader",
            (),
            {
                "config": type("Config", (), {"subset": {"roi_crs_epsg": "EPSG:4326"}}),
                "path": "test_path",
            },
        )()
        extractor = GenericNetCDFMetadataExtractor(reader, self.ds)
        md = extractor.get_basic_metadata()
        parsed = json.loads(md["eoio:subset"])
        self.assertEqual(parsed["roi_crs_epsg"], "EPSG:4326")


if __name__ == "__main__":
    unittest.main()
