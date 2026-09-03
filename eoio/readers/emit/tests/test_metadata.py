"""Tests for eoio.readers.emit.metadata"""

import json
import unittest
import xarray as xr
from eoio.readers.emit import metadata


from unittest.mock import MagicMock


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset({"var": (["x"], [1, 2, 3])})
        self.ds.attrs["history"] = "created for test"
        self.ds["var"].attrs = {
            "long_name": "Variable",
            "standard_name": "var",
            "units": "1",
            "_FillValue": -9999,
            "add_offset": 0.0,
        }

    def _make_meta(self):
        meta = MagicMock(spec=metadata.EMITMetadataExtractor)
        meta.ds = self.ds
        meta.layout = MagicMock()
        meta.layout.path = "EMIT_L1B_OBS_001_20230729T113445_2321008_034"
        meta.subset = {"roi": [0, 1, 2, 3]}
        meta.reader = MagicMock()
        meta.reader.ds_src = MagicMock()
        meta.reader.ds_src.title = "title"
        meta.reader.ds_src.time_coverage_end = "2020-01-01T00:00:00"
        meta.reader.ds_src.summary = "summary"
        meta.reader.config = MagicMock()
        meta.reader.config.subset = {"roi": [0, 1, 2, 3]}
        meta.get_product_metadata = metadata.EMITMetadataExtractor.get_product_metadata.__get__(meta)
        meta.get_variable_product_metadata = metadata.EMITMetadataExtractor.get_variable_product_metadata.__get__(meta)
        meta.get_variable_basic_metadata = metadata.EMITMetadataExtractor.get_variable_basic_metadata.__get__(meta)
        meta.get_basic_metadata = metadata.EMITMetadataExtractor.get_basic_metadata.__get__(meta)
        return meta

    def test_get_product_metadata_basic(self):
        meta = self._make_meta()
        md = meta.get_product_metadata()
        self.assertIsInstance(md, dict)
        self.assertIn("history", md)

    def test_get_variable_product_metadata_basic(self):
        meta = self._make_meta()
        md = meta.get_variable_product_metadata("var")
        self.assertIsInstance(md, dict)
        self.assertNotIn("long_name", md)
        self.assertNotIn("_FillValue", md)

    def test_get_variable_basic_metadata_basic(self):
        meta = self._make_meta()
        md = meta.get_variable_basic_metadata("var")
        self.assertIsInstance(md, dict)
        self.assertIn("long_name", md)
        self.assertIn("units", md)
        self.assertIn("add_offset", md)

    def test_get_basic_metadata_basic(self):
        meta = self._make_meta()
        md = meta.get_basic_metadata()
        self.assertIsInstance(md, dict)
        self.assertIn("collection_name", md)
        self.assertIn("product_name", md)
        self.assertIn("platform", md)
        self.assertIn("product_bounds", md)
        self.assertIn("history", md)
        self.assertEqual(md["product_datetime"], "2020-01-01T00:00:00")

    def test_eoio_subset_is_valid_json(self):
        """Regression test: eoio:subset used to be repr(self.subset), Python-only
        syntax (e.g. single quotes) that isn't parseable JSON."""
        meta = self._make_meta()
        md = meta.get_basic_metadata()

        parsed = json.loads(md["eoio:subset"])
        self.assertEqual(parsed["roi"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
