"""eoio.readers.landsat.tests.test_conventions - unit tests for eoio.readers.landsat.conventions"""

import unittest
from unittest.mock import Mock
import xarray as xr
from eoio.readers.landsat.conventions import apply_conventions


class TestLandsatConventions(unittest.TestCase):
    def setUp(self):
        self.layout = Mock()
        self.roi_subset = Mock()
        self.roi_subset.clip_box = (0, 0, 1, 1)
        self.config = Mock()

    def test_apply_conventions_AddsProvenanceAttributes(self):
        ds = xr.Dataset()
        out = apply_conventions(ds, layout=self.layout, roi_subset=self.roi_subset, config=self.config)
        self.assertEqual(out.attrs.get("eoio:reader"), "landsat")

    def test_apply_conventions_ExistingAttributesOverwritten(self):
        # The current implementation uses update, so it overwrites.
        ds = xr.Dataset(attrs={"eoio:reader": "other"})
        out = apply_conventions(ds, layout=self.layout, roi_subset=self.roi_subset, config=self.config)
        self.assertEqual(out.attrs.get("eoio:reader"), "landsat")


if __name__ == "__main__":
    unittest.main()
