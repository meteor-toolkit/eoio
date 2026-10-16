"""eoio.readers.constellr_skybee.tests.test_conventions - unit tests for eoio.readers.constellr_skybee.conventions"""

import unittest

import xarray as xr

from eoio.readers.constellr_skybee.conventions import apply_conventions


class TestSkybeeConventions(unittest.TestCase):
    def test_apply_conventions_sets_reader_and_subset(self):
        ds = xr.Dataset({"B1": ("y", [1, 2])})
        layout = object()
        roi_subset = type("Subset", (), {"clip_box": (0, 0, 1, 1)})()
        config = object()

        out = apply_conventions(ds, layout=layout, roi_subset=roi_subset, config=config)

        self.assertEqual(out.attrs["eoio:reader"], "constellr_skybee")
        self.assertEqual(out.attrs["eoio:subset"], (0, 0, 1, 1))

    def test_apply_conventions_handles_none_subset(self):
        ds = xr.Dataset({"B1": ("y", [1, 2])})
        layout = object()

        out = apply_conventions(ds, layout=layout, roi_subset=None, config=object())

        self.assertEqual(out.attrs["eoio:subset"], "")


if __name__ == "__main__":
    unittest.main()
