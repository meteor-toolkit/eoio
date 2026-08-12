"""
eoio.readers.sentinel2.tests.test_conventions - tests for eoio.readers.sentinel2.conventions
"""

from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import xarray as xr
from eoio.readers.sentinel2.conventions import apply_conventions


class TestApplyConventions(unittest.TestCase):
    def setUp(self) -> None:
        # layout is currently unused by apply_conventions, but required by the signature
        self.layout = MagicMock(name="S2Layout")

    def test_sets_attrs_when_no_subset(self):
        ds = xr.Dataset()
        config = SimpleNamespace(subset=None)

        out = apply_conventions(ds, layout=self.layout, config=config)

        # Function updates in-place and returns the same Dataset object
        self.assertIs(out, ds)

        self.assertEqual(out.attrs["eoio:reader"], "sentinel2")
        self.assertIn("eoio:subset", out.attrs)
        # "" (not None) -- an attr value of None can't be written to netCDF.
        self.assertEqual(out.attrs["eoio:subset"], "")

    def test_sets_attrs_when_subset_present(self):
        ds = xr.Dataset()
        subset = SimpleNamespace(clip_box=(0.0, 1.0, 2.0, 3.0))
        config = SimpleNamespace(subset=subset)

        out = apply_conventions(ds, layout=self.layout, config=config)

        self.assertIs(out, ds)
        self.assertEqual(out.attrs["eoio:reader"], "sentinel2")
        self.assertEqual(out.attrs["eoio:subset"], (0.0, 1.0, 2.0, 3.0))

    def test_preserves_existing_attrs(self):
        ds = xr.Dataset(attrs={"existing": "keep_me", "eoio:reader": "old_value"})
        config = SimpleNamespace(subset=None)

        out = apply_conventions(ds, layout=self.layout, config=config)

        # existing attrs remain, eoio:reader overwritten as per conventions
        self.assertEqual(out.attrs["existing"], "keep_me")
        self.assertEqual(out.attrs["eoio:reader"], "sentinel2")
        self.assertIn("eoio:subset", out.attrs)


if __name__ == "__main__":
    unittest.main()
