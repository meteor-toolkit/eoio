"""Tests for eoio.readers.ecmwf.subset"""

import unittest
import xarray as xr
from eoio.readers.ecmwf import subset


class DummyROISubset:
    pass


class DummyROISubsetResolver:
    def __init__(self, roi, roi_crs, crs, bounds):
        self.roi = roi
        self.roi_crs = roi_crs
        self.crs = crs
        self.bounds = bounds

    def run(self):
        return DummyROISubset()


class TestECMWFSubset(unittest.TestCase):
    def test_build_subset_returns_empty_dict_when_roi_none(self):
        ds = xr.Dataset(
            {
                "latitude": (("y", "x"), [[1, 2], [3, 4]]),
                "longitude": (("y", "x"), [[10, 20], [30, 40]]),
            }
        )
        subset_dict = {"roi": None, "roi_crs": 4326}

        result = subset.build_subset(ds, subset_dict)

        self.assertEqual(result, {})

    def test_build_subset_calls_resolver_and_returns_roi_subset(self):
        ds = xr.Dataset(
            {
                "latitude": (("y", "x"), [[1, 2], [3, 4]]),
                "longitude": (("y", "x"), [[10, 20], [30, 40]]),
            }
        )
        subset_dict = {"roi": ((0, 0), (1, 1)), "roi_crs": 4326}

        original_resolver = subset.ROISubsetResolver
        subset.ROISubsetResolver = DummyROISubsetResolver
        try:
            result = subset.build_subset(ds, subset_dict)
        finally:
            subset.ROISubsetResolver = original_resolver

        self.assertIsInstance(result, dict)
        self.assertIn("roi", result)
        self.assertIsInstance(result["roi"], DummyROISubset)


if __name__ == "__main__":
    unittest.main()
