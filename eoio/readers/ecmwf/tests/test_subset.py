"""eoio.readers.ecmwf.tests.test_subset - unit tests for eoio.readers.ecmwf.subset"""

import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from eoio.deps import lazy_rioxarray
from eoio.readers.ecmwf.subset import ECMWFSubset, build_subset

TIMES = np.array(["2025-07-01", "2025-07-02", "2025-07-03"], dtype="datetime64[ns]")


def _make_ds():
    """Mirror what ECMWFReader passes in: `datetime` as a data variable, post-rename."""
    return xr.Dataset({"datetime": ("x", TIMES), "tcwv": ("x", [1.0, 2.0, 3.0])})


class TestBuildSubset(unittest.TestCase):
    def test_no_constraints_resolves_to_nothing(self):
        result = build_subset(_make_ds(), {"roi": None, "roi_crs": 4326, "datetime": None})
        self.assertIsInstance(result, ECMWFSubset)
        self.assertIsNone(result.datetime_indices)
        self.assertIsNone(result.roi_subset)

    def test_missing_keys_are_tolerated(self):
        """Absent keys mean 'no constraint' rather than KeyError."""
        result = build_subset(_make_ds(), {})
        self.assertIsNone(result.datetime_indices)
        self.assertIsNone(result.roi_subset)

    def test_datetime_range_resolves_indices(self):
        result = build_subset(_make_ds(), {"datetime": {"min": "2025-07-02", "max": "2025-07-03"}})
        self.assertTrue(np.array_equal(np.asarray(result.datetime_indices), np.array([1, 2])))

    @patch("eoio.readers.ecmwf.subset.ROISubsetResolver")
    def test_roi_resolves_without_a_datetime_coordinate(self, mock_resolver):
        """ROI no longer depends on the dataset carrying a time coordinate.

        The gate used to require ``"datetime" in ds.keys()``, copy-pasted from
        the datetime block above it, which silently dropped the ROI for any
        dataset without one.
        """
        lazy_rioxarray()  # registers the .rio accessor, as build_subset does
        ds = xr.Dataset({"tcwv": (("y", "x"), np.ones((2, 2)))}).rio.write_crs("EPSG:4326")
        mock_resolver.return_value.run.return_value = "resolved"

        result = build_subset(ds, {"roi": (0, 0, 1, 1), "roi_crs": 4326})

        self.assertEqual(result.roi_subset, "resolved")
        self.assertEqual(mock_resolver.call_args.kwargs["roi"], (0, 0, 1, 1))
        self.assertEqual(mock_resolver.call_args.kwargs["roi_crs_epsg"], 4326)


if __name__ == "__main__":
    unittest.main()
