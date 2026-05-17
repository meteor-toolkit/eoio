"""Tests for eoio.readers.emit.subset"""

import unittest
import xarray as xr
from eoio.readers.emit import subset
from eoio.readers.emit.subset import EMITSubset


class DummyROISubset:
    pass


class DummyROISubsetResolver:
    def __init__(self, roi, roi_crs, crs, bounds):
        pass

    def run(self):
        return DummyROISubset()


class TestSubset(unittest.TestCase):
    def test_build_subset_roi(self):
        ds = xr.Dataset(
            {
                "latitude": (["y", "x"], [[1, 2], [3, 4]]),
                "longitude": (["y", "x"], [[10, 20], [30, 40]]),
            }
        )
        subset_dict = {"roi": None, "roi_crs": 4326}
        subset.ROISubsetResolver = DummyROISubsetResolver
        result = subset.build_subset(ds, subset_dict)
        self.assertIsInstance(result, EMITSubset)
        assert hasattr(result, "roi")

    def test_build_subset_roi_with_wavelength_dim(self):
        ds = xr.Dataset(
            {
                "latitude": (["wavelength", "y", "x"], [[[1, 2], [3, 4]]]),
                "longitude": (["wavelength", "y", "x"], [[[10, 20], [30, 40]]]),
                "wavelength": (["wavelength"], [800]),
            }
        )
        subset_dict = {"wavelength": {"min": 500, "max": 1000}}
        subset.ROISubsetResolver = DummyROISubsetResolver
        result = subset.build_subset(ds, subset_dict)
        self.assertIsInstance(result, EMITSubset)
        assert hasattr(result, "wavelength_indices")


if __name__ == "__main__":
    unittest.main()
