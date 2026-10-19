"""Tests for eoio.readers.flex.subset"""

import unittest

import numpy as np
import xarray as xr

from eoio.readers.flex.grid import FlexGrid
from eoio.readers.flex.subset import build_subset

GRID = FlexGrid(x0=400000.0, y0=4700000.0, dx=300.0, dy=300.0, nx=10, ny=10, crs="EPSG:32631")
WAVELENGTHS = {
    "wavelength_olci": xr.DataArray(np.array([400.0, 500.0, 600.0, 700.0]), dims="wavelength_olci"),
    "wavelength_sif": xr.DataArray(np.array([670.0, 700.0, 780.0]), dims="wavelength_sif"),
}


class TestBuildSubset(unittest.TestCase):
    def test_no_subset(self):
        s = build_subset(GRID, WAVELENGTHS, {"roi": None, "roi_crs": 4326, "wavelength": None})
        self.assertIsNone(s.x_indices)
        self.assertIsNone(s.y_indices)
        self.assertEqual(s.wavelength_indices, {})

    def test_roi_in_product_crs(self):
        # pixels overlapping x in [400400, 401000], y in [4698500, 4699000] (rows 3-5)
        s = build_subset(
            GRID, WAVELENGTHS, {"roi": (400400.0, 4698500.0, 401000.0, 4699000.0), "roi_crs": 32631, "wavelength": None}
        )
        np.testing.assert_array_equal(s.x_indices, [1, 2, 3])
        np.testing.assert_array_equal(s.y_indices, [3, 4, 5])

    def test_roi_outside_raises(self):
        with self.assertRaises(ValueError):
            build_subset(GRID, WAVELENGTHS, {"roi": (0.0, 0.0, 1000.0, 1000.0), "roi_crs": 32631, "wavelength": None})

    def test_wavelength_applied_to_each_grid(self):
        s = build_subset(GRID, WAVELENGTHS, {"roi": None, "roi_crs": 4326, "wavelength": {"min": 650, "max": 710}})
        np.testing.assert_array_equal(s.wavelength_indices["wavelength_olci"], [3])
        np.testing.assert_array_equal(s.wavelength_indices["wavelength_sif"], [0, 1])


if __name__ == "__main__":
    unittest.main()
