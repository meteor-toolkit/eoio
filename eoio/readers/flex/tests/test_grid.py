"""Tests for eoio.readers.flex.grid"""

import unittest

import numpy as np

from eoio.readers.flex.grid import FlexGrid

ATTRS = {"GeoTransform": "399960.0 300.0 0.0 4700040.0 0.0 -300.0", "spatial_ref": "EPSG:32631"}


class TestFlexGrid(unittest.TestCase):
    def test_coordinates_and_bounds(self):
        grid = FlexGrid.from_attrs(ATTRS, nx=4, ny=3)
        np.testing.assert_allclose(grid.x, [400110, 400410, 400710, 401010])
        np.testing.assert_allclose(grid.y, [4699890, 4699590, 4699290])
        self.assertEqual(grid.bounds, (399960.0, 4699140.0, 401160.0, 4700040.0))
        self.assertEqual(grid.epsg, 32631)

    def test_crs_follows_file(self):
        grid = FlexGrid.from_attrs({**ATTRS, "spatial_ref": "EPSG:32632"}, nx=2, ny=2)
        self.assertEqual(grid.epsg, 32632)

    def test_missing_attr_raises(self):
        with self.assertRaises(ValueError):
            FlexGrid.from_attrs({"spatial_ref": "EPSG:32631"}, nx=2, ny=2)

    def test_rotated_grid_raises(self):
        with self.assertRaises(ValueError):
            FlexGrid.from_attrs({**ATTRS, "GeoTransform": "0 300 5 0 0 -300"}, nx=2, ny=2)


if __name__ == "__main__":
    unittest.main()
