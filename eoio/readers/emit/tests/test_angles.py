"""Tests for eoio.readers.emit.angles"""

import unittest
import xarray as xr
import numpy as np
from eoio.readers.emit import angles


class TestAngles(unittest.TestCase):
    def test_add_angles_basic(self):
        ds = xr.Dataset()
        obs_ds = xr.Dataset(
            {
                "To-sensor zenith": (["y", "x"], np.ones((2, 2))),
                "To-sensor azimuth": (["y", "x"], np.ones((2, 2)) * 2),
                "To-sun zenith": (["y", "x"], np.ones((2, 2)) * 3),
                "To-sun azimuth": (["y", "x"], np.ones((2, 2)) * 4),
            }
        )
        angle_names = ["viewing_zenith_angle", "solar_zenith_angle"]
        result = angles.add_angles(ds, obs_ds, angle_names)
        self.assertIn("viewing_zenith_angle", result)
        self.assertIn("solar_zenith_angle", result)

    def test_add_angles_excludes_unrequested(self):
        ds = xr.Dataset()
        obs_ds = xr.Dataset(
            {
                "To-sensor zenith": (["y", "x"], np.ones((2, 2))),
                "To-sensor azimuth": (["y", "x"], np.ones((2, 2)) * 2),
                "To-sun zenith": (["y", "x"], np.ones((2, 2)) * 3),
                "To-sun azimuth": (["y", "x"], np.ones((2, 2)) * 4),
            }
        )
        angle_names = ["viewing_zenith_angle"]
        result = angles.add_angles(ds, obs_ds, angle_names)
        self.assertIn("viewing_zenith_angle", result)
        self.assertNotIn("solar_zenith_angle", result)
        self.assertNotIn("viewing_azimuth_angle", result)
        self.assertNotIn("solar_azimuth_angle", result)


if __name__ == "__main__":
    unittest.main()
