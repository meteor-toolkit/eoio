"""eoio.readers.sentinel2.aux_vars.tests.test_angles - tests for eoio.readers.sentinel2.aux_vars.angles"""

import unittest
from unittest.mock import Mock
import numpy as np
import xarray as xr
from eoio.readers.sentinel2.aux_vars.angles import add_angles


class TestAddAngles(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.mtd = self._make_mock_mtd()

    def _make_mock_mtd(self) -> Mock:
        """
        Minimal mock of S2MSIMetadataExtractor and its tl_xml_reader.
        """
        mtd = Mock()

        tl = Mock()
        mtd.tl_xml_reader = tl

        # Sun angle grid metadata
        tl.find_sun_angle_steps.return_value = (5000, 5000)

        # Solar angle grid: 2D (rows, cols)
        solar_grid = np.ones((3, 4), dtype=float)
        tl.find_sun_angle_grid.return_value = solar_grid

        # Viewing angle grid: 3D (detector, rows, cols)
        viewing_grid = np.ones((2, 3, 4), dtype=float)
        detector_ids = np.array([1, 2])
        tl.find_viewing_angle_grid.return_value = (viewing_grid, detector_ids)

        # Geoposition (10 m band, shared origin)
        mtd.variable_product_metadata.return_value = {
            "geoposition": {
                "ulx": 600000.0,
                "uly": 5100000.0,
            }
        }

        return mtd

    def test_add_solar_angle(self):
        out = add_angles(
            ds=self.ds,
            angle_names=["solar_zenith_angle"],
            mtd=self.mtd,
        )

        self.assertIn("solar_zenith_angle", out)

        da = out["solar_zenith_angle"]
        self.assertEqual(da.dims, ("y_5000m", "x_5000m"))
        self.assertEqual(da.shape, (3, 4))

        # Coordinate sanity
        self.assertEqual(da.coords["x_5000m"].size, 4)
        self.assertEqual(da.coords["y_5000m"].size, 3)

    def test_add_viewing_angle_no_average(self):
        out = add_angles(
            ds=self.ds,
            angle_names=["viewing_zenith_angle_B02"],
            mtd=self.mtd,
            ave_det=False,
        )

        da = out["viewing_zenith_angle_B02"]
        self.assertEqual(da.dims, ("detector", "y_5000m", "x_5000m"))
        self.assertEqual(da.shape, (2, 3, 4))
        self.assertIn("detector", da.coords)
        np.testing.assert_array_equal(da.coords["detector"].values, np.array([1, 2]))

    def test_add_viewing_angle_with_average(self):
        out = add_angles(
            ds=self.ds,
            angle_names=["viewing_azimuth_angle_B02"],
            mtd=self.mtd,
            ave_det=True,
        )

        da = out["viewing_azimuth_angle_B02"]

        # Detector dimension should be gone
        self.assertEqual(da.dims, ("y_5000m", "x_5000m"))
        self.assertEqual(da.shape, (3, 4))

        # Mean of ones should still be ones
        np.testing.assert_allclose(da.values, 1.0)

    def test_multiple_angles_share_same_coords(self):
        out = add_angles(
            ds=self.ds,
            angle_names=["solar_zenith_angle", "viewing_zenith_angle_B02"],
            mtd=self.mtd,
        )

        solar = out["solar_zenith_angle"]
        viewing = out["viewing_zenith_angle_B02"]

        np.testing.assert_array_equal(solar.coords["x_5000m"].values, viewing.coords["x_5000m"].values)
        np.testing.assert_array_equal(solar.coords["y_5000m"].values, viewing.coords["y_5000m"].values)

    def test_mutates_in_place(self):
        out = add_angles(
            ds=self.ds,
            angle_names=["solar_zenith_angle"],
            mtd=self.mtd,
        )

        # Function should mutate the provided dataset in place and return it
        self.assertIs(out, self.ds)
        self.assertIn("solar_zenith_angle", self.ds)


if __name__ == "__main__":
    unittest.main()
