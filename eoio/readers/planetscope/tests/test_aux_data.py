"""eoio.readers.planetscope.tests.test_aux_data - tests for eoio.readers.planetscope.aux_data"""

import unittest
from unittest.mock import MagicMock, patch
import warnings
import numpy as np
import xarray as xr

from eoio.readers.planetscope.aux_data import (
    read_aux,
    add_masks,
    add_angles,
)


class TestReadAux(unittest.TestCase):
    """Unit tests for read_aux function."""

    def setUp(self) -> None:
        self.ds = xr.Dataset(
            {
                "B1": xr.DataArray(np.zeros((10, 10))),
                "B2": xr.DataArray(np.zeros((10, 10))),
            }
        )
        self.mtd = MagicMock(name="mtd")

    def test_read_aux_raises_when_aux_none(self):
        """Test that read_aux raises ValueError when aux is None."""
        with self.assertRaises(ValueError):
            read_aux(self.ds, None, self.mtd)

    def test_read_aux_raises_when_aux_empty(self):
        """Test that read_aux raises ValueError when aux_data is empty."""
        with self.assertRaises(ValueError):
            read_aux(self.ds, [], self.mtd)

    def test_read_aux_with_observation_geometry(self):
        """Test read_aux with observation_geometry enabled."""
        aux = ["observation_geometry"]
        with patch("eoio.readers.planetscope.aux_data.add_angles") as mock_angles:
            mock_angles.return_value = self.ds
            _result = read_aux(self.ds, aux, self.mtd)
            mock_angles.assert_called_once()

    def test_read_aux_with_masks_warns(self):
        """Test that read_aux warns when masks are requested."""
        aux = ["mask"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _result = read_aux(self.ds, aux, self.mtd)
            self.assertGreater(len(w), 0)
            self.assertTrue(any("Mask" in str(warning.message) for warning in w))

    def test_read_aux_warns_auxiliary_not_implemented(self):
        """Test that read_aux warns that auxiliary data is not fully implemented."""
        aux = ["mask"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _result = read_aux(self.ds, aux, self.mtd)
            self.assertGreater(len(w), 0)
            self.assertTrue(any("Auxiliary" in str(warning.message) for warning in w))


class TestAddMasks(unittest.TestCase):
    """Unit tests for add_masks function."""

    def setUp(self) -> None:
        self.ds = xr.Dataset(
            {
                "B1": xr.DataArray(np.zeros((10, 10))),
            }
        )

    def test_add_masks_returns_unchanged_dataset(self):
        """Test that add_masks returns the dataset unchanged - Placeholder test (not fully implemented)."""
        result = add_masks(ds=self.ds, layout=None, aux_names=None)
        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual(len(result.data_vars), len(self.ds.data_vars))
        self.assertEqual(list(result.data_vars), list(self.ds.data_vars))


class TestAddAngles(unittest.TestCase):
    """Unit tests for add_angles function."""

    def setUp(self) -> None:
        self.ds = xr.Dataset(
            {
                "B1": xr.DataArray(np.zeros((10, 10))),
            }
        )
        self.mtd = MagicMock(name="mtd")
        self.mtd.product_metadata = {
            "product_geospatial_bounds": [10.0, -20.0, 20.0, -10.0],
            "product_properties": {
                "sun_azimuth": 180.0,
                "sun_elevation": 45.0,
                "satellite_azimuth": 90.0,
                "view_angle": 30.0,
            },
        }

    def test_add_angles_returns_dataset(self):
        """Test that add_angles returns a Dataset."""
        result = add_angles(self.ds, self.mtd)
        self.assertIsInstance(result, xr.Dataset)

    def test_add_angles_adds_solar_zenith_angle(self):
        """Test that add_angles adds solar_zenith_angle variable."""
        result = add_angles(self.ds, self.mtd)
        self.assertIn("solar_zenith_angle", result)
        self.assertIn("solar_azimuth_angle", result)
        self.assertIn("sensor_zenith_angle", result)
        self.assertIn("sensor_azimuth_angle", result)

    def test_add_angles_adds_coordinates(self):
        """Test that add_angles adds latitude coordinate."""
        result = add_angles(self.ds, self.mtd)
        self.assertIn("latitude_3m_angles", result.coords)
        self.assertIn("longitude_3m_angles", result.coords)

    def test_add_angles_solar_zenith_conversion(self):
        """Test that solar_elevation is converted to solar_zenith_angle (90 - elevation)."""
        result = add_angles(self.ds, self.mtd)
        # sun_elevation = 45, so solar_zenith = 90 - 45 = 45
        expected_zenith = 90 - self.mtd.product_metadata["product_properties"]["sun_elevation"]
        actual_zenith = float(result["solar_zenith_angle"].values.flat[0])
        self.assertAlmostEqual(actual_zenith, expected_zenith, places=5)

    def test_add_angles_preserves_existing_data(self):
        """Test that add_angles preserves existing data variables."""
        result = add_angles(self.ds, self.mtd)
        self.assertIn("B1", result)
        np.testing.assert_array_equal(result["B1"].values, self.ds["B1"].values)


if __name__ == "__main__":
    unittest.main()
