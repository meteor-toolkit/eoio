"""eoio.readers.modis.aux_vars.tests.test_angles - tests for eoio.readers.modis.aux_vars.angles"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import xarray as xr
from eoio.readers.modis.aux_vars.angles import add_angles, ANGLE_DICT


class TestAddAngles(unittest.TestCase):
    def setUp(self) -> None:
        # Create mock metadata extractor
        self.mtd = MagicMock()

        # Create a minimal dataset
        self.ds = xr.Dataset()

        # Create aux dataset with angle variables
        self.aux_ds = xr.Dataset(
            {
                "SolarZenith": xr.DataArray(
                    np.random.rand(10, 10),
                    dims=["nscans*10", "mframes"],
                    coords={"nscans*10": np.arange(10), "mframes": np.arange(10)},
                ),
                "SolarAzimuth": xr.DataArray(
                    np.random.rand(10, 10),
                    dims=["nscans*10", "mframes"],
                    coords={"nscans*10": np.arange(10), "mframes": np.arange(10)},
                ),
                "SensorZenith": xr.DataArray(
                    np.random.rand(10, 10),
                    dims=["nscans*10", "mframes"],
                    coords={"nscans*10": np.arange(10), "mframes": np.arange(10)},
                ),
                "SensorAzimuth": xr.DataArray(
                    np.random.rand(10, 10),
                    dims=["nscans*10", "mframes"],
                    coords={"nscans*10": np.arange(10), "mframes": np.arange(10)},
                ),
            }
        )

    def test_add_angles_adds_solar_zenith(self):
        """Test that add_angles adds solar_zenith_angle variable."""
        result = add_angles(
            ds=self.ds,
            aux_ds=self.aux_ds,
            angle_names=["solar_zenith_angle"],
            mtd=self.mtd,
        )

        self.assertIn("solar_zenith_angle", result.data_vars)
        self.assertEqual(result["solar_zenith_angle"].shape, (10, 10))

    def test_add_angles_adds_multiple_angles(self):
        """Test that add_angles adds multiple angle variables."""
        angle_names = [
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
        ]

        result = add_angles(
            ds=self.ds,
            aux_ds=self.aux_ds,
            angle_names=angle_names,
            mtd=self.mtd,
        )

        for angle_name in angle_names:
            self.assertIn(angle_name, result.data_vars)

    def test_add_angles_renames_dimensions(self):
        """Test that add_angles renames dimensions correctly."""
        result = add_angles(
            ds=self.ds,
            aux_ds=self.aux_ds,
            angle_names=["solar_zenith_angle"],
            mtd=self.mtd,
        )

        # Check that dimensions are renamed
        dims = result["solar_zenith_angle"].dims
        self.assertIn("y_grid_1000m", dims)
        self.assertIn("x_grid_1000m", dims)
        self.assertNotIn("nscans*10", dims)
        self.assertNotIn("mframes", dims)

    def test_add_angles_raises_on_unknown_angle(self):
        """Test that add_angles raises ValueError for unknown angle names."""
        with self.assertRaises(ValueError) as ctx:
            add_angles(
                ds=self.ds,
                aux_ds=self.aux_ds,
                angle_names=["unknown_angle"],
                mtd=self.mtd,
            )

        self.assertIn("Unknown angle", str(ctx.exception))

    def test_add_angles_raises_when_variable_missing_in_aux_ds(self):
        """Test that add_angles raises ValueError when aux variable is missing."""
        aux_ds_incomplete = xr.Dataset()  # Missing angle variables

        with self.assertRaises(ValueError) as ctx:
            add_angles(
                ds=self.ds,
                aux_ds=aux_ds_incomplete,
                angle_names=["solar_zenith_angle"],
                mtd=self.mtd,
            )

        self.assertIn("not found in aux dataset", str(ctx.exception))

    def test_add_angles_returns_dataset_with_angles_attached(self):
        """Test that the returned dataset contains the input dataset data plus angles."""
        self.ds["some_band"] = xr.DataArray(
            np.random.rand(5, 5),
            dims=["y", "x"],
        )

        result = add_angles(
            ds=self.ds,
            aux_ds=self.aux_ds,
            angle_names=["solar_zenith_angle"],
            mtd=self.mtd,
        )

        # Original data should still be present
        self.assertIn("some_band", result.data_vars)
        # New angle data should be added
        self.assertIn("solar_zenith_angle", result.data_vars)

    def test_angle_dict_mapping(self):
        """Test that ANGLE_DICT contains expected mappings."""
        expected_mappings = {
            "solar_zenith_angle": "SolarZenith",
            "solar_azimuth_angle": "SolarAzimuth",
            "viewing_zenith_angle": "SensorZenith",
            "viewing_azimuth_angle": "SensorAzimuth",
        }

        for key, expected_val in expected_mappings.items():
            self.assertIn(key, ANGLE_DICT)
            self.assertEqual(ANGLE_DICT[key], expected_val)


if __name__ == "__main__":
    unittest.main()
