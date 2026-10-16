"""eoio.readers.gistda_theos.tests.test_aux_data - unit tests for eoio.readers.gistda_theos.aux_data"""

import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np
import xarray as xr

from eoio.readers.gistda_theos.aux_data import add_angles, add_aux, read_aux


class TestReadAux(unittest.TestCase):
    """Unit tests for read_aux function."""

    def setUp(self):
        """Set up test fixtures."""
        self.ds = xr.Dataset({"B1": xr.DataArray(np.zeros((2, 2)))})
        self.mtd = Mock()

    def test_read_aux_raises_when_aux_is_none(self):
        """Test that read_aux raises ValueError when aux is None."""
        with self.assertRaises(ValueError) as ctx:
            read_aux(self.ds, aux=None, mtd=self.mtd)
        self.assertIn("No auxiliary data files", str(ctx.exception))

    def test_read_aux_warns_for_masks(self):
        """Test that read_aux warns when masks are requested (not implemented) but still returns xr.Dataset."""
        aux = ["mask"]

        with self.assertWarns(UserWarning):
            result = read_aux(self.ds, aux=aux, mtd=self.mtd)

        self.assertIsInstance(result, xr.Dataset)

    def test_read_aux_calls_add_angles_and_warns_only_for_mask(self):
        """observation_geometry is fully implemented (add_angles) -- only the
        still-unimplemented "mask" should warn, not a blanket "not implemented"
        that would misrepresent the successful observation_geometry read too."""
        aux = ["observation_geometry", "mask"]

        with patch("eoio.readers.gistda_theos.aux_data.add_angles") as mock_add_angles:
            mock_add_angles.return_value = self.ds
            # capture all warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = read_aux(self.ds, aux=aux, mtd=self.mtd)

            mock_add_angles.assert_called_once_with(self.ds, self.mtd)
            self.assertIsInstance(result, xr.Dataset)
            messages = [str(x.message) for x in w]
            self.assertTrue(any("Mask data is not yet implemented" in m for m in messages))
            self.assertFalse(any("Auxiliary data" in m and "not yet implemented" in m for m in messages))

    def test_read_aux_does_not_warn_when_only_observation_geometry_requested(self):
        """The one fully-implemented aux name alone should produce no warnings at all."""
        with patch("eoio.readers.gistda_theos.aux_data.add_angles") as mock_add_angles:
            mock_add_angles.return_value = self.ds
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                read_aux(self.ds, aux=["observation_geometry"], mtd=self.mtd)
            self.assertEqual(w, [])

    def test_read_aux_warns_for_genuinely_unimplemented_name(self):
        """An aux name that isn't observation_geometry or mask still gets the
        not-yet-implemented warning, naming exactly what wasn't handled."""
        with self.assertWarns(UserWarning) as ctx:
            read_aux(self.ds, aux=["some_future_aux"], mtd=self.mtd)
        self.assertIn("some_future_aux", str(ctx.warning))


class TestAddAux(unittest.TestCase):
    """Unit tests for add_aux function. Currently a placeholder method."""

    def test_add_aux_returns_unchanged_dataset(self):
        """Test that add_aux returns dataset unchanged (placeholder implementation)."""
        ds = xr.Dataset({"B1": xr.DataArray(np.zeros((2, 2)))})
        layout = Mock()
        aux_names = ["aux1"]

        result = add_aux(ds=ds, layout=layout, aux_names=aux_names)

        self.assertIs(result, ds)
        self.assertIn("B1", result.data_vars)

    def test_add_aux_handles_empty_dataset(self):
        """Test that add_aux handles empty dataset (placeholder implementation)."""
        ds = xr.Dataset()
        layout = Mock()
        aux_names = []

        result = add_aux(ds=ds, layout=layout, aux_names=aux_names)

        self.assertIs(result, ds)
        self.assertEqual(len(result.data_vars), 0)


class TestAddAngles(unittest.TestCase):
    """Unit tests for add_angles function."""

    def get_metadata_observation_geometry_values(self):
        return {
            "observation_geometry": {
                "VIEWING_ANGLE_ALONG_TRACK".lower(): "0.555224",
                "VIEWING_ANGLE_ACROSS_TRACK".lower(): "-9.370350",
                "viewing_zenith_angle": "10.634494",
                "viewing_azimuth_angle": "100.364357",
                "SOLAR_AZIMUTH_angle".lower(): "84.182195",
                "solar_zenith_angle": str(90 - float("52.628636")),
            },
        }

    def _make_dataset_with_coords(self):
        """Create a dataset with required coordinates for angle calculations."""
        lon = np.array([[100.0, 100.5, 101.0], [100.0, 100.5, 101.0], [100.0, 100.5, 101.0]])
        lat = np.array([[50.0, 50.0, 50.0], [50.5, 50.5, 50.5], [51.0, 51.0, 51.0]])
        ds = xr.Dataset(
            {"B1": xr.DataArray(np.zeros((3, 3)), dims=("y_15m", "x_15m"))},
            coords={
                "longitude_15m": (("y_15m", "x_15m"), lon),
                "latitude_15m": (("y_15m", "x_15m"), lat),
                "x_15m": np.arange(3),
                "y_15m": np.arange(3),
            },
        )
        return ds

    def test_add_angles_returns_unchanged_when_no_geom_list(self):
        """Test that add_angles returns dataset unchanged when no metadata found."""
        ds = xr.Dataset({"B1": xr.DataArray(np.zeros((2, 2)))})
        mtd = Mock()
        mtd.get_angle_metadata.return_value = {
            "solar_azimuth_angle": {"units": "degrees"},
            "solar_zenith_angle": {"units": "degrees"},
            "viewing_azimuth_angle": {"units": "degrees"},
            "viewing_zenith_angle": {"units": "degrees"},
        }
        mtd.product_metadata = {}

        result = add_angles(ds=ds, mtd=mtd)

        self.assertIs(result, ds)
        self.assertIn("B1", result.data_vars)
        self.assertNotIn("solar_azimuth_angle", result.data_vars)

    def test_add_angles_reads_from_product_metadata(self):
        """Test that add_angles reads from mtd.product_metadata."""
        ds = self._make_dataset_with_coords()

        # Set up product metadata with located_geometric_values
        ds.attrs["product_metadata"] = self.get_metadata_observation_geometry_values()

        mtd = Mock()

        mtd.get_angle_metadata.return_value = {
            "solar_azimuth_angle": {"units": "degrees"},
            "solar_zenith_angle": {"units": "degrees"},
            "viewing_azimuth_angle": {"units": "degrees"},
            "viewing_zenith_angle": {"units": "degrees"},
        }
        result = add_angles(ds=ds, mtd=mtd)

        # Check that angle variables were added
        self.assertIn("solar_azimuth_angle", result.data_vars)
        self.assertIn("solar_zenith_angle", result.data_vars)
        self.assertIn("viewing_azimuth_angle", result.data_vars)
        self.assertIn("viewing_zenith_angle", result.data_vars)

    def test_add_angles_variables_carry_brdf_names_and_grid_id(self):
        """brdf_correction needs viewing_{zenith,azimuth}_angle; to_datatree needs the 15m
        geometry_id to put these scalars in the 15m node rather than the tree root."""
        ds = self._make_dataset_with_coords()
        ds.attrs["product_metadata"] = self.get_metadata_observation_geometry_values()

        result = add_angles(ds=ds, mtd=Mock())

        for name in ("viewing_zenith_angle", "viewing_azimuth_angle", "solar_zenith_angle", "solar_azimuth_angle"):
            self.assertEqual(result[name].ndim, 0)
            self.assertEqual(result[name].attrs["product_metadata"]["geometry_id"], "15m")
            self.assertEqual(result[name].attrs["units"], "degrees")
        self.assertEqual(result["viewing_zenith_angle"].attrs["standard_name"], "sensor_zenith_angle")
        self.assertNotIn("sensor_zenith_angle", result.data_vars)

    def test_add_angles_deep_search_list_scene_source(self):
        """Test deep-search path when Scene_Source is a list (exercise deep-search branch)."""
        ds = self._make_dataset_with_coords()

        # Scene_Source is a list where the second item contains SUN_AZIMUTH/SUN_ELEVATION
        mtd = Mock()
        mtd.product_metadata = {
            "Scene_Source": [
                {"OTHER": "ignore"},
                {"SUN_AZIMUTH": "84.182195", "SUN_ELEVATION": "52.628636"},
            ]
        }

        result = add_angles(ds=ds, mtd=mtd)

        # Check that angle variables were added and values parsed
        self.assertIn("solar_azimuth_angle", result.data_vars)
        self.assertIn("solar_zenith_angle", result.data_vars)
        self.assertAlmostEqual(float(result["solar_azimuth_angle"].values), 84.182195, places=6)
        # solar_zenith = 90 - SUN_ELEVATION
        self.assertAlmostEqual(float(result["solar_zenith_angle"].values), 90.0 - 52.628636, places=6)

    def test_add_angles_skips_a_missing_value_instead_of_crashing(self):
        """A single None angle value (e.g. solar_zenith_angle when SUN_ELEVATION is
        absent upstream -- see metadata._solar_zenith_from_elevation) is skipped,
        not a crash -- the other, present angles are still added."""
        ds = self._make_dataset_with_coords()
        values = self.get_metadata_observation_geometry_values()
        values["observation_geometry"]["solar_zenith_angle"] = None

        mtd = Mock()
        ds.attrs["product_metadata"] = values

        result = add_angles(ds=ds, mtd=mtd)

        self.assertNotIn("solar_zenith_angle", result.data_vars)
        self.assertIn("solar_azimuth_angle", result.data_vars)
        self.assertIn("viewing_zenith_angle", result.data_vars)

    def test_add_angles_handles_invalid_numeric_values(self):
        ds = self._make_dataset_with_coords()

        # Provide non-numeric sun elevation to exercise error handling
        mtd = Mock()
        mtd.get_angle_metadata.return_value = {
            "solar_azimuth_angle": {"units": "degrees"},
            "solar_zenith_angle": {"units": "degrees"},
            "viewing_azimuth_angle": {"units": "degrees"},
            "viewing_zenith_angle": {"units": "degrees"},
        }
        mtd.product_metadata = {"Scene_Source": {"SUN_AZIMUTH": "nan", "SUN_ELEVATION": "not_a_number"}}

        # Should not raise; any parsing errors should be handled gracefully
        result = add_angles(ds=ds, mtd=mtd)
        self.assertIsInstance(result, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
