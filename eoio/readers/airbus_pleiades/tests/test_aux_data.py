"""
eoio.readers.airbus_pleiades.tests.test_aux_data - unit tests for eoio.readers.airbus_pleiades.aux_data
"""

import unittest
from unittest.mock import Mock
import numpy as np
import xarray as xr
from eoio.readers.airbus_pleiades.aux_data import read_aux, add_aux, add_angles


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

    def get_metadata_located_geometric_values(self):
        return {
            "located_geometric_values": [
                {
                    "Solar_Incidences": {
                        "SUN_AZIMUTH": {"#text": "120.5"},
                        "SUN_ELEVATION": {"#text": "45.3"},
                    },
                    "Acquisition_Angles": {
                        "AZIMUTH_ANGLE": "95.2",
                        "VIEWING_ANGLE": {"#text": "30.1"},
                    },
                },
                {
                    "Solar_Incidences": {
                        "SUN_AZIMUTH": {"#text": "121.2"},
                        "SUN_ELEVATION": {"#text": "44.8"},
                    },
                    "Acquisition_Angles": {
                        "AZIMUTH_ANGLE": "95.5",
                        "VIEWING_ANGLE": {"#text": "30.4"},
                    },
                },
                {
                    "Solar_Incidences": {
                        "SUN_AZIMUTH": {"#text": "122.0"},
                        "SUN_ELEVATION": {"#text": "44.2"},
                    },
                    "Acquisition_Angles": {
                        "AZIMUTH_ANGLE": "96.0",
                        "VIEWING_ANGLE": {"#text": "30.7"},
                    },
                },
            ]
        }

    def _make_dataset_with_coords(self):
        """Create a dataset with required coordinates for angle calculations."""
        lon = np.array([[100.0, 100.5, 101.0], [100.0, 100.5, 101.0], [100.0, 100.5, 101.0]])
        lat = np.array([[50.0, 50.0, 50.0], [50.5, 50.5, 50.5], [51.0, 51.0, 51.0]])
        ds = xr.Dataset(
            {"B1": xr.DataArray(np.zeros((3, 3)), dims=("y_2m", "x_2m"))},
            coords={
                "longitude_2m": (("y_2m", "x_2m"), lon),
                "latitude_2m": (("y_2m", "x_2m"), lat),
                "x_2m": np.arange(3),
                "y_2m": np.arange(3),
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
            "sensor_azimuth_angle": {"units": "degrees"},
            "sensor_zenith_angle": {"units": "degrees"},
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
        ds.attrs["product_metadata"] = self.get_metadata_located_geometric_values()

        mtd = Mock()

        mtd.get_angle_metadata.return_value = {
            "solar_azimuth_angle": {"units": "degrees"},
            "solar_zenith_angle": {"units": "degrees"},
            "sensor_azimuth_angle": {"units": "degrees"},
            "sensor_zenith_angle": {"units": "degrees"},
        }
        result = add_angles(ds=ds, mtd=mtd)

        # Check that angle variables were added
        self.assertIn("solar_azimuth_angle", result.data_vars)
        self.assertIn("solar_zenith_angle", result.data_vars)
        self.assertIn("sensor_azimuth_angle", result.data_vars)
        self.assertIn("sensor_zenith_angle", result.data_vars)

    def test_add_angles_transforms_elevation_to_zenith(self):
        """Test that solar elevation is correctly transformed to zenith angle (90 - elevation)."""
        ds = self._make_dataset_with_coords()

        ds.attrs["product_metadata"] = self.get_metadata_located_geometric_values()

        mtd = Mock()
        mtd.get_angle_metadata.return_value = {
            "solar_azimuth_angle": {"units": "degrees"},
            "solar_zenith_angle": {"units": "degrees"},
            "sensor_azimuth_angle": {"units": "degrees"},
            "sensor_zenith_angle": {"units": "degrees"},
        }
        result = add_angles(ds=ds, mtd=mtd)

        # Check that zenith angle was transformed (90 - elevation)
        # SUN_ELEVATION values: 45.3, 44.8, 44.2 -> zenith: 44.7, 45.2, 45.8
        expected_zenith = [90.0 - 45.3, 90.0 - 44.8, 90.0 - 44.2]
        np.testing.assert_array_almost_equal(result["solar_zenith_angle"].values.flatten(), expected_zenith, decimal=1)

    def test_add_angles_adds_angle_attributes(self):
        """Test that angle values are also added as dataset attributes."""
        ds = self._make_dataset_with_coords()

        ds.attrs["product_metadata"] = self.get_metadata_located_geometric_values()

        mtd = Mock()
        mtd.get_angle_metadata.return_value = {
            "solar_azimuth_angle": {"units": "degrees"},
            "solar_zenith_angle": {"units": "degrees"},
            "sensor_azimuth_angle": {"units": "degrees"},
            "sensor_zenith_angle": {"units": "degrees"},
        }

        result = add_angles(ds=ds, mtd=mtd)

        # Check that angle attributes exist
        self.assertIn("solar_azimuth_angle", result.attrs)
        self.assertIsInstance(result.attrs["solar_azimuth_angle"], list)
        self.assertEqual(len(result.attrs["solar_azimuth_angle"]), 3)

    def test_add_angles_creates_angle_coordinates(self):
        """Test that add_angles creates latitude_angles and longitude_angles coordinates."""
        ds = self._make_dataset_with_coords()

        ds.attrs["product_metadata"] = self.get_metadata_located_geometric_values()

        mtd = Mock()
        mtd.get_angle_metadata.return_value = {
            "solar_azimuth_angle": {"units": "degrees"},
            "solar_zenith_angle": {"units": "degrees"},
            "sensor_azimuth_angle": {"units": "degrees"},
            "sensor_zenith_angle": {"units": "degrees"},
        }

        result = add_angles(ds=ds, mtd=mtd)

        # Check that angle coordinates were added
        self.assertIn("latitude_2m_angles", result.coords)
        self.assertIn("longitude_2m_angles", result.coords)

        # Check shapes
        self.assertEqual(result["solar_azimuth_angle"].shape, (3, 1))
        self.assertEqual(result.coords["latitude_2m_angles"].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
