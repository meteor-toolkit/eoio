"""eoio.readers.constellr_skybee.tests.test_aux_data - unit tests for eoio.readers.constellr_skybee.aux_data"""

import unittest
from unittest.mock import MagicMock

import xarray as xr

from eoio.readers.constellr_skybee.aux_data import add_angles, read_aux


class TestSkybeeAuxData(unittest.TestCase):
    def test_read_aux_raises_for_none(self):
        ds = xr.Dataset()
        with self.assertRaises(ValueError):
            read_aux(ds, None, MagicMock())

    def test_read_aux_adds_angles(self):
        ds = xr.Dataset()
        mtd = MagicMock()
        mtd.product_metadata = {"observation_geometry": {"solar_zenith_angle": 30.0, "solar_azimuth_angle": 45.0}}

        out = read_aux(ds, ["observation_geometry"], mtd)

        self.assertIn("solar_zenith_angle", out)
        self.assertEqual(float(out["solar_zenith_angle"]), 30.0)

    def test_add_angles_names_and_places_angles(self):
        mtd = MagicMock()
        mtd.product_metadata = {"observation_geometry": {"viewing_zenith_angle": 2.8, "solar_zenith_angle": 48.0}}

        out = add_angles(xr.Dataset(), mtd)

        self.assertEqual(out["viewing_zenith_angle"].attrs["standard_name"], "sensor_zenith_angle")
        self.assertEqual(out["viewing_zenith_angle"].attrs["units"], "degrees")
        self.assertEqual(out["solar_zenith_angle"].attrs["product_metadata"], {"geometry_id": "10m"})
        self.assertNotIn("measurand", out["solar_zenith_angle"].attrs)

    def test_add_angles_adds_a_copy_to_every_grid_the_bands_are_on(self):
        """eoalign's brdf_correction reads the angles from the node holding the reflectance, so each
        resolution node needs them -- not just the 10 m one."""
        ds = xr.Dataset(
            {
                "B2": (("y_10m", "x_10m"), [[1.0]]),
                "B5": (("y_20m", "x_20m"), [[1.0]]),
                "B1": (("y_60m", "x_60m"), [[1.0]]),
            }
        )
        mtd = MagicMock()
        mtd.product_metadata = {"observation_geometry": {"solar_zenith_angle": 48.0, "viewing_azimuth_angle": 100.0}}

        out = add_angles(ds, mtd)

        for res in ("10m", "20m", "60m"):
            self.assertEqual(float(out[f"solar_zenith_angle_{res}"]), 48.0)
            self.assertEqual(out[f"viewing_azimuth_angle_{res}"].attrs["product_metadata"], {"geometry_id": res})
        self.assertNotIn("solar_zenith_angle", out)

    def test_read_aux_warns_for_unimplemented_request(self):
        ds = xr.Dataset()
        mtd = MagicMock()
        mtd.product_metadata = {}

        with self.assertWarns(UserWarning):
            read_aux(ds, ["something"], mtd)

    def test_add_angles_returns_dataset_when_missing(self):
        ds = xr.Dataset()
        mtd = MagicMock()
        mtd.product_metadata = {}

        out = add_angles(ds, mtd)
        self.assertIs(out, ds)

    def test_add_angles_assigns_values(self):
        ds = xr.Dataset()
        mtd = MagicMock()
        mtd.product_metadata = {"observation_geometry": {"viewing_zenith_angle": 20.0}}

        out = add_angles(ds, mtd)
        self.assertIn("viewing_zenith_angle", out)
        self.assertEqual(float(out["viewing_zenith_angle"]), 20.0)


if __name__ == "__main__":
    unittest.main()
