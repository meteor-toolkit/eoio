import math
import unittest
from unittest.mock import patch
import xarray as xr
import numpy as np


from eoio.processors.units.drivers.s2msi1c import S2MSI1CUnitDriver


def make_ds(with_b01: bool = True, solar_irrad: float | None = 3.3) -> xr.Dataset:
    ds = xr.Dataset()
    if with_b01:
        da = xr.DataArray(
            np.array(
                [[1.0, 2.0], [3.0, 4.0]],
            ),
            dims=("y_60m", "x_60m"),
            attrs={
                "product_metadata": {"solar_irradiance": solar_irrad},
                "measurand": "reflectance",
            },
        )
        ds["B01"] = da
    ds.attrs["collection_name"] = "s2msi1c"
    return ds


class TestS2MSI1CUnitDriver(unittest.TestCase):
    def test_matches_default(self):
        driver = S2MSI1CUnitDriver()
        ds = make_ds()
        ds.attrs["collection_name"] = "S2MSI1C"
        self.assertTrue(driver.matches(ds, {}))

    def test_matches_nonmatching_collection_name(self):
        driver = S2MSI1CUnitDriver()
        ds = make_ds()
        ds.attrs["collection_name"] = "unrelated"
        self.assertFalse(driver.matches(ds, {}))

    def test_supported_returns_expected_pair(self):
        driver = S2MSI1CUnitDriver()
        self.assertEqual(driver.supported(), {("reflectance", "radiance")})

    @patch("eoio.processors.units.drivers.s2msi1c.S2MSI1CUnitDriver.rfl_to_rad")
    @patch("eoio.processors.units.drivers.s2msi1c.S2MSI1CUnitDriver.interp_var_s2")
    def test_convert_radiance_returns_converted_dataset(self, interp_var_s2, rfl_to_rad):
        driver = S2MSI1CUnitDriver()
        ds = make_ds(solar_irrad=2)
        ds.attrs["product_metadata"] = {"reflectance_conversion_u": 2}

        # Mock interp_var_s2 output
        interp_var_s2.return_value = (np.ndarray((1, 1)), np.ndarray((1, 1)))

        # Mock rfl_to_rad output
        rfl_to_rad.return_value = ds["B01"] * 2

        out = driver.convert(ds.copy(), to="radiance", var_names=["B01"], context={})

        # check data is converted
        xr.testing.assert_equal(out, ds * 2)

        # check attrs are updates to radiance
        self.assertIn("radiance", out["B01"].attrs["standard_name"])
        self.assertIn("radiance", out["B01"].attrs["long_name"])
        self.assertIn("radiance", out["B01"].attrs["measurand"])

    def test_convert_missing_variable_raises_keyerror(self):
        driver = S2MSI1CUnitDriver()
        ds = make_ds(with_b01=False)
        with self.assertRaises(KeyError):
            driver.convert(ds, to="radiance", var_names=["B01"], context={})

    def test_convert_unsupported_target_raises_valueerror(self):
        driver = S2MSI1CUnitDriver()
        ds = make_ds()
        with self.assertRaises(ValueError):
            driver.convert(ds, to="reflectance", var_names=["B01"], context={})

    def test_convert_missing_solar_irrad_returns_dataset(self):
        driver = S2MSI1CUnitDriver()
        ds = make_ds(solar_irrad=None)
        out = driver.convert(ds.copy(), to="radiance", var_names=["B01"], context={})
        self.assertIn("B01", out)
        self.assertIsNone(out["B01"].attrs["product_metadata"]["solar_irradiance"])

    def test_rfl_to_rad_scalar(self):
        rfl = 0.1
        sza = math.radians(30.0)
        e_sol = 1.0
        u = 1.0
        out = S2MSI1CUnitDriver.rfl_to_rad(rfl, sza, e_sol, u)
        expected = (rfl * math.cos(sza) * e_sol * u) / math.pi
        self.assertAlmostEqual(out, expected)

    def test_rfl_to_rad_array(self):
        rfl = np.array([0.1, 0.2], dtype=float)
        sza = np.array([math.radians(30.0), math.radians(60.0)], dtype=float)
        e_sol = np.array([1.0, 2.0], dtype=float)
        u = 1.0
        out = S2MSI1CUnitDriver.rfl_to_rad(rfl, sza, e_sol, u)
        expected = (rfl * np.cos(sza) * e_sol * u) / math.pi
        np.testing.assert_allclose(out, expected)

    def test_interp_var_s2_returns_same_values_when_target_grid_matches_source(self):
        ds = xr.Dataset()
        ds["solar_zenith_angle"] = xr.DataArray(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
            dims=("y_60m", "x_60m"),
        )
        ds["x_60m"] = xr.DataArray(np.array([0.0, 1.0], dtype=float), dims=("x_60m",))
        ds["y_60m"] = xr.DataArray(np.array([0.0, 1.0], dtype=float), dims=("y_60m",))

        out = S2MSI1CUnitDriver.interp_var_s2(ds, 60, var="solar_zenith_angle")
        np.testing.assert_allclose(out, ds["solar_zenith_angle"].data)

    def test_interp_var_s2_missing_var_raises_keyerror(self):
        # test raises when solar_zenith_angle missing from dataset
        ds = xr.Dataset()
        ds["x_60m"] = xr.DataArray(np.array([0.0, 1.0], dtype=float), dims=("x_60m",))
        ds["y_60m"] = xr.DataArray(np.array([0.0, 1.0], dtype=float), dims=("y_60m",))

        with self.assertRaises(KeyError):
            S2MSI1CUnitDriver.interp_var_s2(ds, 60, var="solar_zenith_angle")

    def test_interp_var_s2_missing_target_resolution_raises_keyerror(self):
        # test raises when 20 m dim missing from dataset
        ds = xr.Dataset()
        ds["solar_zenith_angle"] = xr.DataArray(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
            dims=("y", "x"),
        )
        ds["x_60m"] = xr.DataArray(np.array([0.0, 1.0], dtype=float), dims=("x_60m",))
        ds["y_60m"] = xr.DataArray(np.array([0.0, 1.0], dtype=float), dims=("y_60m",))

        with self.assertRaises(KeyError):
            S2MSI1CUnitDriver.interp_var_s2(ds, 20, var="solar_zenith_angle")


if __name__ == "__main__":
    unittest.main()
