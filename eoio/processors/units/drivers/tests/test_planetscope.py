import unittest
import numpy as np
import xarray as xr

from eoio.processors.units.drivers.planetscope import PlanetScopeUnitDriver


def make_ds(
    var_name: str = "B2",
    value: float = 10.0,
    long_name: str = "radiance",
    measurand: str = "radiance",
    rad_coeff: float = 2.0,
    refl_coeff: float | None = 1.5,
) -> xr.Dataset:
    da = xr.DataArray(
        np.array([value], dtype=float),
        dims=("x",),
        attrs={
            "product_metadata": {
                "radiometric_scale_factor": rad_coeff,
                "reflectance_coefficient": refl_coeff,
                "long_name": long_name,
                "measurand": measurand,
            }
        },
    )
    ds = xr.Dataset({var_name: da})
    ds.attrs["eoio:reader"] = "planetscope"
    return ds


class TestPlanetScope(unittest.TestCase):
    def test_matches_true_and_false(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds()
        self.assertTrue(driver.matches(ds, {}))
        ds.attrs["eoio:reader"] = "PLANETSCOPE"
        self.assertTrue(driver.matches(ds, {}))
        ds.attrs["eoio:reader"] = "other"
        self.assertFalse(driver.matches(ds, {}))

    def test_supported_pairs(self):
        driver = PlanetScopeUnitDriver()
        expected = {
            ("radiance", "reflectance"),
            ("radiance", "DNs"),
            ("reflectance", "DNs"),
        }
        self.assertEqual(driver.supported(), expected)

    def test_convert_no_vars_converts_all(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds(value=4.0, rad_coeff=2.0, refl_coeff=1.0, long_name="Radiance band")
        out = driver.convert(ds.copy(), to="reflectance", var_names=None, context={})
        self.assertAlmostEqual(out["B2"].values[0], 4.0 * (1.0 / 2.0))

    def test_convert_var_missing_raises_keyerror(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds()
        with self.assertRaises(KeyError):
            driver.convert(ds, to="reflectance", var_names=["B1"], context={})

    def test_convert_bad_target_raises(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds()
        with self.assertRaises(ValueError):
            driver.convert(ds, to="foo", var_names=["B2"], context={})

    def test_reflectance_missing_refl_coeff_raises(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds()
        ds["B2"].attrs["product_metadata"].pop("reflectance_coefficient")
        with self.assertRaises(KeyError):
            driver.convert(ds, to="reflectance", var_names=["B2"], context={})

    def test_reflectance_from_radiance(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds(value=4.0, rad_coeff=2.0, refl_coeff=1.0, long_name="Radiance band")
        out = driver.convert(ds.copy(), to="reflectance", var_names=["B2"], context={})
        self.assertAlmostEqual(out["B2"].values[0], 4.0 * (1.0 / 2.0))

    # TODO tests for converting to DNs if keeping functionality

    def test_dns_from_radiance(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds(
            value=9.0,
            rad_coeff=3.0,
            long_name="Radiance",
            measurand="Radiance",
        )
        out = driver.convert(ds.copy(), to="DNs", var_names=["B2"], context={})
        self.assertEqual(out["B2"].values[0], 3)
        self.assertEqual(out["B2"].attrs["product_metadata"]["measurand"], "DNs")

    def test_dns_from_reflectance(self):
        driver = PlanetScopeUnitDriver()
        ds = make_ds(
            value=5.0,
            refl_coeff=2.0,
            long_name="reflectance",
            measurand="reflectance",
        )
        out = driver.convert(ds.copy(), to="DNs", var_names=["B2"], context={})
        self.assertEqual(out["B2"].values[0], 2.5)
        self.assertEqual(out["B2"].attrs["product_metadata"]["measurand"], "DNs")


if __name__ == "__main__":
    unittest.main()
