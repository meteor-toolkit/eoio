import unittest
import numpy as np
import xarray as xr

from eoio.processors.units.drivers.airbus_pleiades import AirbusPleiadesUnitDriver
from eoio.processors.units.drivers.registry import (
    UNIT_DRIVER_REGISTRY,
    register_unit_driver,
)


def make_base_ds(
    var_name: str = "B1",
    value: float = 10.0,
    long_name: str = "Radiance",
    solar_zenith: float | None = None,
    use_attrs_for_zenith: bool = False,
) -> xr.Dataset:
    """Construct a minimal dataset for driver tests."""
    da = xr.DataArray(
        np.array([value], dtype=float),
        dims=("x",),
        attrs={
            "product_metadata": {
                "band_gain": 1.0,
                "band_bias": 0.0,
                "band_solar_irradiance": 2.0,
                "long_name": long_name,
                "measurand": long_name.lower(),
            }
        },
    )
    ds = xr.Dataset({var_name: da})
    ds.attrs["eoio:reader"] = "airbus_pleiades"

    if solar_zenith is not None:
        # optionally add variable or attrs for solar zenith
        if use_attrs_for_zenith:
            ds.attrs["located_geometric_values"] = {"SUN_ELEVATION": {"#text": str(90.0 - solar_zenith)}}
        else:
            ds["solar_zenith_angle"] = xr.DataArray(np.array([solar_zenith], dtype=float), dims=("x",))
    return ds


class TestAirbusPleiades(unittest.TestCase):
    def test_registry_contains_airbus_driver(self):
        self.assertIn("airbus_pleiades", UNIT_DRIVER_REGISTRY)
        self.assertIs(UNIT_DRIVER_REGISTRY["airbus_pleiades"], AirbusPleiadesUnitDriver)

    def test_register_duplicate_name_raises(self):
        with self.assertRaises(KeyError):

            @register_unit_driver("airbus_pleiades")
            class DummyDriver:
                pass

    def test_matches_true_and_false(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds()
        self.assertTrue(driver.matches(ds, {}))

        ds.attrs["eoio:reader"] = "other"
        self.assertFalse(driver.matches(ds, {}))

    def test_supported_returns_expected_set(self):
        driver = AirbusPleiadesUnitDriver()
        self.assertEqual(
            driver.supported(),
            {("radiance", "DNs"), ("radiance", "reflectance")},
        )

    def test_convert_no_vars_returns_same_dataset(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds()
        out = driver.convert(ds.copy(), to="DNs", var_names=None, context={})
        xr.testing.assert_equal(out, ds)

    def test_convert_variable_not_in_dataset_raises_keyerror(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds()
        with self.assertRaises(KeyError):
            driver.convert(ds, to="DNs", var_names=["B2"], context={})

    def test_convert_unsupported_to_raises_valueerror(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds()
        with self.assertRaises(ValueError):
            driver.convert(ds, to="foo", var_names=["B1"], context={})

    def test_reflectance_conversion_missing_metadata_raises(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds()
        del ds["B1"].attrs["product_metadata"]["band_gain"]
        with self.assertRaises(KeyError):
            driver.convert(ds, to="reflectance", var_names=["B1"], context={})

    def test_reflectance_conversion_with_variable_zenith(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds(value=10.0, solar_zenith=0.0)
        out = driver.convert(ds.copy(), to="reflectance", var_names=["B1"], context={})
        expected = (np.pi * (10.0 / 1.0 + 0.0)) / (2.0 * np.cos(np.deg2rad(0.0)))
        self.assertAlmostEqual(out["B1"].values[0], expected)
        self.assertEqual(out["B1"].attrs["product_metadata"]["measurand"], "reflectance")

    def test_reflectance_conversion_with_attr_zenith(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds(value=5.0, solar_zenith=30.0, use_attrs_for_zenith=True)
        out = driver.convert(ds.copy(), to="reflectance", var_names=["B1"], context={})
        zen = 30.0
        expected = (np.pi * (5.0 / 1.0)) / (2.0 * np.cos(np.deg2rad(zen)))
        self.assertAlmostEqual(out["B1"].values[0], expected)

    def test_reflectance_conversion_missing_irradiance_raises(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds(solar_zenith=10.0)
        ds["B1"].attrs["product_metadata"].pop("band_solar_irradiance")
        with self.assertRaises(KeyError):
            driver.convert(ds, to="reflectance", var_names=["B1"], context={})

    # TODO tests for converting to DNs if keeping functionality

    def test_dns_conversion_computes_expected_values(self):
        driver = AirbusPleiadesUnitDriver()
        ds = make_base_ds(value=4.0)  # bias=0, gain=1
        out = driver.convert(ds.copy(), to="DNs", var_names=["B1"], context={})
        self.assertEqual(out["B1"].values[0], 4.0)
        self.assertEqual(out["B1"].attrs["product_metadata"]["measurand"], "DNs")


if __name__ == "__main__":
    unittest.main()
