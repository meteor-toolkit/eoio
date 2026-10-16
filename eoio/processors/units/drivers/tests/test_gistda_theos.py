import unittest
import numpy as np
import xarray as xr

from eoio.processors.units.drivers.gistda_theos import GISTDATHEOSUnitDriver
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
    ds.attrs["eoio:reader"] = "gistda_theos"

    if solar_zenith is not None:
        # optionally add variable or attrs for solar zenith
        if use_attrs_for_zenith:
            ds.attrs["observation_geometry"] = {"solar_zenith_angle": solar_zenith}
        else:
            ds["solar_zenith_angle"] = xr.DataArray(np.array([solar_zenith], dtype=float), dims=("x",))
    return ds


class TestGISTDATHEOS(unittest.TestCase):
    def test_registry_contains_gistda_theos_driver(self):
        self.assertIn("gistda_theos", UNIT_DRIVER_REGISTRY)
        self.assertIs(UNIT_DRIVER_REGISTRY["gistda_theos"], GISTDATHEOSUnitDriver)

    def test_register_duplicate_name_raises(self):
        with self.assertRaises(KeyError):

            @register_unit_driver("gistda_theos")
            class DummyDriver:
                pass

    def test_matches_true_and_false(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds()
        self.assertTrue(driver.matches(ds, {}))

        ds.attrs["eoio:reader"] = "other"
        self.assertFalse(driver.matches(ds, {}))

    def test_supported_returns_expected_set(self):
        driver = GISTDATHEOSUnitDriver()
        self.assertEqual(
            driver.supported(),
            {("radiance", "reflectance")},
        )

    def test_convert_no_vars_converts_every_measurement_variable(self):
        """None (the processor's default) means all measurement variables, as the docstring says -- not
        "leave the data as radiance" (which let a bare ``units.convert: {to: reflectance}`` do nothing)."""
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds(value=10.0, solar_zenith=0.0)
        ds["B2"] = ds["B1"].copy(deep=True)
        out = driver.convert(ds.copy(deep=True), to="reflectance", var_names=None, context={})
        for name in ("B1", "B2"):
            self.assertAlmostEqual(out[name].values[0], np.pi * 10.0 / 2.0)
            self.assertEqual(out[name].attrs["product_metadata"]["measurand"], "reflectance")

    def test_convert_empty_selection_converts_nothing(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds()
        out = driver.convert(ds.copy(deep=True), to="reflectance", var_names=[], context={})
        xr.testing.assert_equal(out, ds)

    def test_matches_a_dataset_with_no_reader_attr(self):
        """Every registered driver is asked about every dataset; a plain "no", not a KeyError."""
        self.assertFalse(GISTDATHEOSUnitDriver().matches(xr.Dataset(), {}))

    def test_missing_earth_sun_distance_warns(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds(value=10.0, solar_zenith=0.0)
        with self.assertWarnsRegex(UserWarning, "earth_sun_distance"):
            driver.convert(ds.copy(deep=True), to="reflectance", var_names=["B1"], context={})

    def test_convert_variable_not_in_dataset_raises_keyerror(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds()
        with self.assertRaises(KeyError):
            driver.convert(ds, to="reflectance", var_names=["B2"], context={})

    def test_convert_unsupported_to_raises_valueerror(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds()
        with self.assertRaises(ValueError):
            driver.convert(ds, to="foo", var_names=["B1"], context={})

    def test_reflectance_conversion_missing_metadata_raises(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds()
        del ds["B1"].attrs["product_metadata"]["band_gain"]
        with self.assertRaises(KeyError):
            driver.convert(ds, to="reflectance", var_names=["B1"], context={})

    def test_reflectance_conversion_with_variable_zenith(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds(value=10.0, solar_zenith=0.0)
        out = driver.convert(ds.copy(), to="reflectance", var_names=["B1"], context={})
        expected = (np.pi * (10.0 / 1.0 + 0.0)) / (2.0 * np.cos(np.deg2rad(0.0)))
        self.assertAlmostEqual(out["B1"].values[0], expected)
        self.assertEqual(out["B1"].attrs["product_metadata"]["measurand"], "reflectance")

    def test_reflectance_conversion_applies_earth_sun_distance(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds(value=10.0, solar_zenith=0.0)
        ds["B1"].attrs["product_metadata"]["earth_sun_distance"] = 1.02
        out = driver.convert(ds.copy(), to="reflectance", var_names=["B1"], context={})
        self.assertAlmostEqual(out["B1"].values[0], np.pi * 10.0 * 1.02**2 / 2.0)

    def test_reflectance_conversion_with_attr_zenith(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds(value=5.0, solar_zenith=30.0, use_attrs_for_zenith=True)
        out = driver.convert(ds.copy(), to="reflectance", var_names=["B1"], context={})
        zen = 30.0
        expected = (np.pi * (5.0 / 1.0)) / (2.0 * np.cos(np.deg2rad(zen)))
        self.assertAlmostEqual(out["B1"].values[0], expected)

    def test_reflectance_conversion_missing_irradiance_raises(self):
        driver = GISTDATHEOSUnitDriver()
        ds = make_base_ds(solar_zenith=10.0)
        ds["B1"].attrs["product_metadata"].pop("band_solar_irradiance")
        with self.assertRaises(KeyError):
            driver.convert(ds, to="reflectance", var_names=["B1"], context={})


if __name__ == "__main__":
    unittest.main()
