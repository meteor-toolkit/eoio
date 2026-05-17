"""eoio.processors.units.drivers.tests.test_l89 - tests for eoio.processors.units.drivers.l89"""

import unittest
import numpy as np
import xarray as xr
from eoio.processors.units.drivers.l89 import (
    LANDSATC2L1UnitDriver,
    LANDSATC2L2UnitDriver,
)


def make_l1_ds() -> xr.Dataset:
    # Create a sample L1 dataset for testing
    ds = xr.Dataset(
        data_vars={
            "B1": xr.DataArray(
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                dims=["y", "x"],
                attrs={
                    "reflectance_mult": 2.0e-5,
                    "reflectance_add": -0.1,
                    "radiance_mult": 1.0,
                    "radiance_add": 0.0,
                    "measurand": "reflectance",
                    "units": "1",
                },
            ),
            "B2": xr.DataArray(
                np.array([[0.15, 0.25], [0.35, 0.45]], dtype=np.float32),
                dims=["y", "x"],
                attrs={
                    "reflectance_mult": 2.0e-5,
                    "reflectance_add": -0.1,
                    "radiance_mult": 1.5,
                    "radiance_add": 0.5,
                    "measurand": "reflectance",
                    "units": "1",
                },
            ),
            "B10": xr.DataArray(
                np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32),
                dims=["y", "x"],
                attrs={
                    "radiance_mult": 1.0,
                    "radiance_add": 0.0,
                    "measurand": "radiance",
                    "units": "W/(m²*sr*μm)",
                    "product_metadata": {
                        "k1": 700,
                        "k2": 1300,
                    },
                },
            ),
            "B11": xr.DataArray(
                np.array([[55.0, 65.0], [75.0, 85.0]], dtype=np.float32),
                dims=["y", "x"],
                attrs={
                    "radiance_mult": 1.0,
                    "radiance_add": 0.0,
                    "measurand": "radiance",
                    "units": "W/(m²*sr*μm)",
                    "product_metadata": {
                        "k1": 480,
                        "k2": 1200,
                    },
                },
            ),
        },
        attrs={
            "collection_name": "LANDSAT_C2_L1",
            "product_metadata": {
                "sun_elevation": 45.0,
            },
        },
    )
    return ds


def make_l2_ds() -> xr.Dataset:
    # Create a sample L2 dataset for testing
    ds = xr.Dataset(
        data_vars={
            "B1": xr.DataArray(
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                dims=["y", "x"],
                attrs={
                    "measurand": "reflectance",
                    "units": "1",
                },
            ),
            "B10": xr.DataArray(
                np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32),
                dims=["y", "x"],
                attrs={
                    "measurand": "radiance",
                    "units": "W/(m²*sr*μm)",
                    "product_metadata": {
                        "mult": 2,
                        "add": 1,
                    },
                },
            ),
            "B11": xr.DataArray(
                np.array([[55.0, 65.0], [75.0, 85.0]], dtype=np.float32),
                dims=["y", "x"],
                attrs={
                    "measurand": "radiance",
                    "units": "W/(m²*sr*μm)",
                    "product_metadata": {
                        "mult": 2,
                        "add": 1,
                    },
                },
            ),
        },
        attrs={
            "collection_name": "LANDSAT_C2_L2",
        },
    )
    return ds


class TestLANDSATC2L1UnitDriver(unittest.TestCase):
    def setUp(self):
        self.ds = make_l1_ds()
        self.driver = LANDSATC2L1UnitDriver()
        self.context = {}

    def test_matches_Matching(self):
        self.assertTrue(self.driver.matches(self.ds, self.context))

    def test_matches_NotMatching(self):
        ds_wrong = self.ds.copy()
        ds_wrong.attrs["collection_name"] = "SENTINEL_C2_L1"
        self.assertFalse(self.driver.matches(ds_wrong, self.context))

    def test_supported(self):
        expected = {
            "vnir": ("reflectance", "radiance"),
            "tir": ("radiance", "brightness_temperature"),
        }
        self.assertEqual(self.driver.supported(), expected)

    def test_convert_vnirRadiance_AllVars(self):
        to = {"vnir": "radiance"}
        result = self.driver.convert(self.ds, to, None, context=self.context)
        self.assertEqual(result["B1"].attrs["measurand"], "radiance")
        self.assertEqual(result["B1"].attrs["units"], "W/( m² * sr * μm)")
        self.assertEqual(result["B2"].attrs["measurand"], "radiance")
        self.assertEqual(result["B2"].attrs["units"], "W/( m² * sr * μm)")
        # B10 and B11 unchanged
        self.assertEqual(result["B10"].attrs["measurand"], "radiance")
        self.assertEqual(result["B11"].attrs["measurand"], "radiance")

    def test_convert_vnirRadiance_SelectedVars(self):
        to = {"vnir": "radiance"}
        result = self.driver.convert(self.ds, to, ["B1"], context=self.context)
        self.assertEqual(result["B1"].attrs["measurand"], "radiance")
        self.assertEqual(result["B2"].attrs["measurand"], "reflectance")  # unchanged

    def test_convert_tirBrightnessTemperature_AllVars(self):
        to = {"tir": "brightness_temperature"}
        result = self.driver.convert(self.ds, to, None, context=self.context)
        self.assertEqual(result["B10"].attrs["measurand"], "brightness_temperature")
        self.assertEqual(result["B10"].attrs["units"], "K")
        self.assertEqual(result["B11"].attrs["measurand"], "brightness_temperature")
        self.assertEqual(result["B11"].attrs["units"], "K")
        # B1 and B2 unchanged
        self.assertEqual(result["B1"].attrs["measurand"], "reflectance")

    def test_convert_Both(self):
        to = {"vnir": "radiance", "tir": "brightness_temperature"}
        result = self.driver.convert(self.ds, to, None, context=self.context)
        self.assertEqual(result["B1"].attrs["measurand"], "radiance")
        self.assertEqual(result["B10"].attrs["measurand"], "brightness_temperature")

    def test_convert_NoVars_Converts_All(self):
        to = {"vnir": "radiance"}
        result = self.driver.convert(self.ds, to, [], context=self.context)
        # should return all vnir vars converted to radiance
        self.assertEqual(result["B1"].attrs["measurand"], "radiance")
        self.assertEqual(result["B2"].attrs["measurand"], "radiance")
        # B10 and B11 unchanged
        self.assertEqual(result["B10"].attrs["measurand"], "radiance")
        self.assertEqual(result["B11"].attrs["measurand"], "radiance")

    def test_convert_AllVars(self):
        to = {"vnir": "radiance"}
        result = self.driver.convert(self.ds, to, "all", context=self.context)
        self.assertEqual(result["B1"].attrs["measurand"], "radiance")
        self.assertEqual(result["B2"].attrs["measurand"], "radiance")

    def test_convert_VarMissing(self):
        to = {"vnir": "radiance"}
        with self.assertRaises(KeyError):
            self.driver.convert(self.ds, to, ["B3"], context=self.context)

    def test_convert_InvalidvnirTo(self):
        to = {"vnir": "invalid"}
        with self.assertRaises(ValueError):
            self.driver.convert(self.ds, to, None, context=self.context)

    def test_convert_InvalidtirTo(self):
        to = {"tir": "invalid"}
        with self.assertRaises(ValueError):
            self.driver.convert(self.ds, to, None, context=self.context)


class TestLANDSATC2L2UnitDriver(unittest.TestCase):
    def setUp(self):
        self.ds = make_l2_ds()
        self.driver = LANDSATC2L2UnitDriver()
        self.context = {}

    def test_matches_Matching(self):
        self.assertTrue(self.driver.matches(self.ds, self.context))

    def test_matches_NotMatching(self):
        ds_wrong = self.ds.copy()
        ds_wrong.attrs["collection_name"] = "SENTINEL_C2_L2"
        self.assertFalse(self.driver.matches(ds_wrong, self.context))

    def test_supported(self):
        expected = {
            "vnir": ("reflectance", "radiance"),
            "tir": ("radiance", "brightness_temperature"),
        }
        self.assertEqual(self.driver.supported(), expected)

    def test_convert_tirBrightnessTemperature_AllVars(self):
        to = {"tir": "brightness_temperature"}
        result = self.driver.convert(self.ds, to, None, context=self.context)
        self.assertEqual(result["B10"].attrs["measurand"], "brightness_temperature")
        self.assertEqual(result["B10"].attrs["units"], "K")
        self.assertEqual(result["B11"].attrs["measurand"], "brightness_temperature")
        self.assertEqual(result["B11"].attrs["units"], "K")
        # B1 unchanged
        self.assertEqual(result["B1"].attrs["measurand"], "reflectance")

    def test_convert_InvalidTo(self):
        to = {"vnir": "radiance"}
        with self.assertRaises(ValueError):
            self.driver.convert(self.ds, to, None, context=self.context)


if __name__ == "__main__":
    unittest.main()
