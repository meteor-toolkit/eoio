"""Tests for eoio.processors.resample_wavelength.processor"""

import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from eoio.processors.resample_wavelength.processor import ResampleWavelength
from eoio.readers.flex.reader import FLEXL2Reader
from eoio.readers.flex.tests.synthetic import write_synthetic_flex

NAME = "flx_l2__flxsyn_20190930T102242_20190930T102254_20260312T190507_31TDG_100_036_01.nc"


def _linear_ds():
    wl = np.array([400.0, 500.0, 600.0])
    return xr.Dataset(
        {"a": (("wavelength_a", "y"), np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]))},
        coords={"wavelength_a": wl},
    )


class TestParams(unittest.TestCase):
    def test_target_required(self):
        with self.assertRaises(ValueError):
            ResampleWavelength(params={})

    def test_target_dict(self):
        p = ResampleWavelength(params={"target_wavelength": {"min": 400, "max": 410, "step": 5}})
        np.testing.assert_allclose(p.config.target_wavelength, [400, 405, 410])

    def test_bad_target_dict(self):
        with self.assertRaises(ValueError):
            ResampleWavelength(params={"target_wavelength": {"min": 400, "max": 410}})
        with self.assertRaises(ValueError):
            ResampleWavelength(params={"target_wavelength": {"min": 400, "max": 410, "step": 0}})

    def test_bad_on_missing(self):
        with self.assertRaises(ValueError):
            ResampleWavelength(params={"target_wavelength": [1], "on_missing": "x"})


class TestRun(unittest.TestCase):
    def test_linear_interpolation_and_naming(self):
        out = ResampleWavelength(params={"target_wavelength": [450.0, 550.0, 700.0]}).run(_linear_ds())
        np.testing.assert_allclose(out["a_resampled"].values[:2, 0], [0.5, 1.5])
        self.assertTrue(np.isnan(out["a_resampled"].values[2]).all())  # outside source range: no extrapolation
        self.assertEqual(out["a_resampled"].dims, ("wavelength", "y"))
        self.assertIn("a", out)  # original kept
        self.assertEqual(out["wavelength"].attrs["units"], "nm")
        self.assertEqual(out.attrs["eoio:processing_steps"][-1]["processor"], "resample_wavelength")

    def test_inplace_replaces(self):
        out = ResampleWavelength(params={"target_wavelength": [450.0], "inplace": True}).run(_linear_ds())
        self.assertEqual(out["a"].dims, ("wavelength", "y"))
        self.assertNotIn("a_resampled", out)
        self.assertNotIn("wavelength_a", out.coords)

    def test_drop_original(self):
        out = ResampleWavelength(params={"target_wavelength": [450.0], "drop_original": True}).run(_linear_ds())
        self.assertNotIn("a", out)
        self.assertIn("a_resampled", out)

    def test_nearest(self):
        out = ResampleWavelength(params={"target_wavelength": [490.0], "method": "nearest"}).run(_linear_ds())
        np.testing.assert_allclose(out["a_resampled"].values[0], [1.0, 20.0])

    def test_missing_var(self):
        with self.assertRaises(ValueError):
            ResampleWavelength(params={"target_wavelength": [450.0], "data_vars": ["nope"]}).run(_linear_ds())
        out = ResampleWavelength(
            params={"target_wavelength": [450.0], "data_vars": ["nope"], "on_missing": "skip"}
        ).run(_linear_ds())
        self.assertNotIn("a_resampled", out)

    def test_existing_dim_name_raises(self):
        ds = _linear_ds().assign_coords(wavelength=("wavelength", [1.0]))
        ds["b"] = ("wavelength", [1.0])
        with self.assertRaises(ValueError):
            ResampleWavelength(params={"target_wavelength": [450.0]}).run(ds)

    def test_not_a_dataset(self):
        with self.assertRaises(TypeError):
            ResampleWavelength(params={"target_wavelength": [450.0]}).run(_linear_ds()["a"])


class TestWithFlexReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(cls.tmp.name, NAME)
        write_synthetic_flex(path)
        cls.ds = FLEXL2Reader(
            path, vars_sel={"meas": ["olci_apparent_reflectance", "floris_apparent_reflectance"]}
        ).open()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_spectra_from_different_sensors_share_grid(self):
        out = ResampleWavelength(params={"target_wavelength": {"min": 700, "max": 780, "step": 10}}).run(self.ds)
        a, b = out["olci_apparent_reflectance_resampled"], out["floris_apparent_reflectance_resampled"]
        self.assertEqual(a.sizes["wavelength"], 9)
        self.assertEqual(a.dims, b.dims)
        self.assertEqual(a.dtype, np.float32)

    def test_uncertainty_follows_base_and_keeps_err_corr(self):
        out = ResampleWavelength(params={"target_wavelength": {"min": 700, "max": 780, "step": 10}}).run(self.ds)
        unc_name = "floris_apparent_reflectance_uncertainty_resampled"
        self.assertEqual(out["floris_apparent_reflectance_resampled"].attrs["unc_comps"], unc_name)
        attrs = out[unc_name].attrs
        self.assertEqual(attrs["err_corr_2_dim"], ["wavelength"])
        self.assertEqual(attrs["err_corr_2_form"], "systematic")


if __name__ == "__main__":
    unittest.main()
