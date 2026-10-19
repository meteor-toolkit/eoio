"""Tests for eoio.readers.flex.data_io"""

import os
import tempfile
import unittest

import numpy as np

from eoio.readers.flex import data_io
from eoio.readers.flex.subset import FLEXSubset
from eoio.readers.flex.tests.synthetic import N, WAVELENGTHS, write_synthetic_flex


class TestWavelengths(unittest.TestCase):
    def test_collapse_single_populated_row(self):
        values = np.full((3, 4), np.nan)
        values[0] = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(data_io._collapse_wavelengths(values, "w"), [1, 2, 3, 4])

    def test_collapse_warns_when_rows_differ(self):
        values = np.array([[1.0, 2.0], [1.5, 2.5], [np.nan, np.nan]])
        with self.assertWarns(UserWarning):
            out = data_io._collapse_wavelengths(values, "w")
        np.testing.assert_allclose(out, [1.25, 2.25])

    def test_collapse_all_nan_raises(self):
        with self.assertRaises(ValueError):
            data_io._collapse_wavelengths(np.full((2, 3), np.nan), "w")


class TestErrCorr(unittest.TestCase):
    def test_spectral_gets_systematic_wavelength_component(self):
        ec = data_io.err_corr_for(["wavelength_sif", "y", "x"])
        self.assertEqual(ec[0], {"dim": ["y", "x"], "form": "random", "params": [], "units": []})
        self.assertEqual(ec[1]["dim"], ["wavelength_sif"])
        self.assertEqual(ec[1]["form"], "systematic")

    def test_non_spectral_only_random(self):
        self.assertEqual(len(data_io.err_corr_for(["y", "x"])), 1)


class TestReadDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(
            cls.tmp.name, "flx_l2__flxsyn_20190930T102242_20190930T102254_20260312T190507_31TDG_100_036_01.nc"
        )
        write_synthetic_flex(cls.path)
        cls.grid = data_io.read_grid(cls.path)
        cls.wavelengths = data_io.read_wavelengths(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_read_grid(self):
        self.assertEqual((self.grid.nx, self.grid.ny), (N, N))
        self.assertEqual(self.grid.epsg, 32631)

    def test_read_wavelengths_floris_collapsed(self):
        w = self.wavelengths["wavelength_floris"]
        self.assertEqual(w.ndim, 1)
        np.testing.assert_allclose(w.values, WAVELENGTHS["wavelength_floris"], rtol=1e-6)

    def test_dims_and_coords(self):
        ds = data_io.read_dataset(self.path, ["sif_emission_spectrum"], self.grid, self.wavelengths)
        self.assertEqual(ds["sif_emission_spectrum"].dims, ("wavelength_sif", "y", "x"))
        for coord in ("x", "y", "latitude", "longitude", "wavelength_sif"):
            self.assertIn(coord, ds.coords)
        self.assertGreater(ds["y"].values[0], ds["y"].values[-1])  # north-up: y decreases

    def test_packed_values_unpacked_and_fill_to_nan(self):
        ds = data_io.read_dataset(self.path, ["olci_apparent_reflectance"], self.grid, self.wavelengths)
        values = ds["olci_apparent_reflectance"].values
        self.assertTrue(np.isnan(values[:, 0, 0]).all())
        self.assertTrue(((values[~np.isnan(values)] > 0) & (values[~np.isnan(values)] < 1)).all())
        self.assertEqual(values.dtype, np.float32)

    def test_instrument_labels(self):
        ds = data_io.read_dataset(self.path, ["sun_zenith_angle"], self.grid, self.wavelengths)
        self.assertEqual(list(ds["instrument"].values), ["FLORIS", "OLCI", "SLSTR_nad", "SLSTR_obl"])
        self.assertEqual(ds["sun_zenith_angle"].dims, ("instrument", "y", "x"))

    def test_o2_uncertainty_uses_o2_band_dim(self):
        ds = data_io.read_dataset(
            self.path, ["sif_O2_bands_value", "sif_O2_bands_value_uncertainty"], self.grid, self.wavelengths
        )
        self.assertEqual(ds["sif_O2_bands_value_uncertainty"].dims, ds["sif_O2_bands_value"].dims)

    def test_subset_applied(self):
        subset = FLEXSubset(
            x_indices=np.array([2, 3]),
            y_indices=np.array([0, 1, 2]),
            wavelength_indices={"wavelength_sif": np.array([1, 2])},
        )
        ds = data_io.read_dataset(self.path, ["sif_emission_spectrum"], self.grid, self.wavelengths, subset)
        self.assertEqual(ds["sif_emission_spectrum"].shape, (2, 3, 2))
        np.testing.assert_allclose(ds["x"].values, self.grid.x[[2, 3]])
        np.testing.assert_allclose(ds["wavelength_sif"].values, WAVELENGTHS["wavelength_sif"][[1, 2]], rtol=1e-6)

    def test_empty_wavelength_selection_omits_variable_and_its_uncertainty(self):
        subset = FLEXSubset(wavelength_indices={"wavelength_sif": np.array([], dtype=int)})
        with self.assertWarns(UserWarning):
            ds = data_io.read_dataset(
                self.path,
                ["sif_emission_spectrum", "sif_emission_spectrum_uncertainty", "total_integrated_sif"],
                self.grid,
                self.wavelengths,
                subset,
            )
        self.assertEqual(list(ds.data_vars), ["total_integrated_sif"])

    def test_unknown_variable_raises(self):
        with self.assertRaises(ValueError):
            data_io.read_dataset(self.path, ["nope"], self.grid, self.wavelengths)

    def test_uncertainty_of_variable_without_one_raises(self):
        with self.assertRaises(ValueError):
            data_io.read_dataset(self.path, ["PSI_PSII_relative_contribution_uncertainty"], self.grid, self.wavelengths)

    def test_register_uncertainties(self):
        ds = data_io.read_dataset(
            self.path, ["sif_emission_spectrum", "sif_emission_spectrum_uncertainty"], self.grid, self.wavelengths
        )
        ds = data_io.register_uncertainties(ds)
        self.assertEqual(ds["sif_emission_spectrum"].attrs["unc_comps"], "sif_emission_spectrum_uncertainty")
        attrs = ds["sif_emission_spectrum_uncertainty"].attrs
        self.assertEqual(attrs["err_corr_1_form"], "random")
        self.assertEqual(attrs["err_corr_2_dim"], ["wavelength_sif"])

    def test_register_uncertainties_without_base_warns(self):
        ds = data_io.read_dataset(self.path, ["sif_emission_spectrum_uncertainty"], self.grid, self.wavelengths)
        with self.assertWarnsRegex(UserWarning, "no matching variable"):
            data_io.register_uncertainties(ds)


if __name__ == "__main__":
    unittest.main()
