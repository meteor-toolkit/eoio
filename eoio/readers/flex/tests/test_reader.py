"""Tests for eoio.readers.flex.reader"""

import datetime as dt
import os
import tempfile
import unittest

import netCDF4
import numpy as np

from eoio.readers.factory import ReaderFactory
from eoio.readers.flex.data_io import DEFAULT_CHUNKS
from eoio.readers.flex.reader import FLEXL2Reader
from eoio.readers.flex.tests.synthetic import N, write_synthetic_flex
from eoio.readers.flex.variables import VARIABLES, variables_for

NAME = "flx_l2__flxsyn_20190930T102242_20190930T102254_20260312T190507_31TDG_100_036_01.nc"
NAME_32632 = "flx_l2__flxsyn_20250727T085700_20250727T090000_20260410T090707_31TDG_100_036_01.nc"


class TestFLEXL2Reader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, NAME)
        write_synthetic_flex(cls.path)
        cls.path32632 = os.path.join(cls.tmp.name, NAME_32632)
        write_synthetic_flex(cls.path32632, epsg=32632, x0=680706.0, y0=5025934.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_factory_selects_reader(self):
        self.assertIs(ReaderFactory.get_reader(self.path), FLEXL2Reader)
        self.assertIs(ReaderFactory.get_reader(self.path32632), FLEXL2Reader)

    def test_default_reads_all_meas_with_uncertainties(self):
        ds = FLEXL2Reader(self.path).open()
        meas = set(variables_for("meas"))
        self.assertTrue(meas.issubset(ds.data_vars))
        self.assertFalse(set(variables_for("aux")) & set(ds.data_vars))
        self.assertIn("leaf_area_index_uncertainty", ds)
        self.assertIn("leaf_area_index_uncertainty", ds.unc["leaf_area_index"].keys())

    def test_include_uncertainties_false(self):
        ds = FLEXL2Reader(self.path, read_params={"include_uncertainties": False}).open()
        self.assertFalse([v for v in ds.data_vars if str(v).endswith("_uncertainty")])

    def test_uncertainties_only_for_selected_variables(self):
        ds = FLEXL2Reader(self.path, vars_sel={"meas": ["leaf_area_index"], "aux": ["direct_irradiance"]}).open()
        self.assertEqual(
            sorted(ds.data_vars),
            ["direct_irradiance", "direct_irradiance_uncertainty", "leaf_area_index", "leaf_area_index_uncertainty"],
        )

    def test_group_shortcuts(self):
        ds = FLEXL2Reader(self.path, vars_sel={"meas": "olci", "aux": "geometry", "mask": "all"}).open()
        self.assertEqual(
            sorted(ds.data_vars),
            [
                "cirrus_cloud_mask",
                "olci_apparent_reflectance",
                "relative_azimuth_angle",
                "sun_zenith_angle",
                "viewing_zenith_angle",
            ],
        )

    def test_unknown_selection_raises(self):
        with self.assertRaises(ValueError):
            FLEXL2Reader(self.path, vars_sel={"meas": "not_a_variable"}).open()

    def test_all_variables_readable(self):
        ds = FLEXL2Reader(self.path, vars_sel={"meas": "all", "aux": "all", "mask": "all"}).open()
        for name in VARIABLES:
            self.assertIn(name, ds.data_vars)

    def test_wavelength_info_kept_per_grid(self):
        ds = FLEXL2Reader(
            self.path, vars_sel={"meas": ["olci_apparent_reflectance", "floris_apparent_reflectance"]}
        ).open()
        self.assertEqual(ds["olci_apparent_reflectance"].sizes["wavelength_olci"], 5)
        self.assertEqual(ds["floris_apparent_reflectance"].sizes["wavelength_floris"], 8)

    def test_values_outside_valid_range_left_alone(self):
        path = os.path.join(self.tmp.name, "range", NAME)
        os.makedirs(os.path.dirname(path))
        write_synthetic_flex(path)
        with netCDF4.Dataset(path, "a") as nc:
            var = nc["L2_Vegetation"]["leaf_area_index"]
            var.valid_min, var.valid_max = 0.0, 8.0
            var[1, 1] = -5.0  # outside the product's stated valid range
        ds = FLEXL2Reader(path, vars_sel={"meas": ["leaf_area_index"]}).open()
        self.assertEqual(float(ds["leaf_area_index"].values[1, 1]), -5.0)
        self.assertEqual(ds["leaf_area_index"].attrs["product_metadata"]["valid_max"], 8.0)

    def test_roi_and_wavelength_subset(self):
        full = FLEXL2Reader(self.path, vars_sel={"meas": ["sif_emission_spectrum"]}).open()
        x, y = full["x"].values, full["y"].values
        roi = (x[2] - 10, y[4] - 10, x[3] + 10, y[3] + 10)
        ds = FLEXL2Reader(
            self.path,
            vars_sel={"meas": ["sif_emission_spectrum"]},
            subset={"roi": roi, "roi_crs": 32631, "wavelength": {"min": 700, "max": 760}},
        ).open()
        self.assertEqual(ds.sizes["x"], 2)
        self.assertEqual(ds.sizes["y"], 2)
        wl = ds["wavelength_sif"].values
        self.assertTrue(((wl >= 700) & (wl <= 760)).all())
        self.assertEqual(ds["sif_emission_spectrum_uncertainty"].shape, ds["sif_emission_spectrum"].shape)

    def test_roi_in_latlon(self):
        full = FLEXL2Reader(self.path, vars_sel={"meas": ["leaf_area_index"]}).open()
        lon, lat = full["longitude"].values, full["latitude"].values
        roi = (float(lon.min()), float(lat.min()), float(lon.mean()), float(lat.mean()))
        ds = FLEXL2Reader(self.path, vars_sel={"meas": ["leaf_area_index"]}, subset={"roi": roi}).open()
        self.assertLess(ds.sizes["x"], N)
        self.assertLess(ds.sizes["y"], N)

    def test_basic_metadata(self):
        ds = FLEXL2Reader(self.path, vars_sel={"meas": ["leaf_area_index"]}).open()
        self.assertEqual(ds.attrs["eoio:reader"], "flex")
        self.assertEqual(ds.attrs["processing_level"], 2)
        self.assertEqual(ds.attrs["product_datetime"], dt.datetime(2019, 9, 30, 10, 22, 42, tzinfo=dt.timezone.utc))
        self.assertEqual(ds.attrs["crs"], "EPSG:32631")
        self.assertEqual(ds.attrs["footprint"]["crs"], "EPSG:32631")
        self.assertIn("product_metadata", ds.attrs)

    def test_explicit_start_time_attribute_overrides_filename(self):
        path = os.path.join(self.tmp.name, "time", NAME)
        os.makedirs(os.path.dirname(path))
        write_synthetic_flex(path)
        with netCDF4.Dataset(path, "a") as nc:
            nc.time_coverage_start = "2019-09-30T10:23:00Z"
        ds = FLEXL2Reader(path, vars_sel={"meas": ["leaf_area_index"]}).open()
        self.assertEqual(ds.attrs["product_datetime"], dt.datetime(2019, 9, 30, 10, 23, 0, tzinfo=dt.timezone.utc))

    def test_metadata_basic_level_and_none(self):
        basic = FLEXL2Reader(
            self.path, vars_sel={"meas": ["leaf_area_index"]}, read_params={"metadata_level": "basic"}
        ).open()
        self.assertNotIn("product_metadata", basic.attrs)
        self.assertIn("units", basic["leaf_area_index"].attrs)
        none = FLEXL2Reader(
            self.path, vars_sel={"meas": ["leaf_area_index"]}, read_params={"metadata_level": None}
        ).open()
        self.assertNotIn("eoio:reader", none.attrs)

    def test_crs_read_per_file(self):
        ds = FLEXL2Reader(self.path32632, vars_sel={"meas": ["leaf_area_index"]}).open()
        self.assertEqual(ds.attrs["crs"], "EPSG:32632")
        self.assertEqual(ds.attrs["product_datetime"], dt.datetime(2025, 7, 27, 8, 57, 0, tzinfo=dt.timezone.utc))
        self.assertAlmostEqual(float(ds["x"].values[0]), 680706.0 + 150.0)

    def test_use_chunks_is_lazy_and_matches_eager(self):
        vars_sel = {"meas": ["floris_apparent_reflectance", "leaf_area_index"], "aux": ["sun_zenith_angle"]}
        eager = FLEXL2Reader(self.path, vars_sel=vars_sel).open()
        lazy = FLEXL2Reader(
            self.path, vars_sel=vars_sel, read_params={"use_chunks": True, "chunks": {"y": 5, "x": 5}}
        ).open()
        for name in ("floris_apparent_reflectance", "floris_apparent_reflectance_uncertainty", "sun_zenith_angle"):
            self.assertIsNotNone(lazy[name].chunks, name)
            self.assertIsNone(eager[name].chunks, name)
            np.testing.assert_array_equal(lazy[name].values, eager[name].values)
        self.assertEqual(lazy["leaf_area_index"].chunks[0][0], 5)
        self.assertEqual(
            lazy["floris_apparent_reflectance"].attrs["unc_comps"], "floris_apparent_reflectance_uncertainty"
        )

    def test_chunks_resolution(self):
        def chunks(read_params):
            return FLEXL2Reader(self.path, read_params=read_params)._chunks()

        self.assertIsNone(chunks({}))
        self.assertEqual(chunks({"use_chunks": True}), DEFAULT_CHUNKS)
        self.assertEqual(chunks({"chunks": {"y": 8, "x": 4}}), {"y": 8, "x": 4})

    def test_use_chunks_with_subset(self):
        ds = FLEXL2Reader(
            self.path,
            vars_sel={"meas": ["sif_emission_spectrum"]},
            subset={"wavelength": {"min": 700, "max": 760}},
            read_params={"use_chunks": True},
        ).open()
        self.assertIsNotNone(ds["sif_emission_spectrum"].chunks)
        self.assertTrue(((ds["wavelength_sif"] >= 700) & (ds["wavelength_sif"] <= 760)).all())

    def test_all_options(self):
        options = FLEXL2Reader(self.path).all_options
        self.assertIn("olci", options["vars_sel"]["meas"])
        self.assertIn("atmosphere", options["vars_sel"]["aux"])
        self.assertIn("cirrus_cloud_mask", options["vars_sel"]["mask"])


if __name__ == "__main__":
    unittest.main()
