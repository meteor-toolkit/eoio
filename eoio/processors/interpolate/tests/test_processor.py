"""Tests for eoio.processors.interpolate.processor

Format and style follow other tests in the repository (unittest-based).
"""

import unittest
import xarray as xr
import numpy as np

from eoio.processors.interpolate.processor import Interpolate, _grid_suffix_from_dims, _refresh_grid_metadata


class TestInterpolateProcessor(unittest.TestCase):
    def test_init_raises_if_missing_coords(self):
        with self.assertRaises(ValueError):
            Interpolate(params={})

    def test_init_raises_if_missing_target_grid(self):
        with self.assertRaises(ValueError):
            Interpolate(params={"coords": ["x"]})

    def test_init_raises_if_length_mismatch(self):
        with self.assertRaises(ValueError):
            Interpolate(params={"coords": ["x", "y"], "target_grid": [[0]]})

    def test_init_raises_on_invalid_on_missing(self):
        with self.assertRaises(ValueError):
            Interpolate(params={"coords": ["x"], "target_grid": [[0]], "on_missing": "bad"})

    def test_format_coords_missing_raises(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0]]})
        ds = xr.Dataset({"a": ("x", [1, 2])}, coords={"x": [0, 1]})
        with self.assertRaises(ValueError):
            inst._format_coords(ds, ["y"])  # 'y' not present

    def test_format_target_grid_resolves_string_coordinate(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": ["x"]})
        ds = xr.Dataset({"a": ("x", [1, 2])}, coords={"x": [0, 1]})

        tg = inst._format_target_grid(ds, ["x"])  # should return the DataArray for coord 'x'
        self.assertEqual(len(tg), 1)
        # compare values of the returned DataArray to the original coordinate
        self.assertTrue((tg[0].values == ds.coords["x"].values).all())

    def test_run_numeric_target_coords(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0.0, 1.0]], "inplace": True})

        ds = xr.Dataset({"var": ("x", [0.0, 10.0, 20.0])}, coords={"x": [0.0, 1.0, 2.0]})

        # call the interpolate method directly (avoids run() which orchestrates helpers)
        result = inst.run(ds)

        # result should have interpolated values at x = [0.0, 1.0]
        self.assertIn("var", result)
        self.assertEqual(result["var"].shape[0], 2)
        self.assertAlmostEqual(float(result["var"].values[0]), 0.0)
        self.assertAlmostEqual(float(result["var"].values[1]), 10.0)

    def test_parse_params_normalises_method_and_accepts_valid(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0]], "method": "LINEAR"})
        # method should be normalised to lower-case
        self.assertEqual(inst.interpolate_config.method, "linear")

    def test_format_target_grid_missing_string_raises(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": ["z"]})
        ds = xr.Dataset({"a": ("x", [1])}, coords={"x": [0]})
        with self.assertRaises(ValueError):
            inst._format_target_grid(ds, ["z"])  # 'z' not a coord on ds

    def test_run_end_to_end_and_provenance_added(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0.0, 1.0]], "inplace": True})
        ds = xr.Dataset({"var": ("x", [0.0, 10.0, 20.0])}, coords={"x": [0.0, 1.0, 2.0]})

        out = inst.run(ds)
        # interpolated dataset should have two values along x
        self.assertIn("var", out)
        self.assertEqual(out["var"].shape[0], 2)

    def test_run_provenance_added(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0.0, 1.0]], "inplace": True})
        ds = xr.Dataset({"var": ("x", [0.0, 10.0, 20.0])}, coords={"x": [0.0, 1.0, 2.0]})

        out = inst.run(ds)
        # provenance step appended
        steps = out.attrs.get("eoio:processing_steps")
        self.assertIsNotNone(steps)
        self.assertIsInstance(steps, list)

    def test_record_provenance_handles_existing_non_list(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0]]})
        ds = xr.Dataset({"a": ("x", [1, 2])}, coords={"x": [0, 1]})
        ds.attrs["eoio:processing_steps"] = "previous"

        out = inst._record_provenance(ds)
        steps = out.attrs["eoio:processing_steps"]
        # previous string should be preserved as a list element and new dict appended
        self.assertIsInstance(steps, list)
        self.assertEqual(str(steps[0]), "previous")
        self.assertEqual(steps[-1]["processor"], "interpolate")

    def test_2d_interpolation(self):
        # Build a simple 2x2 grid dataset
        x = [0.0, 1.0]
        y = [0.0, 1.0]

        vals = np.array([[0.0, 1.0], [10.0, 11.0]])
        ds = xr.Dataset({"z": (("x", "y"), vals)}, coords={"x": x, "y": y})

        inst = Interpolate(params={"coords": ["x", "y"], "target_grid": [[0.0, 1.0], [0.0, 1.0]]})
        out = inst.run(ds)
        self.assertIn("z", out)
        # shape should be (2,2)
        self.assertEqual(out["z"].shape, (2, 2))

    def test_run_raises_on_non_dataset(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0]]})
        with self.assertRaises(TypeError):
            inst.run("not a dataset")

    def test_run_inplace_false(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0.0, 1.0]]})
        ds = xr.Dataset({"var": ("x", [0.0, 10.0, 20.0])}, coords={"x": [0.0, 1.0, 2.0]})

        out = inst.run(ds)
        # check that the output has the expected variable
        self.assertIn("var_interp", out)

    def test_record_provenance_appends_step(self):
        inst = Interpolate(params={"coords": ["x"], "target_grid": [[0]]})
        ds = xr.Dataset({"a": ("x", [1, 2])}, coords={"x": [0, 1]})

        out = inst._record_provenance(ds)
        self.assertIn("eoio:processing_steps", out.attrs)
        steps = out.attrs["eoio:processing_steps"]
        self.assertIsInstance(steps, list)
        self.assertGreaterEqual(len(steps), 1)
        last = steps[-1]
        self.assertIsInstance(last, dict)
        self.assertEqual(last.get("processor"), "interpolate")
        self.assertEqual(last.get("interpolated_coords"), inst.interpolate_config.coords)


class TestGridSuffixFromDims(unittest.TestCase):
    def test_resolution_dims(self):
        self.assertEqual(_grid_suffix_from_dims(("y_60m", "x_60m")), "60m")

    def test_generic_dims_returns_none(self):
        self.assertIsNone(_grid_suffix_from_dims(("y", "x")))

    def test_non_grid_underscore_prefix(self):
        # No special-casing -- any "x_"/"y_" prefixed dim is treated as a grid suffix,
        # matching the identical helper in stack/datatree processors.
        self.assertEqual(_grid_suffix_from_dims(("x_interp",)), "interp")


class TestRefreshGridMetadata(unittest.TestCase):
    def _ds(self):
        ds = xr.Dataset(
            {
                "reflectance_60m": (("y_60m", "x_60m"), np.zeros((2, 2))),
                "angle_interp": (("y_60m", "x_60m"), np.zeros((2, 2))),
            }
        )
        ds["reflectance_60m"].attrs["product_metadata"] = {
            "geometry_id": "60m",
            "spatial_resolution": 60,
            "geoposition": {"origin": "60m-origin"},
        }
        ds["angle_interp"].attrs["product_metadata"] = {"geometry_id": "5000m", "spatial_resolution": 5000}
        return ds

    def test_copies_grid_keys_from_donor(self):
        ds = self._ds()
        _refresh_grid_metadata(ds, ["angle_interp"])
        pm = ds["angle_interp"].attrs["product_metadata"]
        self.assertEqual(pm["geometry_id"], "60m")
        self.assertEqual(pm["spatial_resolution"], 60)
        self.assertEqual(pm["geoposition"], {"origin": "60m-origin"})

    def test_donor_metadata_is_a_copy_not_aliased(self):
        ds = self._ds()
        _refresh_grid_metadata(ds, ["angle_interp"])
        ds["angle_interp"].attrs["product_metadata"]["geoposition"]["origin"] = "mutated"
        self.assertEqual(ds["reflectance_60m"].attrs["product_metadata"]["geoposition"]["origin"], "60m-origin")

    def test_does_not_copy_band_specific_keys_from_donor(self):
        # Regression for a real crash: two non-band-specific variables (e.g. Sentinel-2's
        # scene-wide solar zenith/azimuth angles) independently interpolated onto the same
        # grid, both borrowing product_metadata from the same donor band, must NOT both end
        # up with that donor's band_central_wavelength -- that caused eoio.processors.stack
        # to see two "different" variables claiming the same wavelength and crash with a
        # pandas "duplicate values" alignment error when concatenating them.
        ds = xr.Dataset(
            {
                "B02": (("y_10m", "x_10m"), np.zeros((2, 2))),
                "solar_zenith_angle_10m": (("y_10m", "x_10m"), np.zeros((2, 2))),
                "solar_azimuth_angle_10m": (("y_10m", "x_10m"), np.zeros((2, 2))),
            }
        )
        ds["B02"].attrs["product_metadata"] = {
            "geometry_id": "10m",
            "spatial_resolution": 10,
            "band_central_wavelength": 492.4,
            "band_id": 1,
        }
        ds["solar_zenith_angle_10m"].attrs["product_metadata"] = {"geometry_id": "5000m"}
        ds["solar_azimuth_angle_10m"].attrs["product_metadata"] = {"geometry_id": "5000m"}

        _refresh_grid_metadata(ds, ["solar_zenith_angle_10m", "solar_azimuth_angle_10m"])

        for name in ["solar_zenith_angle_10m", "solar_azimuth_angle_10m"]:
            pm = ds[name].attrs["product_metadata"]
            self.assertEqual(pm["geometry_id"], "10m")
            self.assertNotIn("band_central_wavelength", pm)
            self.assertNotIn("band_id", pm)

    def test_removes_grid_key_absent_from_donor(self):
        # A grid-identity key that isn't even present on the donor is stale info from the
        # variable's old grid and should be dropped, not kept.
        ds = xr.Dataset(
            {
                "reflectance_60m": (("y_60m", "x_60m"), np.zeros((2, 2))),
                "angle_interp": (("y_60m", "x_60m"), np.zeros((2, 2))),
            }
        )
        ds["reflectance_60m"].attrs["product_metadata"] = {"geometry_id": "60m"}  # no geoposition
        ds["angle_interp"].attrs["product_metadata"] = {"geometry_id": "5000m", "geoposition": {"origin": "stale"}}

        _refresh_grid_metadata(ds, ["angle_interp"])

        self.assertNotIn("geoposition", ds["angle_interp"].attrs["product_metadata"])

    def test_falls_back_to_patching_when_no_donor(self):
        ds = xr.Dataset({"angle_interp": (("y_20m", "x_20m"), np.zeros((2, 2)))})
        ds["angle_interp"].attrs["product_metadata"] = {
            "geometry_id": "5000m",
            "spatial_resolution": 5000,
            "geoposition": {"origin": "5000m-origin"},
        }
        with self.assertWarns(UserWarning):
            _refresh_grid_metadata(ds, ["angle_interp"])
        pm = ds["angle_interp"].attrs["product_metadata"]
        self.assertEqual(pm["geometry_id"], "20m")
        self.assertEqual(pm["spatial_resolution"], 20)
        self.assertNotIn("geoposition", pm)

    def test_noop_when_no_grid_suffix_in_dims(self):
        ds = xr.Dataset({"var": (("y", "x"), np.zeros((2, 2)))})
        ds["var"].attrs["product_metadata"] = {"geometry_id": "5000m"}
        _refresh_grid_metadata(ds, ["var"])
        self.assertEqual(ds["var"].attrs["product_metadata"]["geometry_id"], "5000m")

    def test_noop_when_no_product_metadata_present(self):
        ds = xr.Dataset({"var": (("y_60m", "x_60m"), np.zeros((2, 2)))})
        # Should not raise even though there's nothing to refresh.
        _refresh_grid_metadata(ds, ["var"])
        self.assertNotIn("product_metadata", ds["var"].attrs)

    def test_two_vars_from_same_call_do_not_donor_from_each_other(self):
        # Both land on the same new grid in one call and neither is a valid donor for the
        # other -- each must fall back to patching its own metadata independently rather than
        # one borrowing the other's (still-stale) product_metadata wholesale.
        ds = xr.Dataset(
            {
                "solar_zenith_angle_60m": (("y_60m", "x_60m"), np.zeros((2, 2))),
                "solar_azimuth_angle_60m": (("y_60m", "x_60m"), np.zeros((2, 2))),
            }
        )
        ds["solar_zenith_angle_60m"].attrs["product_metadata"] = {"geometry_id": "5000m"}
        ds["solar_azimuth_angle_60m"].attrs["product_metadata"] = {"geometry_id": "5000m"}

        _refresh_grid_metadata(ds, ["solar_zenith_angle_60m", "solar_azimuth_angle_60m"])

        self.assertEqual(ds["solar_zenith_angle_60m"].attrs["product_metadata"]["geometry_id"], "60m")
        self.assertEqual(ds["solar_azimuth_angle_60m"].attrs["product_metadata"]["geometry_id"], "60m")


class TestInterpolateGridAttrFix(unittest.TestCase):
    """The originally-reported bug: interpolating a variable onto a named grid
    coordinate (e.g. Sentinel-2 angles at 5000m -> 60m) left its product_metadata
    pointing at the old grid, so ToDataTree's grid_attr-based grouping filed it
    into the wrong node even though its dims were correctly on the new grid."""

    def _ds(self):
        ds = xr.Dataset(
            {
                "reflectance_60m": (("y_60m", "x_60m"), np.zeros((2, 2))),
                "solar_zenith_angle": (("y_5000m", "x_5000m"), np.array([[10.0, 20.0], [30.0, 40.0]])),
            },
            coords={"x_60m": [0.0, 1.0], "y_60m": [0.0, 1.0], "x_5000m": [0.0, 1.0], "y_5000m": [0.0, 1.0]},
        )
        ds["reflectance_60m"].attrs["product_metadata"] = {"geometry_id": "60m", "spatial_resolution": 60}
        ds["solar_zenith_angle"].attrs["product_metadata"] = {"geometry_id": "5000m", "spatial_resolution": 5000}
        return ds

    def test_named_coord_target_renamed_with_resolution_suffix_not_interp(self):
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["solar_zenith_angle"],
            }
        )
        out = inst.run(self._ds())
        self.assertIn("solar_zenith_angle_60m", out.data_vars)
        self.assertNotIn("solar_zenith_angle_interp", out.data_vars)

    def test_named_coord_target_updates_geometry_id(self):
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["solar_zenith_angle"],
            }
        )
        out = inst.run(self._ds())
        self.assertEqual(out["solar_zenith_angle_60m"].attrs["product_metadata"]["geometry_id"], "60m")

    def test_inplace_true_keeps_name_and_updates_geometry_id(self):
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["solar_zenith_angle"],
                "inplace": True,
            }
        )
        out = inst.run(self._ds())
        self.assertIn("solar_zenith_angle", out.data_vars)
        self.assertEqual(out["solar_zenith_angle"].dims, ("y_60m", "x_60m"))
        self.assertEqual(out["solar_zenith_angle"].attrs["product_metadata"]["geometry_id"], "60m")

    def test_raw_array_target_keeps_previous_interp_suffix_behaviour(self):
        # Guards against the false-positive described in _run_xy's comment: a raw
        # array/list target must not trigger the resolution-suffix rename, since the
        # output dim is a placeholder ("<coord>_interp"), not a real grid identity.
        ds = xr.Dataset(
            {"solar_zenith_angle": (("y_5000m", "x_5000m"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
            coords={"x_5000m": [0.0, 1.0], "y_5000m": [0.0, 1.0]},
        )
        ds["solar_zenith_angle"].attrs["product_metadata"] = {"geometry_id": "5000m"}
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": [[0.0, 1.0], [0.0, 1.0]],
                "data_vars": ["solar_zenith_angle"],
            }
        )
        out = inst.run(ds)
        self.assertIn("solar_zenith_angle_interp", out.data_vars)
        self.assertEqual(out["solar_zenith_angle_interp"].attrs["product_metadata"]["geometry_id"], "5000m")


class TestDropOriginal(unittest.TestCase):
    def _ds(self):
        return xr.Dataset(
            {"solar_zenith_angle": (("y_5000m", "x_5000m"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
            coords={"x_5000m": [0.0, 1.0], "y_5000m": [0.0, 1.0], "x_60m": [0.0, 1.0], "y_60m": [0.0, 1.0]},
        )

    def test_default_keeps_original(self):
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["solar_zenith_angle"],
            }
        )
        out = inst.run(self._ds())
        self.assertIn("solar_zenith_angle", out.data_vars)
        self.assertIn("solar_zenith_angle_60m", out.data_vars)

    def test_drop_original_true_drops_source(self):
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["solar_zenith_angle"],
                "drop_original": True,
            }
        )
        out = inst.run(self._ds())
        self.assertNotIn("solar_zenith_angle", out.data_vars)
        self.assertIn("solar_zenith_angle_60m", out.data_vars)

    def test_drop_original_is_noop_when_inplace(self):
        # Nothing separate to drop when inplace=True -- the source variable *is* the output.
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["solar_zenith_angle"],
                "inplace": True,
                "drop_original": True,
            }
        )
        out = inst.run(self._ds())
        self.assertIn("solar_zenith_angle", out.data_vars)
        self.assertEqual(out["solar_zenith_angle"].dims, ("y_60m", "x_60m"))

    def test_drop_original_does_not_drop_a_var_untouched_by_this_call(self):
        ds = self._ds()
        ds["other_5000m_var"] = (("y_5000m", "x_5000m"), np.zeros((2, 2)))
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["solar_zenith_angle"],
                "drop_original": True,
            }
        )
        out = inst.run(ds)
        self.assertIn("other_5000m_var", out.data_vars)


class TestInplacePreservesSourceGrid(unittest.TestCase):
    """inplace=True must not delete the source-grid coordinates: several interpolate
    steps in one pipeline can share a grid (e.g. Sentinel-2's 5000m tie-point grid
    feeding solar angles onto the 60m/20m/10m band grids in turn)."""

    def _ds(self):
        return xr.Dataset(
            {
                "sza": (("y_5000m", "x_5000m"), np.array([[10.0, 20.0], [30.0, 40.0]])),
                "saa": (("y_5000m", "x_5000m"), np.array([[1.0, 2.0], [3.0, 4.0]])),
            },
            coords={
                "x_5000m": [0.0, 1.0],
                "y_5000m": [0.0, 1.0],
                "x_60m": [0.0, 0.5, 1.0],
                "y_60m": [0.0, 0.5, 1.0],
                "x_20m": [0.0, 0.25, 0.5, 0.75, 1.0],
                "y_20m": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        )

    def test_inplace_true_keeps_source_coords_when_last_var_is_moved(self):
        # sza is the only var on the 5000m grid it touches -> the whole-dataset
        # interp path -> the source coords used to be dropped by the cleanup loop.
        inst = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["sza"],
                "inplace": True,
            }
        )
        out = inst.run(self._ds().drop_vars("saa"))
        self.assertIn("x_5000m", out.coords)
        self.assertIn("y_5000m", out.coords)

    def test_source_grid_still_usable_by_a_second_interpolate_step(self):
        ds = self._ds()
        first = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["sza"],
                "inplace": True,
            }
        ).run(ds)
        second = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_20m", "y_20m"],
                "data_vars": ["saa"],
                "inplace": True,
            }
        ).run(first)
        self.assertEqual(second["sza"].dims, ("y_60m", "x_60m"))
        self.assertEqual(second["saa"].dims, ("y_20m", "x_20m"))

    def test_inplace_false_leaves_the_source_grid_in_place_too(self):
        out = Interpolate(
            params={
                "coords": ["x_5000m", "y_5000m"],
                "target_grid": ["x_60m", "y_60m"],
                "data_vars": ["sza"],
            }
        ).run(self._ds())
        self.assertIn("x_5000m", out.coords)
        self.assertIn("y_5000m", out.coords)


if __name__ == "__main__":
    unittest.main()
