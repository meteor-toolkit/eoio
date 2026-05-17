"""Tests for eoio.processors.interpolate.processor

Format and style follow other tests in the repository (unittest-based).
"""

import unittest
import xarray as xr
import numpy as np

from eoio.processors.interpolate.processor import Interpolate


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


if __name__ == "__main__":
    unittest.main()
