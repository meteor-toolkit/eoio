"""eoio.readers.tests.test_base - tests for eoio.readers.base"""

from __future__ import annotations

import unittest
import unittest.mock as mock

import xarray as xr

from eoio.readers.base import BaseReader, BaseRasterReader


class DummyReader(BaseReader):
    """Concrete BaseReader for unit tests."""

    # Provide some realistic option definitions
    meas_def = {"all": ["m1", "m2"], "basic": ["m1"]}
    aux_def = {"all": ["a1", "a2"], "basic": ["a1"]}
    mask_def = {"all": ["q1"]}

    # Used if include_uncertainties=True
    uncertainty_vars = ["u_m1", "u_m2"]

    # Provide some subset and read options for all_options()
    default_subset = {"roi": None, "wavelength": None}
    all_subset = {"roi": "TBD", "wavelength": "min/max/nearest/tolerance"}

    def open_dataset(self) -> xr.Dataset:
        return xr.Dataset({"m1": (("x",), [1, 2, 3])})

    def get_extension(self) -> str:
        return ""


class DummyRasterReader(BaseRasterReader):
    """Concrete BaseRasterReader for unit tests."""

    meas_def = {"all": ["b1"]}
    aux_def = {"all": []}
    mask_def = {"all": []}
    uncertainty_vars: list[str] = []

    def open_dataset(self) -> xr.Dataset:
        return xr.Dataset({"b1": (("x",), [0, 1])})

    def get_extension(self) -> str:
        return ""


class TestMergeAndValidate(unittest.TestCase):
    def test_merge_and_validate_merges_and_keeps_defaults(self) -> None:
        defaults = {"a": 1, "b": 2}
        provided = {"a": 10}
        merged = DummyReader._merge_and_validate(provided, defaults, kind="test")
        self.assertEqual(merged, {"a": 10, "b": 2})

    def test_merge_and_validate_rejects_unknown_keys(self) -> None:
        defaults = {"a": 1}
        with self.assertRaises(ValueError) as ctx:
            DummyReader._merge_and_validate({"b": 2}, defaults, kind="test")
        self.assertIn("Unknown test parameter(s)", str(ctx.exception))
        self.assertIn("Allowed", str(ctx.exception))


class TestBaseReaderInitAndConfig(unittest.TestCase):
    @mock.patch("eoio.readers.base.os.path.exists", return_value=False)
    def test_init_raises_if_path_missing(self, exists_mock) -> None:
        with self.assertRaises(ValueError):
            DummyReader("missing_path")
        exists_mock.assert_called_with("missing_path")

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_init_default_config_is_merged(self, exists_mock) -> None:
        r = DummyReader("dummy_path")
        # defaults from BaseReader.default_vars_sel etc
        self.assertEqual(r.config.vars_sel["meas"], "all")
        self.assertIsNone(r.config.vars_sel["aux"])
        self.assertIsNone(r.config.vars_sel["mask"])

        # resolved should expand "all" into the actual lists
        self.assertEqual(r.resolved_config.vars_sel["meas"], ["m1", "m2"])
        self.assertEqual(r.resolved_config.vars_sel["aux"], [])
        self.assertEqual(r.resolved_config.vars_sel["mask"], [])

        exists_mock.assert_called_with("dummy_path")

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_init_rejects_unknown_vars_sel_keys(self, exists_mock) -> None:
        with self.assertRaises(ValueError):
            DummyReader("dummy_path", vars_sel={"not_a_key": 1})
        exists_mock.assert_called_with("dummy_path")

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_init_rejects_unknown_subset_keys(self, exists_mock) -> None:
        with self.assertRaises(ValueError):
            DummyReader("dummy_path", subset={"not_a_key": 1})
        exists_mock.assert_called_with("dummy_path")

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_init_rejects_unknown_read_params_keys(self, exists_mock) -> None:
        with self.assertRaises(ValueError):
            DummyReader("dummy_path", read_params={"not_a_key": 1})
        exists_mock.assert_called_with("dummy_path")


class TestVariableSelection(unittest.TestCase):
    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_selected_meas_from_group(self, _) -> None:
        r = DummyReader("dummy_path", vars_sel={"meas": "basic"})
        self.assertEqual(r.list_selected_meas(), ["m1"])

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_selected_meas_from_single_name(self, _) -> None:
        r = DummyReader("dummy_path", vars_sel={"meas": "m2"})
        self.assertEqual(r.list_selected_meas(), ["m2"])

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_selected_meas_from_list(self, _) -> None:
        r = DummyReader("dummy_path", vars_sel={"meas": ["m2", "m1"]})
        self.assertEqual(r.list_selected_meas(), ["m2", "m1"])

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_selected_meas_none_means_empty(self, _) -> None:
        r = DummyReader("dummy_path", vars_sel={"meas": None})
        self.assertEqual(r.list_selected_meas(), [])

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_selected_meas_unknown_raises(self, _) -> None:
        with self.assertRaises(ValueError) as ctx:
            DummyReader("dummy_path", vars_sel={"meas": "nope"})
        self.assertIn("Unknown meas requested", str(ctx.exception))

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_selected_aux_and_mask(self, _) -> None:
        r = DummyReader(
            "dummy_path",
            vars_sel={"meas": "basic", "aux": "basic", "mask": "all"},
        )
        self.assertEqual(r.list_selected_aux(), ["a1"])
        self.assertEqual(r.list_selected_mask(), ["q1"])

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_include_vars_without_uncertainties(self, _) -> None:
        r = DummyReader(
            "dummy_path",
            vars_sel={"meas": "basic", "aux": "basic", "mask": "all"},
            read_params={"include_uncertainties": False},
        )
        self.assertEqual(r.list_include_vars(), ["m1", "a1", "q1"])

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_list_include_vars_with_uncertainties(self, _) -> None:
        r = DummyReader(
            "dummy_path",
            vars_sel={"meas": "basic", "aux": None, "mask": None},
            read_params={"include_uncertainties": True},
        )
        self.assertEqual(r.list_include_vars(), ["m1", "u_m1", "u_m2"])


class TestAllOptions(unittest.TestCase):
    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_all_options_contains_expected_sections(self, _) -> None:
        r = DummyReader("dummy_path")
        opts = r.all_options

        self.assertIn("subset", opts)
        self.assertIn("read_params", opts)
        self.assertIn("vars_sel", opts)

        self.assertIn("roi", opts["subset"])
        self.assertIn("metadata_level", opts["read_params"])

        self.assertIn("meas", opts["vars_sel"])
        self.assertIn("aux", opts["vars_sel"])
        self.assertIn("mask", opts["vars_sel"])

        # Should include group keys + any explicit variables from "all"
        self.assertIn("all", opts["vars_sel"]["meas"])
        self.assertIn("basic", opts["vars_sel"]["meas"])
        self.assertIn("m1", opts["vars_sel"]["meas"])
        self.assertIn("m2", opts["vars_sel"]["meas"])

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_all_options_is_cached(self, _) -> None:
        r = DummyReader("dummy_path")
        o1 = r.all_options
        o2 = r.all_options
        self.assertIs(o1, o2)


class TestBaseRasterReader(unittest.TestCase):
    def test_base_raster_reader_is_abstract(self) -> None:
        class Incomplete(BaseRasterReader):
            pass

        with self.assertRaises(TypeError):
            Incomplete("dummy_path")

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    def test_dummy_raster_reader_initialises_and_resolves_subset_defaults(self, _) -> None:
        r = DummyRasterReader("dummy_path")
        # BaseRasterReader provides default_subset keys
        self.assertIn("roi", r.config.subset)
        self.assertIn("roi_crs", r.config.subset)
        self.assertIn("angle", r.config.subset)
        self.assertIn("wavelength", r.config.subset)

        # default vars_sel from BaseReader: meas="all"
        self.assertEqual(r.resolved_config.vars_sel["meas"], ["b1"])


if __name__ == "__main__":
    unittest.main()
