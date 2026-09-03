"""eoio.read - tests for eoio.read"""

import os
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
import xarray as xr

from eoio.interface import (
    # mid_lon_lat,
    # product_bounds,
    read,
    read_multi,
    process,
    product_processors,
    _resolve_multi_paths,
    # write,
)
from eoio.processors.registry import PROCESSOR_REGISTRY
from eoio.readers.base import BaseReader
import eoio.processors  # noqa: F401 - ensures all built-in processors are registered

__author__ = [
    "Sam Hunt <sam.hunt@npl.co.uk>",
    "Mattea Goalen <mattea.goalen@npl.co.uk>",
    "Maddie Stedman <maddie.stedman@npl.co.uk>",
]
__all__ = []


class MockReaderFactory:
    def get_reader(self, path):
        return


class MockReader(BaseReader):
    def __init__(self, path: str):
        self._all_options = None

    @staticmethod
    def get_extension() -> str:
        return ""

    def extract_metadata(self):
        pass

    @property
    def all_options(self):
        if self._all_options is None:
            self._all_options = {"options": "values"}
        return self._all_options

    def open_dataset(self) -> xr.Dataset:
        return xr.Dataset()


class TestInterface(unittest.TestCase):
    @mock.patch("eoio.interface.os.path.exists")
    @mock.patch("eoio.utils.read_utils.extract_file")
    @mock.patch("eoio.interface.ReaderFactory")
    def test_read(self, reader_factory_mock, mock_extract_file, mock_path_exists):
        mock_extract_file.return_value = "path_string", {}, False
        mock_path_exists.return_value = True

        reader_mock = mock.MagicMock()
        reader_mock.open.return_value = "test_ds"
        get_reader_mock = mock.MagicMock(return_value=reader_mock)
        reader_factory_mock().get_reader.return_value = get_reader_mock

        # reader_factory_mock().get_reader()().open_dataset.return_value = "test_ds"
        test_ds = read("path_string")
        self.assertEqual(test_ds, "test_ds")

        mock_extract_file.assert_called_once_with("path_string", None)
        mock_path_exists.assert_called_once_with("path_string")

    def test_product_processors_does_not_raise_for_any_registered_processor(self):
        """Regression test: every registered processor must define `_all_options`."""
        info = product_processors("unused_path")
        self.assertEqual(set(info), set(PROCESSOR_REGISTRY))
        for name, cls in PROCESSOR_REGISTRY.items():
            self.assertIs(info[name], cls._all_options)

    @mock.patch("eoio.interface.ProcessorPipeline")
    def test_process(self, processor_pipeline_mock):
        """
        process() should instantiate ProcessorPipeline, run it,
        and return the processed dataset.
        """
        ds = xr.Dataset({"a": ("x", [1, 2, 3])})
        processed_ds = xr.Dataset({"a": ("x", [2, 4, 6])})

        processors = {"dummy_processor": {"param": 1}}
        context = {"ctx": "value"}

        pipeline_instance = mock.MagicMock()
        pipeline_instance.run.return_value = processed_ds
        processor_pipeline_mock.return_value = pipeline_instance

        result = process(ds, processors, context)

        processor_pipeline_mock.assert_called_once_with(
            processor_params=processors,
            context=context,
        )
        pipeline_instance.run.assert_called_once_with(ds)
        self.assertIs(result, processed_ds)


def _time_ds(times, value):
    return xr.Dataset({"spectra": ("time", np.full(len(times), value))}, coords={"time": times})


class TestResolveMultiPaths(unittest.TestCase):
    def test_list_returned_as_is(self):
        paths = ["a.nc", "b.nc"]
        self.assertEqual(_resolve_multi_paths(paths), paths)

    def test_single_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "a.nc")
            open(fp, "w").close()
            self.assertEqual(_resolve_multi_paths(fp), [fp])

    def test_glob_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.nc", "b.nc", "c.txt"]:
                open(os.path.join(d, name), "w").close()
            resolved = _resolve_multi_paths(os.path.join(d, "*.nc"))
            self.assertEqual(sorted(resolved), sorted(os.path.join(d, n) for n in ["a.nc", "b.nc"]))

    def test_directory_filters_by_reader_factory(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.nc", "b.nc", "README.txt"]:
                open(os.path.join(d, name), "w").close()

            def fake_get_reader(path):
                if path.endswith(".nc"):
                    return object()
                raise ValueError("no reader")

            with mock.patch("eoio.interface.ReaderFactory") as reader_factory_mock:
                reader_factory_mock.return_value.get_reader.side_effect = fake_get_reader
                resolved = _resolve_multi_paths(d)

            self.assertEqual(sorted(resolved), sorted(os.path.join(d, n) for n in ["a.nc", "b.nc"]))

    def test_no_files_found_raises(self):
        with self.assertRaises(ValueError):
            _resolve_multi_paths("/nonexistent/path/*.nc")

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            _resolve_multi_paths(123)


class TestReadMulti(unittest.TestCase):
    def test_concatenates_datasets_from_each_path(self):
        ds1 = _time_ds([0, 1], 1.0)
        ds2 = _time_ds([2], 2.0)
        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]) as read_mock:
            result = read_multi(["a.nc", "b.nc"])

        self.assertEqual(read_mock.call_count, 2)
        self.assertEqual(list(result["spectra"].values), [1.0, 1.0, 2.0])
        self.assertIn("eoio:processing_steps", result.attrs)

    def test_read_args_forwarded_identically_per_file(self):
        ds = _time_ds([0], 1.0)
        with mock.patch("eoio.interface.read", return_value=ds) as read_mock:
            read_multi(
                ["a.nc", "b.nc"],
                vars_sel={"meas": ["B1"]},
                subset={"roi": "x"},
                read_params={"metadata_level": None},
                processors={"stack": {}},
            )

        for call in read_mock.call_args_list:
            self.assertEqual(call.kwargs["vars_sel"], {"meas": ["B1"]})
            self.assertEqual(call.kwargs["subset"], {"roi": "x"})
            self.assertEqual(call.kwargs["read_params"], {"metadata_level": None})
            self.assertEqual(call.kwargs["processors"], {"stack": {}})

    def test_on_mismatch_error_propagates(self):
        ds1 = _time_ds([0], 1.0)
        ds2 = xr.Dataset({"other": ("time", [1.0])}, coords={"time": [1]})
        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            with self.assertRaises(ValueError):
                read_multi(["a.nc", "b.nc"])

    def test_concat_processors_run_once_on_final_dataset(self):
        ds1 = _time_ds([0], 1.0)
        ds2 = _time_ds([1], 2.0)
        processed = _time_ds([0, 1], 9.0)
        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            with mock.patch("eoio.interface.process", return_value=processed) as process_mock:
                result = read_multi(["a.nc", "b.nc"], concat_processors={"sort": {}})

        process_mock.assert_called_once()
        self.assertIs(result, processed)

    def test_raster_stack_creates_new_dimension_via_concat_coord(self):
        ds1 = xr.Dataset({"reflectance_10m": (("y_10m", "x_10m"), np.full((2, 2), 1.0))})
        ds1.attrs["sensing_time"] = "2026-07-01"
        ds2 = xr.Dataset({"reflectance_10m": (("y_10m", "x_10m"), np.full((2, 2), 2.0))})
        ds2.attrs["sensing_time"] = "2026-07-05"

        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            result = read_multi(["a.nc", "b.nc"], concat_coord="sensing_time")

        self.assertEqual(result["reflectance_10m"].dims, ("time", "y_10m", "x_10m"))
        self.assertEqual(list(result.coords["time"].values), ["2026-07-01", "2026-07-05"])

    def test_raster_grid_mismatch_governed_by_on_mismatch(self):
        ds1 = xr.Dataset({"reflectance_10m": (("y_10m", "x_10m"), np.full((2, 2), 1.0))})
        ds1.attrs["sensing_time"] = "2026-07-01"
        ds2 = xr.Dataset({"reflectance_10m": (("y_10m", "x_10m"), np.full((3, 2), 2.0))})  # mismatched grid
        ds2.attrs["sensing_time"] = "2026-07-05"

        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            with self.assertRaises(ValueError):
                read_multi(["a.nc", "b.nc"], concat_coord="sensing_time")

    def test_attrs_reconciled_by_default_not_silently_dropped(self):
        ds1 = _time_ds([0], 1.0)
        ds1.attrs["product_name"] = "scene_A"
        ds2 = _time_ds([1], 2.0)
        ds2.attrs["product_name"] = "scene_B"

        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            result = read_multi(["a.nc", "b.nc"])

        self.assertEqual(result.attrs["product_name"], ["scene_A", "scene_B"])

    def test_attrs_first_keeps_legacy_behaviour(self):
        ds1 = _time_ds([0], 1.0)
        ds1.attrs["product_name"] = "scene_A"
        ds2 = _time_ds([1], 2.0)
        ds2.attrs["product_name"] = "scene_B"

        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            result = read_multi(["a.nc", "b.nc"], attrs="first")

        self.assertEqual(result.attrs["product_name"], "scene_A")

    def test_coord_attrs_promotes_attr_to_real_coordinate(self):
        ds1 = _time_ds([0], 1.0)
        ds1.attrs["product_name"] = "scene_A"
        ds2 = _time_ds([1], 2.0)
        ds2.attrs["product_name"] = "scene_B"

        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            result = read_multi(["a.nc", "b.nc"], coord_attrs=["product_name"])

        self.assertNotIn("product_name", result.attrs)
        self.assertEqual(list(result.coords["product_name"].values), ["scene_A", "scene_B"])

    def test_concatenates_variable_with_repeated_dimension(self):
        # e.g. a (measurement, measurement) error-correlation matrix - some
        # real EO products legitimately use this, even though xr.concat can't
        # handle a repeated dim directly.
        def corr_ds(value, t):
            corr = np.eye(3, dtype="float32") * value
            return xr.Dataset(
                {"err_corr_radiance": (("measurement", "measurement"), corr)},
                coords={"time": [t]},
            )

        ds1, ds2 = corr_ds(1.0, 0), corr_ds(2.0, 1)
        with mock.patch("eoio.interface.read", side_effect=[ds1, ds2]):
            result = read_multi(["a.nc", "b.nc"])

        self.assertEqual(result["err_corr_radiance"].dims, ("time", "measurement", "measurement"))
        self.assertEqual(dict(result.sizes), {"time": 2, "measurement": 3})
        np.testing.assert_array_equal(result["err_corr_radiance"].isel(time=1).values, np.eye(3, dtype="float32") * 2.0)


if __name__ == "__main__":
    unittest.main()
