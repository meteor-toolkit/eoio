"""Unit tests for `eoio.readers.sentinel3_slstr.reader`.

Focus on `resolve_subset` and `open_dataset` behaviour using mocks
to avoid heavy I/O.
"""

import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import xarray as xr
from pathlib import Path

from eoio.readers.sentinel3_slstr.reader import SLSTRL1Reader


# Optional real SEN3 path for integration tests (skipped when absent)
SEN3_PATH = r"T:\ECO\EOServer\data\unittest_datasets\S3SLSTR\S3B_SL_1_RBT____20230819T211849_20230819T212149_20230819T235829_0179_083_086_0900_PS2_O_NR_004.SEN3"


class TestSLSTRL1Reader(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "/tmp/fake.SEN3"

    def _ensure_aux_data(self, config):
        if "aux" not in config["vars_sel"]:
            config["vars_sel"]["aux"] = []
        if "mask" not in config["vars_sel"]:
            config["vars_sel"]["mask"] = []
        return config

    @patch("eoio.readers.sentinel3_slstr.reader.ROISubsetResolver")
    def test_resolve_subset_none_when_subset_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel3_slstr.reader.ROISubsetResolver")
    def test_resolve_subset_raises_when_image_crs_missing(self, ROISubsetResolver):
        reader = self._make_reader_minimal(product_metadata={"crs_code": None})
        subset = {"roi": [0, 0, 1, 1], "roi_crs": 4326}
        with self.assertRaises(ValueError):
            reader.resolve_subset(subset)

    @patch("eoio.readers.sentinel3_slstr.reader.ROISubsetResolver")
    def test_resolve_subset_calls_resolver(self, ROISubsetResolver):
        reader = self._make_reader_minimal(product_metadata={"crs_code": 32630})
        expected = Mock(name="resolved")
        ROISubsetResolver.return_value.run.return_value = expected
        subset = {"roi": [0, 0, 1, 1], "roi_crs": 4326}
        out = reader.resolve_subset(subset)
        self.assertIs(out, expected)
        ROISubsetResolver.assert_called_once_with(
            roi=[0, 0, 1, 1],
            roi_crs_epsg=4326,
            image_crs_epsg=32630,
            image_bounds=None,
        )
        ROISubsetResolver.return_value.run.assert_called_once_with()

    @patch("eoio.readers.sentinel3_slstr.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_empty(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel3_slstr.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_roi_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({"roi": None, "roi_crs": 4326}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel3_slstr.reader.read_lat_lon_coordinates")
    @patch("eoio.readers.sentinel3_slstr.reader.apply_conventions")
    @patch("eoio.readers.sentinel3_slstr.reader.read_bands_into_dataset")
    def test_open_dataset_reads_bands_when_meas_requested(self, read_bands, apply_conventions, read_latlon):
        reader = self._make_reader_minimal()
        config = {
            "vars_sel": {"meas": ["S1_radiance_an"]},
            "subset": Mock(name="subset"),
            "read_params": {
                "metadata_level": None,
                "use_chunks": True,
                "chunks": {"x": 256, "y": 256},
            },
        }
        config = self._ensure_aux_data(config)
        reader.resolved_config = SimpleNamespace(**config)

        read_latlon.return_value = (xr.Dataset(), None)
        ds1 = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds2 = xr.Dataset({"final": ("x", [4, 5, 6])})

        read_bands.return_value = ds1
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands.assert_called_once()
        _, kwargs = read_bands.call_args
        self.assertIs(kwargs["layout"], reader.layout)
        self.assertEqual(kwargs["meas"], ["S1_radiance_an"])
        self.assertIs(kwargs["subset"], reader.resolved_config.subset)
        self.assertIs(kwargs["mtd"], reader.mtd)
        self.assertTrue(kwargs["use_chunks"])
        self.assertEqual(kwargs["chunks"], {"x": 256, "y": 256})

        apply_conventions.assert_called_once_with(
            ds1,
            layout=reader.layout,
            config=reader.resolved_config,
        )

    @patch("eoio.readers.sentinel3_slstr.reader.read_lat_lon_coordinates")
    @patch("eoio.readers.sentinel3_slstr.reader.apply_conventions")
    @patch("eoio.readers.sentinel3_slstr.reader.read_bands_into_dataset")
    def test_open_dataset_does_not_read_bands_when_meas_vars_none(self, read_bands, apply_conventions, read_latlon):
        reader = self._make_reader_minimal()
        config = {
            "vars_sel": {"meas": []},
            "subset": None,
            "read_params": {
                "metadata_level": None,
                "use_chunks": False,
                "chunks": None,
            },
        }
        config = self._ensure_aux_data(config)
        reader.resolved_config = SimpleNamespace(**config)

        read_latlon.return_value = (xr.Dataset(), None)

        ds2 = xr.Dataset({"bar": ("x", [4, 5, 6])})
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands.assert_not_called()
        apply_conventions.assert_called_once()
        args, _ = apply_conventions.call_args
        self.assertIsInstance(args[0], xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.reader.read_lat_lon_coordinates")
    @patch("eoio.readers.sentinel3_slstr.reader.apply_conventions")
    @patch("eoio.readers.sentinel3_slstr.reader.read_bands_into_dataset")
    def test_open_dataset_attaches_metadata_when_level_all_or_basic(self, read_bands, apply_conventions, read_latlon):
        reader = self._make_reader_minimal()
        config = {
            "vars_sel": {"meas": ["S1_radiance_an"]},
            "subset": None,
            "read_params": {
                "metadata_level": "basic",
                "use_chunks": False,
                "chunks": None,
            },
        }
        config = self._ensure_aux_data(config)
        reader.resolved_config = SimpleNamespace(**config)

        read_latlon.return_value = (xr.Dataset(), None)

        ds1 = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds_meta = xr.Dataset({"meta": ("x", [9, 9, 9])})
        ds2 = xr.Dataset({"final": ("x", [7, 8, 9])})

        read_bands.return_value = ds1
        reader.mtd.attach_metadata = Mock(return_value=ds_meta)
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        reader.mtd.attach_metadata.assert_called_once_with(ds1, level="basic")
        apply_conventions.assert_called_once_with(
            ds_meta,
            layout=reader.layout,
            config=reader.resolved_config,
        )

    def test_get_extension(self):
        self.assertEqual(SLSTRL1Reader.get_extension(), ".SEN3")

    def test_open_dataset_unknown_meas_raises(self):
        reader = self._make_reader_minimal()
        # request an unknown meas string
        config = {
            "vars_sel": {"meas": "unknown", "aux": [], "mask": []},
            "subset": None,
            "read_params": {
                "metadata_level": None,
                "use_chunks": False,
                "chunks": None,
            },
        }
        reader.resolved_config = SimpleNamespace(**config)
        with self.assertRaises(ValueError):
            reader.open_dataset()

    def _make_reader_minimal(self, product_metadata=None):
        reader = object.__new__(SLSTRL1Reader)
        reader.path = None
        reader.layout = Mock(name="layout")
        reader.layout.proc_version = 3
        reader.mtd = Mock(name="mtd")
        if product_metadata is None:
            product_metadata = {"crs_code": 4326}
        reader.mtd.product_metadata = product_metadata
        reader.mtd.attach_metadata = Mock()
        return reader


class TestSLSTRReaderData(unittest.TestCase):
    def test_real_sen3_skipped(self):
        if SEN3_PATH is None:
            self.skipTest("No SEN3_PATH configured for integration test")

        sen3_path = Path(SEN3_PATH)
        if not sen3_path.exists():
            self.skipTest(f"Test SEN3 product not found at {sen3_path}")

        vars_sel = {"meas": "all", "aux": None}
        subset = {"roi": None, "roi_crs": None}

        reader = SLSTRL1Reader(
            sen3_path,
            vars_sel=vars_sel,
            subset=subset,
            read_params={},
        )

        ds = reader.open_dataset()
        self.assertIsInstance(ds, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
