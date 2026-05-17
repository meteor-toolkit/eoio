"""eoio.readers.sentinel3_olci.tests.test_reader - unit tests for eoio.readers.sentinel3_olci.reader

These tests mirror the style used by the Sentinel-2 reader tests. They focus on
`resolve_subset` and `open_dataset` behaviour while keeping the heavy parts
mocked so tests run quickly.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import xarray as xr
from eoio.readers.sentinel3_olci.reader import OLCIL1Reader

# Optional: path to a real SEN3 product for integration testing. Skip if absent.
SEN3_PATH = r"T:\ECO\EOServer\data\unittest_datasets\S3OLCI\S3A_OL_1_EFR____20250712T110239_20250712T110539_20250713T115701_0179_128_094_1980_PS1_O_NT_004.SEN3"


class TestOLCIL1Reader(unittest.TestCase):
    def setUp(self) -> None:
        self.sen3_path = "/tmp/fake.SEN3"

    def _ensure_aux_data(self, config):
        """Ensure aux and mask keys exist in vars_sel."""
        if "aux" not in config["vars_sel"]:
            config["vars_sel"]["aux"] = []
        if "mask" not in config["vars_sel"]:
            config["vars_sel"]["mask"] = []
        return config

    # -------------------------
    # resolve_subset() tests
    # -------------------------

    @patch("eoio.readers.sentinel3_olci.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel3_olci.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_empty(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel3_olci.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_roi_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({"roi": None, "roi_crs": 4326}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel3_olci.reader.ROISubsetResolver")
    def test_resolve_subset_uses_default_image_crs_when_missing(self, ROISubsetResolver):
        """When product CRS is None, resolve_subset should raise ValueError."""
        reader = self._make_reader_minimal(product_metadata={"crs_code": None})

        expected = Mock(name="resolved_roi_subset")
        ROISubsetResolver.return_value.run.return_value = expected

        subset = {"roi": [0, 0, 1, 1], "roi_crs": 4326}
        # When crs_code is None, the code raises ValueError
        with self.assertRaises(ValueError):
            reader.resolve_subset(subset)

    @patch("eoio.readers.sentinel3_olci.reader.ROISubsetResolver")
    def test_resolve_subset_calls_resolver_and_returns_result(self, ROISubsetResolver):
        reader = self._make_reader_minimal(product_metadata={"crs_code": 32630})

        expected = Mock(name="resolved_roi_subset")
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

    # -------------------------
    # open_dataset() tests
    # -------------------------

    @patch("eoio.readers.sentinel3_olci.reader.read_lat_lon_coordinates")
    @patch("eoio.readers.sentinel3_olci.reader.apply_conventions")
    @patch("eoio.readers.sentinel3_olci.reader.read_bands_into_dataset")
    def test_open_dataset_reads_bands_when_meas_vars_requested(
        self,
        read_bands_into_dataset,
        apply_conventions,
        read_lat_lon_coordinates,
    ):
        reader = self._make_reader_minimal()
        config = {
            "vars_sel": {"meas": ["Oa01", "Oa02"]},
            "subset": Mock(name="roi_subset"),
            "read_params": {
                "metadata_level": None,
                "use_chunks": True,
                "chunks": {"x": 256, "y": 256},
            },
        }
        config = self._ensure_aux_data(config)
        reader.resolved_config = SimpleNamespace(**config)

        ds1 = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds2 = xr.Dataset({"bar": ("x", [4, 5, 6])})

        # Prevent actual rasterio/open calls by stubbing lat/lon reader
        read_lat_lon_coordinates.return_value = (xr.Dataset(), None)

        read_bands_into_dataset.return_value = ds1
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands_into_dataset.assert_called_once()
        _, kwargs = read_bands_into_dataset.call_args
        self.assertIs(kwargs["layout"], reader.layout)
        self.assertEqual(kwargs["meas"], ["Oa01", "Oa02"])
        self.assertIs(kwargs["subset"], reader.resolved_config.subset)
        self.assertIs(kwargs["mtd"], reader.mtd)
        self.assertTrue(kwargs["use_chunks"])
        self.assertEqual(kwargs["chunks"], {"x": 256, "y": 256})

        apply_conventions.assert_called_once_with(
            ds1,
            layout=reader.layout,
            config=reader.resolved_config,
        )

    @patch("eoio.readers.sentinel3_olci.reader.read_lat_lon_coordinates")
    @patch("eoio.readers.sentinel3_olci.reader.apply_conventions")
    @patch("eoio.readers.sentinel3_olci.reader.read_bands_into_dataset")
    def test_open_dataset_does_not_read_bands_when_meas_vars_none(
        self,
        read_bands_into_dataset,
        apply_conventions,
        read_lat_lon_coordinates,
    ):
        reader = self._make_reader_minimal()
        config = {
            "vars_sel": {"meas": None},
            "subset": None,
            "read_params": {
                "metadata_level": None,
                "use_chunks": False,
                "chunks": None,
            },
        }
        config = self._ensure_aux_data(config)
        reader.resolved_config = SimpleNamespace(**config)

        read_lat_lon_coordinates.return_value = (xr.Dataset(), None)

        ds2 = xr.Dataset({"bar": ("x", [4, 5, 6])})
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands_into_dataset.assert_not_called()
        apply_conventions.assert_called_once()
        args, _ = apply_conventions.call_args
        self.assertIsInstance(args[0], xr.Dataset)

    @patch("eoio.readers.sentinel3_olci.reader.read_lat_lon_coordinates")
    @patch("eoio.readers.sentinel3_olci.reader.apply_conventions")
    @patch("eoio.readers.sentinel3_olci.reader.read_bands_into_dataset")
    def test_open_dataset_attaches_metadata_when_level_all_or_basic(
        self,
        read_bands_into_dataset,
        apply_conventions,
        read_lat_lon_coordinates,
    ):
        reader = self._make_reader_minimal()
        config = {
            "vars_sel": {"meas": ["Oa01"]},
            "subset": None,
            "read_params": {
                "metadata_level": "basic",
                "use_chunks": False,
                "chunks": None,
            },
        }
        config = self._ensure_aux_data(config)
        reader.resolved_config = SimpleNamespace(**config)

        read_lat_lon_coordinates.return_value = (xr.Dataset(), None)

        ds1 = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds_meta = xr.Dataset({"meta": ("x", [9, 9, 9])})
        ds2 = xr.Dataset({"final": ("x", [7, 8, 9])})

        read_bands_into_dataset.return_value = ds1
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

    # -------------------------
    # Helpers
    # -------------------------

    def _make_reader_minimal(self, product_metadata=None):
        """
        Construct a reader instance without calling the real __init__.

        This keeps tests fast and avoids needing a real SEN3 on disk.
        """
        reader = object.__new__(OLCIL1Reader)

        # Minimal attributes used by resolve_subset/open_dataset
        reader.path = None
        reader.layout = Mock(name="layout")
        reader.layout.proc_version = 3
        reader.mtd = Mock(name="mtd")

        if product_metadata is None:
            product_metadata = {"crs_code": 32630}

        reader.mtd.product_metadata = product_metadata

        # Mock attach_metadata method
        reader.mtd.attach_metadata = Mock()

        return reader


class TestOLCIL1ReaderData(unittest.TestCase):
    def test_real_sen3(self):
        if SEN3_PATH is None:
            self.skipTest("No SEN3_PATH configured for integration test")

        sen3_path = Path(SEN3_PATH)
        if not sen3_path.exists():
            self.skipTest(f"Test SEN3 product not found at {sen3_path}")

        vars_sel = {"meas": "rgb", "aux": None}
        subset = {"roi": None, "roi_crs": None}

        reader = OLCIL1Reader(
            sen3_path,
            vars_sel=vars_sel,
            subset=subset,
            read_params={},
        )

        ds = reader.open_dataset()
        self.assertIsInstance(ds, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
