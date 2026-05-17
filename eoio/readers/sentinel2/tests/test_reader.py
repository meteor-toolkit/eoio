"""
eoio.readers.sentinel2.tests.test_reader - unit tests for eoio.readers.sentinel2.reader
"""

import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import xarray as xr
from eoio.readers.sentinel2.reader import S2MSIReader

# Path to a real Sentinel-2 SAFE product for integration testing.
# If you run the tests, ensure this path points to a valid SAFE product on your system.
# Otherwise, the integration test will be skipped.
SAFE_PATH = "/Users/seh2/Library/CloudStorage/OneDrive-NationalPhysicalLaboratory/Data/archive/sentinel2/S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"


class TestS2MSIReader(unittest.TestCase):
    def setUp(self) -> None:
        self.safe_path = "/tmp/fake.SAFE"

    # -------------------------
    # resolve_subset() tests
    # -------------------------

    @patch("eoio.readers.sentinel2.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel2.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_empty(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.sentinel2.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_roi_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({"roi": None, "roi_crs": 4326}))
        ROISubsetResolver.assert_not_called()

    def test_resolve_subset_raises_if_image_crs_missing(self):
        reader = self._make_reader_minimal(product_metadata={"horizontal_cs_code": None})

        with self.assertRaises(ValueError) as ctx:
            reader.resolve_subset({"roi": [0, 0, 1, 1], "roi_crs": 4326})

        self.assertIn("horizontal_cs_code", str(ctx.exception))

    @patch("eoio.readers.sentinel2.reader.ROISubsetResolver")
    def test_resolve_subset_calls_resolver_and_returns_result(self, ROISubsetResolver):
        reader = self._make_reader_minimal(product_metadata={"horizontal_cs_code": 32630})

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

    @patch("eoio.readers.sentinel2.reader.apply_conventions")
    @patch("eoio.readers.sentinel2.reader.read_bands_into_dataset")
    def test_open_dataset_reads_bands_when_meas_vars_requested(
        self,
        read_bands_into_dataset,
        apply_conventions,
    ):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B02", "B03"]},
            subset=Mock(name="roi_subset"),
            read_params={
                "metadata_level": None,
                "use_chunks": True,
                "chunks": {"x": 256, "y": 256},
            },
        )

        ds1 = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds2 = xr.Dataset({"bar": ("x", [4, 5, 6])})

        read_bands_into_dataset.return_value = ds1
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands_into_dataset.assert_called_once()
        _, kwargs = read_bands_into_dataset.call_args
        self.assertIs(kwargs["layout"], reader.layout)
        self.assertEqual(kwargs["meas_vars"], ["B02", "B03"])
        self.assertIs(kwargs["subset"], reader.resolved_config.subset)
        self.assertIs(kwargs["mtd"], reader.mtd)
        self.assertTrue(kwargs["use_chunks"])
        self.assertEqual(kwargs["chunks"], {"x": 256, "y": 256})

        apply_conventions.assert_called_once_with(
            ds1,
            layout=reader.layout,
            config=reader.resolved_config,
        )

    @patch("eoio.readers.sentinel2.reader.apply_conventions")
    @patch("eoio.readers.sentinel2.reader.read_bands_into_dataset")
    def test_open_dataset_does_not_read_bands_when_meas_vars_none(
        self,
        read_bands_into_dataset,
        apply_conventions,
    ):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None},
            subset=None,
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        ds2 = xr.Dataset({"bar": ("x", [4, 5, 6])})
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands_into_dataset.assert_not_called()
        apply_conventions.assert_called_once()
        # It should apply conventions to an empty dataset
        args, _ = apply_conventions.call_args
        self.assertIsInstance(args[0], xr.Dataset)

    @patch("eoio.readers.sentinel2.reader.apply_conventions")
    @patch("eoio.readers.sentinel2.reader.read_bands_into_dataset")
    def test_open_dataset_attaches_metadata_when_level_all_or_basic(
        self,
        read_bands_into_dataset,
        apply_conventions,
    ):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B02"]},
            subset=None,
            read_params={
                "metadata_level": "basic",
                "use_chunks": False,
                "chunks": None,
            },
        )

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

        This keeps tests fast and avoids needing a real SAFE on disk.
        """
        reader = object.__new__(S2MSIReader)

        # Minimal attributes used by resolve_subset/open_dataset
        reader.path = None
        reader.layout = Mock(name="layout")
        reader.mtd = Mock(name="mtd")

        # New reader.open_dataset expects self.config.vars_sel["aux_data"] to exist
        reader.config = SimpleNamespace(vars_sel={"aux": None})

        if product_metadata is None:
            product_metadata = {"horizontal_cs_code": 32630}

        # The code uses: self.mtd.product_metadata.get(...)
        reader.mtd.product_metadata = product_metadata

        return reader


class TestS2MSIReaderData(unittest.TestCase):
    def test_real_safe(self):
        safe_path = Path(SAFE_PATH)
        if not safe_path.exists():
            self.skipTest(f"Test SAFE product not found at {safe_path}")

        # Minimal config that should work with your new "roi=None => no resolver" behaviour
        vars_sel = {"meas": "rgb", "aux": None}
        subset = {"roi": None, "roi_crs": None}

        reader = S2MSIReader(
            safe_path,
            vars_sel=vars_sel,
            subset=subset,
            read_params={},
        )

        ds = reader.open_dataset()
        self.assertIsInstance(ds, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
