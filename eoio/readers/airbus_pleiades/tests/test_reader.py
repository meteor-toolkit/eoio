"""
eoio.readers.airbus_pleiades.tests.test_reader - unit tests for eoio.readers.airbus_pleiades.reader
"""

import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import xarray as xr
from eoio.readers.airbus_pleiades.reader import AirbusPleiadesReader


class TestAirbusPleiadesReader(unittest.TestCase):
    def setUp(self) -> None:
        self.prod_path = "/tmp/fake_pleiades"

    def _make_reader_minimal(self, product_metadata=None):
        """
        Construct a reader instance without calling the real __init__.

        """
        reader = object.__new__(AirbusPleiadesReader)

        # Minimal attributes used by resolve_subset/open_dataset
        reader.path = None
        reader.layout = Mock(name="layout")
        reader.mtd = Mock(name="mtd")

        # New reader.open_dataset expects self.config.vars_sel["aux"] to exist
        reader.config = SimpleNamespace(vars_sel={"aux": None})

        if product_metadata is None:
            product_metadata = {"geospatial_bounds_crs": "32630"}

        reader.mtd.product_metadata = product_metadata

        return reader

    @patch("eoio.readers.airbus_pleiades.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.airbus_pleiades.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_empty(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.airbus_pleiades.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_roi_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({"roi": None, "roi_crs": 4326}))
        ROISubsetResolver.assert_not_called()

    def test_resolve_subset_raises_if_image_crs_missing(self):
        reader = self._make_reader_minimal(product_metadata={"geospatial_bounds_crs": None})

        with self.assertRaises(ValueError) as val_err:
            reader.resolve_subset({"roi": [0, 0, 1, 1], "roi_crs": 4326})

        self.assertIn("CRS", str(val_err.exception))

    @patch("eoio.readers.airbus_pleiades.reader.ROISubsetResolver")
    def test_resolve_subset_calls_resolver_and_returns_result(self, ROISubsetResolver):
        reader = self._make_reader_minimal(product_metadata={"geospatial_bounds_crs": "32630"})

        expected = Mock(name="resolved_roi_subset")
        ROISubsetResolver.return_value.run.return_value = expected

        subset = {"roi": [0, 0, 1, 1], "roi_crs": 4326}
        out = reader.resolve_subset(subset)

        self.assertIs(out, expected)

        ROISubsetResolver.assert_called_once_with(
            roi=[0, 0, 1, 1],
            roi_crs_epsg=4326,
            image_crs_epsg="32630",
            image_bounds=None,
        )
        ROISubsetResolver.return_value.run.assert_called_once_with()

    @patch("eoio.readers.airbus_pleiades.reader.apply_conventions")
    @patch("eoio.readers.airbus_pleiades.reader.read_bands_into_dataset")
    def test_open_dataset_reads_bands_when_meas_requested(
        self,
        read_bands_into_dataset,
        apply_conventions,
    ):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1", "B2"], "aux": None},
            subset=Mock(name="roi_subset"),
            read_params={"metadata_level": None, "use_chunks": True},
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
        self.assertEqual(kwargs["meas"], ["B1", "B2"])
        self.assertIs(kwargs["subset"], reader.resolved_config.subset)
        self.assertIs(kwargs["mtd"], reader.mtd)
        self.assertTrue(kwargs["use_chunks"])

        apply_conventions.assert_called_once_with(
            ds1,
            layout=reader.layout,
            roi_subset=reader.resolved_config.subset,
            config=reader.resolved_config,
        )

    @patch("eoio.readers.airbus_pleiades.reader.apply_conventions")
    @patch("eoio.readers.airbus_pleiades.reader.read_bands_into_dataset")
    def test_open_dataset_does_not_read_bands_when_meas_none(
        self,
        read_bands_into_dataset,
        apply_conventions,
    ):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None, "aux": None},
            subset=None,
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        ds2 = xr.Dataset({"bar": ("x", [4, 5, 6])})
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands_into_dataset.assert_not_called()
        apply_conventions.assert_called_once()
        args, _ = apply_conventions.call_args
        self.assertIsInstance(args[0], xr.Dataset)

    @patch("eoio.readers.airbus_pleiades.reader.apply_conventions")
    @patch("eoio.readers.airbus_pleiades.reader.read_bands_into_dataset")
    def test_open_dataset_attaches_metadata_when_level_all_or_basic(
        self,
        read_bands_into_dataset,
        apply_conventions,
    ):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1"], "aux": None},
            subset=None,
            read_params={"metadata_level": "basic", "use_chunks": False},
        )

        ds1 = xr.Dataset({"B1": ("data", [1, 2, 3])})
        ds_meta = xr.Dataset({"B1": ("data", [4, 5, 6])})
        ds_meta.attrs["mtd"] = {"test_mtd_key": "test_mtd_value"}
        ds2 = xr.Dataset({"B1": ("data", [7, 8, 9])})
        ds2.attrs["mtd"] = {"test_mtd_key": "test_mtd_value"}
        ds2.attrs["conventions"] = {"eoio:convention_key": "eoio:convention_value"}

        read_bands_into_dataset.return_value = ds1
        reader.mtd.attach_metadata = Mock(return_value=ds_meta)
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        reader.mtd.attach_metadata.assert_called_once_with(ds1, level="basic")
        apply_conventions.assert_called_once_with(
            ds_meta,
            layout=reader.layout,
            roi_subset=None,
            config=reader.resolved_config,
        )

    @patch("eoio.readers.airbus_pleiades.reader.read_aux")
    @patch("eoio.readers.airbus_pleiades.reader.apply_conventions")
    @patch("eoio.readers.airbus_pleiades.reader.read_bands_into_dataset")
    def test_open_dataset_calls_read_aux_when_aux_requested(
        self,
        read_bands_into_dataset,
        apply_conventions,
        read_aux,
    ):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1"], "aux": {"observation_geometry": True}},
            subset=None,
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        ds1 = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds_aux = xr.Dataset({"aux": ("x", [4, 5, 6])})
        ds2 = xr.Dataset({"bar": ("x", [7, 8, 9])})

        read_bands_into_dataset.return_value = ds1
        read_aux.return_value = ds_aux
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_aux.assert_called_once()
        args, kwargs = read_aux.call_args
        self.assertIs(kwargs["ds"], ds1)
        self.assertEqual(kwargs["aux"], {"observation_geometry": True})
        self.assertIs(kwargs["mtd"], reader.mtd)


if __name__ == "__main__":
    unittest.main()
