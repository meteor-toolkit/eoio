"""eoio.readers.constellr_skybee.tests.test_reader - unit tests for eoio.readers.constellr_skybee.reader"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import xarray as xr

from eoio.readers.constellr_skybee.reader import ConstellrSkybeeReader


class TestConstellrSkybeeReader(unittest.TestCase):
    def _make_reader_minimal(self, product_metadata=None):
        reader = object.__new__(ConstellrSkybeeReader)
        reader.path = None
        reader.layout = Mock(name="layout")
        reader.mtd = Mock(name="mtd")
        reader.mtd.product_metadata = product_metadata or {"geospatial_bounds_crs": "EPSG:4326"}
        reader.config = SimpleNamespace(vars_sel={"aux": None})
        return reader

    @patch("eoio.readers.constellr_skybee.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_none(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.constellr_skybee.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_roi_missing(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({"roi": None, "roi_crs": 4326}))
        ROISubsetResolver.assert_not_called()

    def test_resolve_subset_raises_if_image_crs_missing(self):
        reader = self._make_reader_minimal({"geospatial_bounds_crs": None})

        with self.assertRaises(ValueError):
            reader.resolve_subset({"roi": [0, 0, 1, 1], "roi_crs": 4326})

    @patch("eoio.readers.constellr_skybee.reader.ROISubsetResolver")
    def test_resolve_subset_calls_resolver(self, ROISubsetResolver):
        reader = self._make_reader_minimal({"geospatial_bounds_crs": "EPSG:4326"})
        expected = Mock(name="resolved_roi_subset")
        ROISubsetResolver.return_value.run.return_value = expected

        out = reader.resolve_subset({"roi": [0, 0, 1, 1], "roi_crs": 4326})

        self.assertIs(out, expected)
        ROISubsetResolver.assert_called_once_with(
            roi=[0, 0, 1, 1],
            roi_crs_epsg=4326,
            image_crs_epsg="EPSG:4326",
            image_bounds=None,
        )

    @patch("eoio.readers.constellr_skybee.reader.apply_conventions")
    @patch("eoio.readers.constellr_skybee.reader.read_bands_into_dataset")
    def test_open_dataset_reads_bands_when_meas_requested(self, read_bands_into_dataset, apply_conventions):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1", "B2"], "aux": None},
            subset=Mock(name="roi_subset"),
            read_params={"metadata_level": None, "use_chunks": True, "chunks": None},
        )

        ds1 = xr.Dataset({"foo": ("x", [1, 2])})
        ds2 = xr.Dataset({"bar": ("x", [3, 4])})
        read_bands_into_dataset.return_value = ds1
        apply_conventions.return_value = ds2

        out = reader.open_dataset()

        self.assertIs(out, ds2)
        read_bands_into_dataset.assert_called_once()
        kwargs = read_bands_into_dataset.call_args.kwargs
        self.assertEqual(kwargs["meas"], ["B1", "B2"])
        self.assertTrue(kwargs["use_chunks"])
        apply_conventions.assert_called_once_with(
            ds1,
            layout=reader.layout,
            roi_subset=reader.resolved_config.subset,
            config=reader.resolved_config,
        )

    @patch("eoio.readers.constellr_skybee.reader.read_aux")
    @patch("eoio.readers.constellr_skybee.reader.apply_conventions")
    @patch("eoio.readers.constellr_skybee.reader.read_bands_into_dataset")
    def test_open_dataset_calls_read_aux_when_aux_requested(self, read_bands_into_dataset, apply_conventions, read_aux):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1"], "aux": ["observation_geometry"]},
            subset=None,
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        ds1 = xr.Dataset({"B1": ("x", [1, 2])})
        ds_aux = xr.Dataset({"obs": ("x", [3, 4])})
        ds2 = xr.Dataset({"B1": ("x", [5, 6])})

        read_bands_into_dataset.return_value = ds1
        read_aux.return_value = ds_aux
        apply_conventions.return_value = ds2

        out = reader.open_dataset()

        self.assertIs(out, ds2)
        read_aux.assert_called_once_with(ds=ds1, aux=["observation_geometry"], mtd=reader.mtd)

    def test_get_extension(self):
        self.assertEqual(ConstellrSkybeeReader.get_extension(), "")


if __name__ == "__main__":
    unittest.main()
