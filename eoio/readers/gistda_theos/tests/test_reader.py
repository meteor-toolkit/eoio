"""eoio.readers.gistda_theos.tests.test_reader - unit tests for eoio.readers.gistda_theos.reader"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import xarray as xr

from eoio.readers.gistda_theos.reader import AUX_OPTIONS, GISTDATHEOSReader


class TestGistdaTheosReader(unittest.TestCase):
    """Unit tests for GistdaTheosReader class."""

    def _make_reader_minimal(self):
        reader = object.__new__(GISTDATHEOSReader)
        reader.path = None
        reader.layout = Mock(name="layout")
        reader.mtd = Mock(name="mtd")
        reader.mtd.product_metadata = {"geospatial_bounds_crs": "EPSG:4326"}
        return reader

    @patch("eoio.readers.gistda_theos.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_none(self, subset_resolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        subset_resolver.assert_not_called()

    @patch("eoio.readers.gistda_theos.reader.ROISubsetResolver")
    def test_resolve_subset_raises_if_image_crs_missing(self, subset_resolver):
        reader = self._make_reader_minimal()
        reader.mtd.product_metadata = {"geospatial_bounds_crs": None}
        with self.assertRaises(ValueError):
            reader.resolve_subset({"roi": [0, 0, 1, 1], "roi_crs": 4326})

    @patch("eoio.readers.gistda_theos.reader.ROISubsetResolver")
    def test_resolve_subset_calls_resolver(self, subset_resolver):
        reader = self._make_reader_minimal()
        subset_resolver.return_value.run.return_value = Mock(name="resolved")

        result = reader.resolve_subset({"roi": [0, 0, 1, 1], "roi_crs": 4326})

        self.assertIsNotNone(result)
        subset_resolver.assert_called_once_with(
            roi=[0, 0, 1, 1],
            roi_crs_epsg=4326,
            image_crs_epsg="EPSG:4326",
            image_bounds=None,
        )

    def test_open_dataset_handles_no_meas_and_aux(self, subset_resolver=None):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None, "aux": None},
            subset=None,
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        # When no meas requested, should still call apply_conventions and return its result
        with patch("eoio.readers.gistda_theos.reader.apply_conventions") as apply_conventions:
            ds_ret = xr.Dataset()
            apply_conventions.return_value = ds_ret
            out = reader.open_dataset()
            self.assertIs(out, ds_ret)

    @patch("eoio.readers.gistda_theos.reader.apply_conventions")
    @patch("eoio.readers.gistda_theos.reader.read_bands_into_dataset")
    def test_open_dataset_reads_bands_when_meas_requested(self, read_bands_into_dataset, apply_conventions):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1", "B2"], "aux": None},
            subset=Mock(name="roi_subset"),
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        ds1 = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds2 = xr.Dataset({"bar": ("x", [4, 5, 6])})
        read_bands_into_dataset.return_value = ds1
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)
        read_bands_into_dataset.assert_called_once()
        self.assertEqual(read_bands_into_dataset.call_args.kwargs["meas"], ["B1", "B2"])

    @patch("eoio.readers.gistda_theos.reader.apply_conventions")
    @patch("eoio.readers.gistda_theos.reader.read_bands_into_dataset")
    def test_open_dataset_attaches_metadata_when_level_all_or_basic(self, read_bands_into_dataset, apply_conventions):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1"], "aux": None},
            subset=None,
            read_params={"metadata_level": "basic", "use_chunks": False, "chunks": None},
        )

        ds1 = xr.Dataset({"B1": ("data", [1, 2, 3])})
        ds_meta = xr.Dataset({"B1": ("data", [4, 5, 6])})
        reader.mtd.attach_metadata = Mock(return_value=ds_meta)
        read_bands_into_dataset.return_value = ds1
        apply_conventions.return_value = ds_meta

        out = reader.open_dataset()
        self.assertIs(out, ds_meta)
        reader.mtd.attach_metadata.assert_called_once_with(ds1, level="basic")

    def test_open_dataset_metadata_level_true_converted_to_all(self):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None, "aux": None},
            subset=None,
            read_params={"metadata_level": True, "use_chunks": False, "chunks": None},
        )

        # attach_metadata should be called with 'all'
        reader.mtd.attach_metadata = Mock(return_value=xr.Dataset())
        with patch("eoio.readers.gistda_theos.reader.apply_conventions") as apply_conventions:
            apply_conventions.return_value = xr.Dataset()
            reader.open_dataset()

        reader.mtd.attach_metadata.assert_called_once()

    def test_open_dataset_adds_aux_when_requested(self):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None, "aux": ["observation_geometry"]},
            subset=None,
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        # patch read_aux and apply_conventions
        with (
            patch("eoio.readers.gistda_theos.reader.read_aux") as mock_read_aux,
            patch("eoio.readers.gistda_theos.reader.apply_conventions") as mock_apply,
        ):
            mock_read_aux.return_value = xr.Dataset()
            mock_apply.return_value = xr.Dataset()
            reader.open_dataset()

        mock_read_aux.assert_called_once()

    def test_get_extension_empty(self):
        self.assertEqual(GISTDATHEOSReader.get_extension(), "")

    def test_string_form_aux_request_is_accepted(self):
        """vars_sel={"aux": "observation_geometry"} (the documented single-string
        form -- see eoio.interface.read's own vars_sel docs) must resolve, not
        raise 'Unknown aux requested': AUX_OPTIONS being left empty made this
        unreachable even though add_angles/read_aux fully implement it."""
        reader = self._make_reader_minimal()
        reader.aux_def = {"all": AUX_OPTIONS}
        reader.config = SimpleNamespace(vars_sel={"aux": "observation_geometry"})

        self.assertEqual(reader.list_selected_aux(), ["observation_geometry"])


if __name__ == "__main__":
    unittest.main()
