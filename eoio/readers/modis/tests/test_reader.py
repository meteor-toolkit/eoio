"""eoio.readers.modis.tests.test_reader - unit tests for eoio.readers.modis.reader"""

import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import xarray as xr
from eoio.readers.modis.reader import MODISReader


class TestMODISReader(unittest.TestCase):
    def _make_reader_minimal(self, product_metadata=None):
        """
        Construct a reader instance without calling the real __init__.
        """
        reader = object.__new__(MODISReader)

        reader.path = "/tmp/fake_MOD02HDF.A2020001.0000.061.2020002121530.hdf"
        reader.layout = Mock(name="layout")
        reader.mtd = Mock(name="mtd")
        reader.config = Mock(name="config")
        reader.config.vars_sel = {"meas": [], "aux": []}

        if product_metadata:
            reader.mtd.basic_metadata = product_metadata
        else:
            reader.mtd.basic_metadata = {"horizontal_cs_code": 4326}

        return reader

    # -------------------------
    # resolve_subset() tests
    # -------------------------

    @patch("eoio.readers.modis.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_none(self, ROISubsetResolver):
        """Test that resolve_subset returns None when subset is None."""
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.modis.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_subset_is_empty(self, ROISubsetResolver):
        """Test that resolve_subset returns None when subset is empty dict."""
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.modis.reader.ROISubsetResolver")
    def test_resolve_subset_returns_none_when_roi_is_none(self, ROISubsetResolver):
        """Test that resolve_subset returns None when roi is None."""
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset({"roi": None, "roi_crs": 4326}))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.modis.reader.ROISubsetResolver")
    def test_resolve_subset_calls_resolver(self, ROISubsetResolver):
        """Test that resolve_subset calls ROISubsetResolver with correct parameters."""
        reader = self._make_reader_minimal(product_metadata={"horizontal_cs_code": 4326})

        expected_result = Mock(name="resolved_roi_subset")
        resolver_instance = Mock()
        resolver_instance.run.return_value = expected_result
        ROISubsetResolver.return_value = resolver_instance

        subset = {"roi": [0, 0, 1, 1], "roi_crs": 4326}
        result = reader.resolve_subset(subset)

        self.assertIs(result, expected_result)
        ROISubsetResolver.assert_called_once_with(
            roi=[0, 0, 1, 1],
            roi_crs_epsg=4326,
            image_crs_epsg=4326,
            image_bounds=None,
        )
        resolver_instance.run.assert_called_once_with()

    # -------------------------
    # open_dataset() tests
    # -------------------------

    @patch("eoio.readers.modis.reader.xr.open_dataset")
    @patch("eoio.readers.modis.reader.read_bands_into_dataset")
    def test_open_dataset_reads_bands_when_meas_vars_requested(
        self,
        mock_read_bands,
        mock_open_xr,
    ):
        """Test that open_dataset reads bands when measurement variables are requested."""
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["Band 1", "Band 2"]},
            subset=Mock(name="roi_subset"),
            read_params={
                "metadata_level": None,
                "use_chunks": False,
                "preferred_resolution": None,
            },
        )
        reader.layout.collection = "MOD03"
        reader.layout.geolocation_path.return_value = "/tmp/MOD03_geoloc.hdf"
        mock_open_xr.return_value = xr.Dataset()

        ds1 = xr.Dataset({"Band 1": (("y", "x"), [[1]])})
        mock_read_bands.return_value = ds1

        _ = reader.open_dataset()

        # Should call read_bands_into_dataset
        mock_read_bands.assert_called_once()
        call_kwargs = mock_read_bands.call_args.kwargs
        self.assertEqual(call_kwargs["meas_vars"], ["Band 1", "Band 2"])
        self.assertIs(call_kwargs["subset"], reader.resolved_config.subset)
        self.assertIs(call_kwargs["mtd"], reader.mtd)

    @patch("eoio.readers.modis.reader.xr.open_dataset")
    @patch("eoio.readers.modis.reader.read_bands_into_dataset")
    def test_open_dataset_does_not_read_bands_when_meas_vars_none(
        self,
        mock_read_bands,
        mock_open_xr,
    ):
        """Test that open_dataset does not read bands when meas_vars is None."""
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None},
            subset=None,
            read_params={
                "metadata_level": None,
                "use_chunks": False,
                "chunks": None,
                "preferred_resolution": None,
            },
        )
        reader.layout.collection = "MOD03"
        reader.layout.geolocation_path.return_value = "/tmp/MOD03_geoloc.hdf"
        mock_open_xr.return_value = xr.Dataset()

        reader.open_dataset()

        # Should not call read_bands_into_dataset
        mock_read_bands.assert_not_called()

    @patch("eoio.readers.modis.reader.xr.open_dataset")
    def test_open_dataset_handles_preferred_resolution_as_string(
        self,
        mock_open_xr,
    ):
        """Test that preferred_resolution string is converted to int."""
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None},
            subset=None,
            read_params={
                "metadata_level": None,
                "use_chunks": False,
                "chunks": None,
                "preferred_resolution": "500",  # String
            },
        )

        reader.layout.geolocation_path.return_value = "/tmp/MOD03_geoloc.hdf"
        reader.layout.collection = "MOD03"
        mock_open_xr.return_value = xr.Dataset()

        # Should not raise an error
        reader.open_dataset()

    @patch("eoio.readers.modis.reader.xr.open_dataset")
    def test_open_dataset_handles_preferred_resolution_none_string(
        self,
        mock_open_xr,
    ):
        """Test that preferred_resolution 'none' string is converted to None."""
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None},
            subset=None,
            read_params={
                "metadata_level": None,
                "use_chunks": False,
                "chunks": None,
                "preferred_resolution": "none",  # String 'none'
            },
        )
        reader.layout.collection = "MOD03"
        reader.layout.geolocation_path.return_value = "/tmp/MOD03_geoloc.hdf"
        mock_open_xr.return_value = xr.Dataset()

        # Should not raise an error
        reader.open_dataset()

    @patch("eoio.readers.modis.reader.xr.open_dataset")
    def test_open_dataset_opens_geolocation_dataset(
        self,
        mock_open_xr,
    ):
        """Test that open_dataset opens the geolocation dataset."""
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": None},
            subset=None,
            read_params={
                "metadata_level": None,
                "use_chunks": False,
                "chunks": None,
                "preferred_resolution": None,
            },
        )
        reader.layout.collection = "MOD03"
        reader.layout.geolocation_path.return_value = "/tmp/MOD03.hdf"
        mock_open_xr.return_value = xr.Dataset()

        reader.open_dataset()

        # Should call xr.open_dataset with geolocation path
        mock_open_xr.assert_called_once()
        call_args = mock_open_xr.call_args
        self.assertEqual(call_args[0][0], "/tmp/MOD03.hdf")


if __name__ == "__main__":
    unittest.main()
