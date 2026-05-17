"""
eoio.readers.airbus_pleiades.tests.test_data_io - unit tests for eoio.readers.airbus_pleiades.data_io
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import xarray as xr
from eoio.readers.airbus_pleiades.data_io import (
    read_bands_into_dataset,
    get_var_geometry,
    get_var_resolution,
    reorder_bands,
)


class TestGetVarGeometry(unittest.TestCase):
    """Unit tests for get_var_geometry function."""

    def test_get_var_geometry(self):
        """Test that get_var_geometry returns '2'."""
        result = get_var_geometry()
        self.assertEqual(result, "2")


class TestGetVarResolution(unittest.TestCase):
    """Unit tests for get_var_resolution function."""

    def test_get_var_resolution(self):
        """Test that get_var_resolution returns '2m'."""
        result = get_var_resolution()
        self.assertEqual(result, "2m")


class TestReorderBands(unittest.TestCase):
    """Unit tests for reorder_bands function."""

    def setUp(self):
        """Create a test dataset with bands."""
        self.ds = xr.Dataset(
            {
                "B1": xr.DataArray(
                    np.array([[1, 2], [3, 4]]),
                    dims=["y", "x"],
                    attrs={"long_name": "Radiance values in band B1"},
                ),
                "B2": xr.DataArray(
                    np.array([[5, 6], [7, 8]]),
                    dims=["y", "x"],
                    attrs={"long_name": "Radiance values in band B2"},
                ),
                "B3": xr.DataArray(
                    np.array([[9, 10], [11, 12]]),
                    dims=["y", "x"],
                    attrs={"long_name": "Radiance values in band B3"},
                ),
                "B4": xr.DataArray(
                    np.array([[13, 14], [15, 16]]),
                    dims=["y", "x"],
                    attrs={"long_name": "Radiance values in band B4"},
                ),
            }
        )

    def test_reorder_bands_swaps_b1_and_b3(self):
        """Test that reorder_bands swaps B1 and B3 data. & updates attrs correctly."""
        original_b1_data = self.ds["B1"].values.copy()
        original_b3_data = self.ds["B3"].values.copy()

        result = reorder_bands(self.ds)

        # After swap: original B1 is now B3, original B3 is now B1
        np.testing.assert_array_equal(result["B3"].values, original_b1_data)
        np.testing.assert_array_equal(result["B1"].values, original_b3_data)

        # B3 (originally B1) should have B3 references in attrs
        self.assertIn("B3", result["B3"].attrs["long_name"])
        # B1 (originally B3) should have B1 references in attrs
        self.assertIn("B1", result["B1"].attrs["long_name"])

    def test_reorder_bands_returns_dataset(self):
        """Test that reorder_bands returns an xarray Dataset."""
        result = reorder_bands(self.ds)
        self.assertIsInstance(result, xr.Dataset)

    def test_reorder_bands_preserves_b2_and_b4(self):
        """Test that B2 and B4 variables are unchanged."""
        original_b2_data = self.ds["B2"].values.copy()
        original_b4_data = self.ds["B4"].values.copy()

        result = reorder_bands(self.ds)

        np.testing.assert_array_equal(result["B2"].values, original_b2_data)
        np.testing.assert_array_equal(result["B4"].values, original_b4_data)


class TestReadBandsIntoDataset(unittest.TestCase):
    """Unit tests for read_bands_into_dataset function."""

    def setUp(self):
        """Set up test fixtures."""
        self.ds = xr.Dataset()
        self.layout = Mock()
        self.layout.image_file = Mock(return_value="/path/to/image.tif")
        self.mtd = Mock()
        self.mtd.variable_product_metadata = Mock(return_value={"band_gain": 1, "band_bias": 1})
        self.meas = ["B1"]
        self.subset_none = None

    def _make_mock_dataarray(self, values: np.ndarray) -> xr.DataArray:
        """Create a mocked DataArray with rioxarray structure."""
        da = xr.DataArray(
            values,
            dims=("band", "y", "x"),
            coords={
                "band": [1],
                "y": np.arange(values.shape[1]),
                "x": np.arange(values.shape[2]),
            },
        )

        return da

    @patch("eoio.readers.airbus_pleiades.data_io.reorder_bands")
    @patch("eoio.readers.airbus_pleiades.data_io.convert_xy")
    @patch("eoio.readers.airbus_pleiades.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_returns_dataset(self, mock_lazy_rioxarray, mock_convert_xy, mock_reorder):
        """Test that read_bands_into_dataset returns an xarray Dataset."""
        # Setup mocks
        mock_rxr = Mock()

        # Create mock data for each band
        band_data = self._make_mock_dataarray(np.random.randint(1, 100, size=(1, 10, 10)))
        mock_rxr.open_rasterio.return_value = band_data

        mock_lazy_rioxarray.return_value = mock_rxr

        # Mock subset
        subset = Mock()
        subset.geometries = None
        subset.clip_box = (100, 0, 100, 0)

        # Mock coordinate conversion
        mock_convert_xy.return_value = (np.ndarray((10, 10)), np.ndarray((10, 10)))

        # Add proj_crs to mocked metadata
        self.mtd.product_metadata = {"geospatial_bounds_crs": "test_crs"}

        # Mock reorder_bands to return input
        mock_reorder.side_effect = lambda ds: ds

        result = read_bands_into_dataset(
            ds=self.ds,
            layout=self.layout,
            meas=self.meas,
            subset=subset,
            mtd=self.mtd,
            use_chunks=False,
            chunks=None,
        )

        self.assertIsInstance(result, xr.Dataset)
        mock_reorder.assert_called_once()

    @patch("eoio.readers.airbus_pleiades.data_io.reorder_bands")
    @patch("eoio.readers.airbus_pleiades.data_io.convert_xy")
    @patch("eoio.readers.airbus_pleiades.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_updates_zero_as_nodata(self, mock_lazy_rioxarray, mock_convert_xy, mock_reorder):
        """Test that read_bands_into_dataset updates zero values as NaN."""
        # Setup mocks
        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr

        # Create data with zeros
        band_data = self._make_mock_dataarray(np.array([[[0, 10], [20, 0]]], dtype=np.uint16))
        mock_rxr.open_rasterio.return_value = band_data

        subset = Mock()
        subset.geometries = None
        subset.clip_box = (100, 0, 100, 0)

        mock_convert_xy.return_value = (
            np.array([[0, 10], [20, 0]]),
            np.array([[0, 10], [20, 0]]),
        )

        # Add proj_crs to mocked metadata
        self.mtd.product_metadata = {"geospatial_bounds_crs": "test_crs"}

        # Mock reorder_bands to return input
        mock_reorder.side_effect = lambda ds: ds

        result = read_bands_into_dataset(
            ds=self.ds,
            layout=self.layout,
            meas=["B2"],
            subset=subset,
            mtd=self.mtd,
            use_chunks=False,
            chunks=None,
            mask_zero_as_nodata=True,
        )

        # Check that zeros were masked as NaN
        self.assertTrue(np.isnan(result["B2"].values[0, 0]))
        self.assertTrue(np.isnan(result["B2"].values[1, 1]))
        self.assertEqual(result["B2"].values[0, 1], 11)
        self.assertEqual(result["B2"].values[1, 0], 21)

    @patch("eoio.readers.airbus_pleiades.data_io.suggest_raster_chunks")
    @patch("eoio.readers.airbus_pleiades.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_uses_chunks_when_requested(self, mock_lazy_rioxarray, mock_suggest_chunks):
        """Test that read_bands_into_dataset uses chunks when use_chunks=True."""
        mock_suggest_chunks.return_value = {"x": 256, "y": 256}

        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_rxr.open_rasterio.return_value = Mock()

        subset = Mock()
        subset.geometries = None
        subset.clip_box = (100, 0, 100, 0)

        with (
            patch("eoio.readers.airbus_pleiades.data_io.convert_xy"),
            patch("eoio.readers.airbus_pleiades.data_io.reorder_bands"),
        ):
            try:
                read_bands_into_dataset(
                    ds=self.ds,
                    layout=self.layout,
                    meas=["B1"],
                    subset=subset,
                    mtd=self.mtd,
                    use_chunks=True,
                    chunks=None,
                )
            except Exception:
                pass  # Test is only checking that suggest_raster_chunks was called, ignore the rest

        mock_suggest_chunks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
