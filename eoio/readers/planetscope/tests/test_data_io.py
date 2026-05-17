"""eoio.readers.planetscope.tests.test_data_io - tests for eoio.readers.planetscope.data_io"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from eoio.readers.planetscope.data_io import read_tif_into_dataset


class TestReadTifIntoDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Mock layout
        self.layout = MagicMock(name="layout")
        self.layout.image_file = "/tmp/fake_planetscope.tif"

        # Default metadata extractor
        self.mtd = MagicMock(name="mtd")

        # Mock subset
        self.subset = MagicMock(name="subset")
        self.subset.geometries = None
        self.subset.clip_box = (None, None, None, None)

        # Base dataset
        self.ds0 = xr.Dataset()

    def _make_da(self, values: np.ndarray) -> xr.DataArray:
        """
        Create a DataArray matching rioxarray.open_rasterio output shape:
        (band, y, x) with coordinate attrs that we expect to be stripped.
        """
        da = xr.DataArray(
            values,
            dims=("band", "y", "x"),
            coords={
                "band": [1],
                "y": np.arange(values.shape[1]),
                "x": np.arange(values.shape[2]),
            },
        )
        # Pretend rioxarray attached attrs on coords
        da.coords["x"].attrs = {
            "axis": "X",
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
            "units": "metre",
        }
        da.coords["y"].attrs = {
            "axis": "Y",
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
            "units": "metre",
        }
        return da

    def _make_ds(self, meas_vars_list: list, values: np.ndarray):
        multi_bnd_ds = xr.Dataset()

        for meas_var in meas_vars_list:
            multi_bnd_ds.assign(meas_var=self._make_da(values))

        return multi_bnd_ds

    @patch("eoio.readers.planetscope.data_io.convert_xy")
    @patch("eoio.readers.planetscope.data_io.lazy_rioxarray")
    def test_read_tif_into_dataset_returns_dataset(self, mock_lazy_rioxarray, mock_convert_xy):
        """Test that read_tif_into_dataset returns a Dataset."""
        mock_rxr = MagicMock()
        mock_lazy_rioxarray.return_value = mock_rxr

        da = self._make_da(np.array([[[100, 200], [300, 400]]], dtype=np.uint16))
        mock_rxr.open_rasterio.return_value = da

        # Mock coordinate conversion
        mock_convert_xy.return_value = (np.ndarray((2, 2)), np.ndarray((2, 2)))

        ds = read_tif_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["B1"],
            subset=self.subset,
            mtd=self.mtd,
        )

        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.planetscope.data_io.convert_xy")
    @patch("eoio.readers.planetscope.data_io.lazy_rioxarray")
    def test_read_tif_into_dataset_applies_scale_factor(self, mock_lazy_rioxarray, mock_convert_xy):
        """Test that scale factor is applied to values."""
        mock_rxr = MagicMock()
        mock_lazy_rioxarray.return_value = mock_rxr

        # Create data with known values
        raw_data = np.array([[[10000, 20000], [30000, 40000]]], dtype=np.uint16)
        da = self._make_da(raw_data)
        da.attrs["scale_factor"] = 2
        da.attrs["add_offset"] = 0
        da.attrs["img_fill_value"] = 0
        mock_rxr.open_rasterio.return_value = da

        # Mock coordinate conversion
        mock_convert_xy.return_value = (np.ndarray((2, 2)), np.ndarray((2, 2)))

        ds = read_tif_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["B1"],
            subset=self.subset,
            mtd=self.mtd,
        )

        # Values should be divided by scale factor
        expected = np.array([[5000, 10000], [15000, 20000]], dtype=np.uint16)
        np.testing.assert_array_almost_equal(ds["B1"].values, expected, decimal=5)

    @patch("eoio.readers.planetscope.data_io.convert_xy")
    @patch("eoio.readers.planetscope.data_io.lazy_rioxarray")
    def test_read_tif_into_dataset_masks_zero_as_nodata(self, mock_lazy_rioxarray, mock_convert_xy):
        """Test that zero values are masked as NaN and non-zero values are preserved."""
        mock_rxr = MagicMock()
        mock_lazy_rioxarray.return_value = mock_rxr

        # Data with a zero value
        raw_data = np.array([[[0, 10000], [20000, 30000]]], dtype=np.uint16)
        da = self._make_da(raw_data)
        da.attrs["scale_factor"] = 2
        da.attrs["add_offset"] = 0
        da.attrs["img_fill_value"] = 0
        mock_rxr.open_rasterio.return_value = da

        # Mock coordinate conversion
        mock_convert_xy.return_value = (np.ndarray((2, 2)), np.ndarray((2, 2)))

        ds = read_tif_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["B1"],
            subset=self.subset,
            mtd=self.mtd,
            mask_zero_as_nodata=True,
        )

        # Zero should be masked as NaN
        self.assertTrue(np.isnan(ds["B1"].values[0, 0]))
        # Other values should be present and scaled correctly
        self.assertAlmostEqual(float(ds["B1"].values[0, 1]), 5000, places=5)
        self.assertAlmostEqual(float(ds["B1"].values[1, 0]), 10000, places=5)
        self.assertAlmostEqual(float(ds["B1"].values[1, 1]), 15000, places=5)

    @patch("eoio.readers.planetscope.data_io.convert_xy")
    @patch("eoio.readers.planetscope.data_io.lazy_rioxarray")
    def test_read_tif_into_dataset_calculates_offset_correctly(self, mock_lazy_rioxarray, mock_convert_xy):
        """Test that offset is added correctly and non-zero data is preserved."""
        mock_rxr = MagicMock()
        mock_lazy_rioxarray.return_value = mock_rxr

        raw_data = np.array([[[50, 100], [0, 200]]], dtype=np.uint16)
        da = self._make_da(raw_data)
        da.attrs["add_offset"] = 5
        da.attrs["img_fill_value"] = 0
        mock_rxr.open_rasterio.return_value = da

        # Mock coordinate conversion
        mock_convert_xy.return_value = (np.ndarray((2, 2)), np.ndarray((2, 2)))

        ds = read_tif_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["B1"],
            subset=self.subset,
            mtd=self.mtd,
        )

        # Values should be present and offset correctly
        self.assertAlmostEqual(float(ds["B1"].values[0, 0]), 55, places=5)
        self.assertAlmostEqual(float(ds["B1"].values[0, 1]), 105, places=5)
        self.assertAlmostEqual(float(ds["B1"].values[1, 1]), 205, places=5)
        # Zero should be masked as NaN
        self.assertTrue(np.isnan(ds["B1"].values[1, 0]))

    @patch("eoio.readers.planetscope.data_io.convert_xy")
    @patch("eoio.readers.planetscope.data_io.lazy_rioxarray")
    def test_read_tif_into_dataset_with_none_subset(self, mock_lazy_rioxarray, mock_convert_xy):
        """Test that None subset is handled correctly."""
        mock_rxr = MagicMock()
        mock_lazy_rioxarray.return_value = mock_rxr

        da = self._make_da(np.array([[[50, 100], [0, 200]]], dtype=np.uint16))
        mock_rxr.open_rasterio.return_value = da

        # Mock coordinate conversion
        mock_convert_xy.return_value = (np.ndarray((2, 2)), np.ndarray((2, 2)))

        ds = read_tif_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["B1"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("B1", ds)


if __name__ == "__main__":
    unittest.main()
