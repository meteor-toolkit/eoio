import unittest
from unittest.mock import Mock, patch

import xarray as xr

from eoio.readers.copernicus_dem.data_io import append_data_vars


class TestAppendDataVars(unittest.TestCase):

    @patch("eoio.readers.copernicus_dem.data_io.lazy_rioxarray")
    def test_append_data_vars(self, mock_lazy_rioxarray):

        ds = xr.Dataset()

        da = xr.DataArray([[1]], dims=("y", "x"))

        rxr = Mock()
        rxr.open_rasterio.return_value.squeeze.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        result = append_data_vars(
            ds=ds,
            layout="/tmp/dem.tif",
        )

        rxr.open_rasterio.assert_called_once_with(
            "/tmp/dem.tif",
            chunks=None,
        )

        self.assertIs(result, ds)
        self.assertIn("elevation", result.data_vars)

    @patch("eoio.readers.copernicus_dem.data_io.suggest_raster_chunks")
    @patch("eoio.readers.copernicus_dem.data_io.lazy_rioxarray")
    def test_append_data_vars_suggests_chunks(
        self,
        mock_lazy_rioxarray,
        mock_suggest_chunks,
    ):

        ds = xr.Dataset()

        mock_suggest_chunks.return_value = {
            "x": 512,
            "y": 512,
        }

        da = xr.DataArray([[1]], dims=("y", "x"))

        rxr = Mock()
        rxr.open_rasterio.return_value.squeeze.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        append_data_vars(
            ds=ds,
            layout="/tmp/dem.tif",
            use_chunks=True,
        )

        mock_suggest_chunks.assert_called_once_with(
            "/tmp/dem.tif"
        )

        rxr.open_rasterio.assert_called_once_with(
            "/tmp/dem.tif",
            chunks={"x": 512, "y": 512},
        )

    @patch("eoio.readers.copernicus_dem.data_io.suggest_raster_chunks")
    @patch("eoio.readers.copernicus_dem.data_io.lazy_rioxarray")
    def test_append_data_vars_uses_provided_chunks(
        self,
        mock_lazy_rioxarray,
        mock_suggest_chunks,
    ):

        ds = xr.Dataset()

        da = xr.DataArray([[1]], dims=("y", "x"))

        rxr = Mock()
        rxr.open_rasterio.return_value.squeeze.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        append_data_vars(
            ds=ds,
            layout="/tmp/dem.tif",
            use_chunks=True,
            chunks={"x": 256, "y": 256},
        )

        mock_suggest_chunks.assert_not_called()

        rxr.open_rasterio.assert_called_once_with(
            "/tmp/dem.tif",
            chunks={"x": 256, "y": 256},
        )
