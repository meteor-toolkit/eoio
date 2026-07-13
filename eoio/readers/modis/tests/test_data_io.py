"""eoio.readers.modis.test_data_io - tests for eoio.readers.modis.data_io"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from eoio.readers.modis.data_io import read_bands_into_dataset


class TestReadBandsIntoDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Mock layout and metadata extractor
        self.layout = MagicMock()
        self.layout.processing_level = "L1B"
        self.layout.file_res_key = "H"
        self.layout.path = "/tmp/fake_MOD02HDF.A2020001.0000.061.2020002121530.hdf"

        self.mtd = MagicMock()
        self.mtd.product_metadata = {}

        # Default per-variable metadata
        def _var_mtd(var):
            return {"band_id": "500m Surface Reflectance Band 1", "band_idx": 0, "geometry_id": "500m"}

        self.mtd.variable_product_metadata.side_effect = _var_mtd

        # Mock geolocation dataset
        self.geolocation_ds = xr.Dataset()

        # Base dataset
        self.ds = xr.Dataset()

    def _make_da(self, values: np.ndarray) -> xr.Dataset:
        """
        Create a DataArray matching rioxarray.open_rasterio output shape:
        (band, y, x)
        """
        da = xr.Dataset(
            data_vars={
                "500m Surface Reflectance Band 1": (["band", "y", "x"], values),
                "500m Reflectance Band Quality": (
                    ["band", "y", "x"],
                    int("0b1000000000000000000000000000000", 2) * np.ones_like(values, dtype=np.uint32),
                ),
            },
            coords={
                "band": [1],
                "y": np.arange(values.shape[1]),
                "x": np.arange(values.shape[2]),
            },
        )
        return da

    @patch("eoio.readers.modis.data_io.add_geolocation")
    @patch("eoio.readers.modis.data_io.lazy_rioxarray")
    def test_reads_bands_into_dataset_basic(self, mock_lazy_rioxarray, mock_add_geoloc):
        """Test basic band reading and data loading."""
        da = self._make_da(np.array([[[100, 200], [300, 400]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr
        mock_add_geoloc.side_effect = lambda ds, **kwargs: ds

        ds = xr.Dataset()
        result = read_bands_into_dataset(
            ds=ds,
            geolocation_ds=self.geolocation_ds,
            layout=self.layout,
            meas_vars=["Band 1"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("Band 1", result.data_vars)
        self.assertEqual(result["Band 1"].dtype, np.dtype("float32"))

    @patch("eoio.readers.modis.data_io.add_geolocation")
    @patch("eoio.readers.modis.data_io.lazy_rioxarray")
    def test_masks_zero_as_nodata(self, mock_lazy_rioxarray, mock_add_geoloc):
        """Test that zeros are masked as nodata when mask_zero_as_nodata=True."""
        da = self._make_da(np.array([[[0, 100], [200, 0]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr
        mock_add_geoloc.side_effect = lambda ds, **kwargs: ds

        ds = xr.Dataset()
        result = read_bands_into_dataset(
            ds=ds,
            geolocation_ds=self.geolocation_ds,
            layout=self.layout,
            meas_vars=["Band 1"],
            subset=None,
            mtd=self.mtd,
            mask_zero_as_nodata=True,
        )

        # Zero values should be masked as NaN
        self.assertTrue(np.isnan(result["Band 1"].values[0, 0]))
        self.assertTrue(np.isnan(result["Band 1"].values[1, 1]))
        self.assertFalse(np.isnan(result["Band 1"].values[0, 1]))

    @patch("eoio.readers.modis.data_io.add_geolocation")
    @patch("eoio.readers.modis.data_io.lazy_rioxarray")
    def test_does_not_mask_zero_when_disabled(self, mock_lazy_rioxarray, mock_add_geoloc):
        """Test that zeros are not masked when mask_zero_as_nodata=False."""
        da = self._make_da(np.array([[[0, 100], [200, 0]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr
        mock_add_geoloc.side_effect = lambda ds, **kwargs: ds

        ds = xr.Dataset()
        result = read_bands_into_dataset(
            ds=ds,
            geolocation_ds=self.geolocation_ds,
            layout=self.layout,
            meas_vars=["Band 1"],
            subset=None,
            mtd=self.mtd,
            mask_zero_as_nodata=False,
        )

        # Zero values should remain as zero
        self.assertEqual(result["Band 1"].values[0, 0], 0.0)
        self.assertEqual(result["Band 1"].values[1, 1], 0.0)

    @patch("eoio.readers.modis.data_io.add_geolocation")
    @patch("eoio.readers.modis.data_io.suggest_raster_chunks")
    @patch("eoio.readers.modis.data_io.lazy_rioxarray")
    def test_use_chunks_calls_suggest_and_passes_chunks(self, mock_lazy_rioxarray, mock_suggest, mock_add_geoloc):
        """Test that use_chunks=True calls suggest_raster_chunks and passes chunks to rioxarray."""
        mock_suggest.return_value = {"x": 256, "y": 256}

        da = self._make_da(np.array([[[100, 200]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr
        mock_add_geoloc.side_effect = lambda ds, **kwargs: ds

        ds = xr.Dataset()
        result = read_bands_into_dataset(
            ds=ds,
            geolocation_ds=self.geolocation_ds,
            layout=self.layout,
            meas_vars=["Band 1"],
            subset=None,
            mtd=self.mtd,
            use_chunks=True,
        )

        mock_suggest.assert_called_once()
        rxr.open_rasterio.assert_called_once()
        # Verify chunks were passed
        call_kwargs = rxr.open_rasterio.call_args.kwargs
        self.assertIn("chunks", call_kwargs)
        self.assertEqual(call_kwargs["chunks"], {"x": 256, "y": 256})

    @patch("eoio.readers.modis.data_io.add_geolocation")
    @patch("eoio.readers.modis.data_io.lazy_rioxarray")
    def test_reads_L2_product(self, mock_lazy_rioxarray, mock_add_geoloc):
        """Test reading L2 MODIS products."""
        self.layout.processing_level = "L2"
        self.layout.file_res_key = "H"

        da = self._make_da(np.array([[[100, 200]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = [da, da]  # L2 products return list indexed by resolution
        mock_lazy_rioxarray.return_value = rxr
        mock_add_geoloc.side_effect = lambda ds, **kwargs: ds

        ds = xr.Dataset()
        result = read_bands_into_dataset(
            ds=ds,
            geolocation_ds=self.geolocation_ds,
            layout=self.layout,
            meas_vars=["Band 1"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("Band 1", result.data_vars)


if __name__ == "__main__":
    unittest.main()
