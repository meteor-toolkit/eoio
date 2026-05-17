"""eoio.readers.landsat.tests.test_data_io - tests for eoio.readers.landsat.data_io"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from eoio.readers.landsat.data_io import read_bands_into_dataset


class TestReadBandsIntoDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = MagicMock()
        self.mtd = MagicMock()
        self.subset = MagicMock()
        self.subset.clip_box = None
        self.subset.geometries = None

        self.band_paths = {"B1": "/tmp/fake_B1.TIF"}
        self.layout.tif_band_files.return_value = self.band_paths

        # Default metadata
        self.mtd.get_product_metadata.return_value = {"sun_elevation": 45.0}  # sin(45) ~= 0.707
        self.mtd.get_variable_product_metadata.return_value = {
            "reflectance_mult": 1.0,
            "reflectance_add": 0.0,
        }

    def _make_da(self, values: np.ndarray) -> xr.DataArray:
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

    @patch("eoio.readers.landsat.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_BasicUse(self, mock_lazy_rioxarray):
        # Input 10. With Sun elevation 45, sin(45)=0.707. 10/0.707 = 14.14
        da = self._make_da(np.array([[[10]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        read_bands_into_dataset(ds=ds, layout=self.layout, meas_vars=["B1"], subset=None, mtd=self.mtd)

        self.assertIn("B1", ds)
        # Check value. 10 * 1.0 + 0.0 = 10. 10 / sin(45).
        expected = 10.0 / np.sin(np.deg2rad(45.0))
        self.assertAlmostEqual(ds["B1"].values[0, 0], expected, places=4)

    @patch("eoio.readers.landsat.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_AppliesScaling(self, mock_lazy_rioxarray):
        self.mtd.get_variable_product_metadata.return_value = {
            "reflectance_mult": 2.0,
            "reflectance_add": 5.0,
        }

        da = self._make_da(np.array([[[10]]], dtype=np.uint16))
        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        read_bands_into_dataset(ds=ds, layout=self.layout, meas_vars=["B1"], subset=None, mtd=self.mtd)

        # (10 * 2.0 + 5.0) / sin(45) = 25 / 0.707 = 35.35
        expected = 25.0 / np.sin(np.deg2rad(45.0))
        self.assertAlmostEqual(ds["B1"].values[0, 0], expected, places=4)

    @patch("eoio.readers.landsat.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_UsesPerPixelSZA(self, mock_lazy_rioxarray):
        da = self._make_da(np.array([[[10]]], dtype=np.uint16))
        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        # Add SZA. 90 - 30 = 60 elevation. sin(60) = 0.866
        ds["solar_zenith_angle"] = xr.DataArray(np.array([[30.0]]), dims=("y", "x"))

        read_bands_into_dataset(ds=ds, layout=self.layout, meas_vars=["B1"], subset=None, mtd=self.mtd)

        # 10 / sin(90-30) = 10 / sin(60)
        expected = 10.0 / np.sin(np.deg2rad(60.0))
        self.assertAlmostEqual(ds["B1"].values[0, 0], expected, places=4)

    @patch("eoio.readers.landsat.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_MasksZeroAsNodata(self, mock_lazy_rioxarray):
        da = self._make_da(np.array([[[0, 10]]], dtype=np.uint16))
        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        read_bands_into_dataset(
            ds=ds,
            layout=self.layout,
            meas_vars=["B1"],
            subset=None,
            mtd=self.mtd,
            mask_zero_as_nodata=True,
        )

        self.assertTrue(np.isnan(ds["B1"].values[0, 0]))
        self.assertFalse(np.isnan(ds["B1"].values[0, 1]))


if __name__ == "__main__":
    unittest.main()
