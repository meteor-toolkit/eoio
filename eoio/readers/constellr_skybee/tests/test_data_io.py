"""eoio.readers.constellr_skybee.tests.test_data_io - unit tests for eoio.readers.constellr_skybee.data_io"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from eoio.readers.constellr_skybee.data_io import get_var_geometry, get_var_resolution, read_bands_into_dataset


class TestSkybeeDataIO(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.layout = MagicMock()
        self.layout.image_file.return_value = "/tmp/image.tiff"
        self.layout.tif_band_files.return_value = {"B1": "/tmp/B1.tiff"}
        self.mtd = MagicMock()
        self.mtd.product_metadata = {"geospatial_bounds_crs": "EPSG:4326"}

    @patch("eoio.readers.constellr_skybee.data_io.convert_xy")
    @patch("eoio.readers.constellr_skybee.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_adds_band_and_coords(self, mock_lazy_rioxarray, mock_convert_xy):
        mock_rxr = MagicMock()
        data = xr.DataArray(
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            dims=("y", "x"),
            coords={"y": [0, 1], "x": [0, 1]},
        )
        mock_rxr.open_rasterio.return_value = data
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_convert_xy.return_value = (
            np.array([[10.0, 11.0], [12.0, 13.0]]),
            np.array([[20.0, 21.0], [22.0, 23.0]]),
        )

        out = read_bands_into_dataset(
            ds=self.ds,
            layout=self.layout,
            meas=["B1"],
            subset=None,
            mtd=self.mtd,
            use_chunks=False,
            chunks=None,
        )

        self.assertIn("B1", out)
        self.assertEqual(out["B1"].values[0, 0], 1.0)
        mock_rxr.open_rasterio.assert_called_once_with("/tmp/B1.tiff", chunks=None, mask_and_scale=True)
        self.assertIn("longitude_60m", out["B1"].coords)
        self.assertIn("latitude_60m", out["B1"].coords)

    @patch("eoio.readers.constellr_skybee.data_io.convert_xy")
    @patch("eoio.readers.constellr_skybee.data_io.lazy_rioxarray")
    def test_single_band_geotiff_leaves_no_scalar_band_coordinate(self, mock_lazy_rioxarray, mock_convert_xy):
        """rasterio gives (band=1, y, x); squeeze() leaves a scalar "band" coord that would collide with
        the stack processor's "band" dimension at to_datatree."""
        mock_rxr = MagicMock()
        mock_rxr.open_rasterio.return_value = xr.DataArray(
            np.ones((1, 2, 2), dtype=np.float32),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": [0, 1], "x": [0, 1]},
        )
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_convert_xy.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))

        out = read_bands_into_dataset(
            ds=self.ds, layout=self.layout, meas=["B1"], subset=None, mtd=self.mtd, use_chunks=False, chunks=None
        )

        self.assertNotIn("band", out.variables)
        self.assertNotIn("band", out["B1"].coords)

    @patch("eoio.readers.constellr_skybee.data_io.convert_xy")
    @patch("eoio.readers.constellr_skybee.data_io.lazy_rioxarray")
    def test_a_one_pixel_wide_roi_keeps_its_spatial_dimensions(self, mock_lazy_rioxarray, mock_convert_xy):
        """Only the band axis is dropped. A bare squeeze() would also collapse an x or y dimension of
        length 1 -- a small ROI on the 60 m grid -- and the band would lose its grid."""
        mock_rxr = MagicMock()
        mock_rxr.open_rasterio.return_value = xr.DataArray(
            np.ones((1, 1, 3), dtype=np.float32),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": [0], "x": [0, 1, 2]},
        )
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_convert_xy.return_value = (np.zeros((1, 3)), np.zeros((1, 3)))

        out = read_bands_into_dataset(
            ds=self.ds, layout=self.layout, meas=["B1"], subset=None, mtd=self.mtd, use_chunks=False, chunks=None
        )

        self.assertEqual(out["B1"].dims, ("y_60m", "x_60m"))
        self.assertEqual(out["B1"].shape, (1, 3))

    @patch("eoio.readers.constellr_skybee.data_io.convert_xy")
    @patch("eoio.readers.constellr_skybee.data_io.lazy_rioxarray")
    def test_gdal_file_tags_are_not_kept_as_variable_attrs(self, mock_lazy_rioxarray, mock_convert_xy):
        """ "UNITS" (a GeoTIFF tag) next to eoio's own "units" cannot be written to netCDF."""
        mock_rxr = MagicMock()
        mock_rxr.open_rasterio.return_value = xr.DataArray(
            np.ones((2, 2), dtype=np.float32),
            dims=("y", "x"),
            coords={"y": [0, 1], "x": [0, 1]},
            attrs={"UNITS": "W/m2/sr/um", "NAME": "vnir04", "STATISTICS_MEAN": "12.3", "AREA_OR_POINT": "Area"},
        )
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_convert_xy.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))

        out = read_bands_into_dataset(
            ds=self.ds, layout=self.layout, meas=["B1"], subset=None, mtd=self.mtd, use_chunks=False, chunks=None
        )

        self.assertEqual(out["B1"].attrs, {})

    @patch("eoio.readers.constellr_skybee.data_io.suggest_raster_chunks")
    @patch("eoio.readers.constellr_skybee.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_uses_chunks_if_requested(self, mock_lazy_rioxarray, mock_suggest_chunks):
        mock_suggest_chunks.return_value = {"x": 128, "y": 128}
        mock_rxr = MagicMock()
        data = xr.DataArray(
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            dims=("y", "x"),
            coords={"y": [0, 1], "x": [0, 1]},
        )
        mock_rxr.open_rasterio.return_value = data
        mock_lazy_rioxarray.return_value = mock_rxr

        read_bands_into_dataset(
            ds=self.ds,
            layout=self.layout,
            meas=["B1"],
            subset=None,
            mtd=self.mtd,
            use_chunks=True,
            chunks=None,
        )

        mock_suggest_chunks.assert_called_once_with("/tmp/image.tiff", target_mb=32.0)
        mock_rxr.open_rasterio.assert_called_once_with("/tmp/B1.tiff", chunks={"x": 128, "y": 128}, mask_and_scale=True)

    @patch("eoio.readers.constellr_skybee.data_io.convert_xy")
    def test_read_bands_into_dataset_masks_nodata_and_scales(self, mock_convert_xy):
        import rasterio
        from rasterio.transform import from_origin

        mock_convert_xy.side_effect = lambda xy, *_: (np.zeros((2, 2)), np.zeros((2, 2)))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "vnir02.tiff")
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint16",
                crs="EPSG:32733",
                transform=from_origin(0, 20, 10, 10),
                nodata=0,
            ) as dst:
                dst.write(np.array([[0, 100], [200, 300]], dtype="uint16"), 1)
                dst.scales = (0.01,)
                dst.offsets = (0.5,)
            self.layout.tif_band_files.return_value = {"B2": path}

            out = read_bands_into_dataset(
                ds=xr.Dataset(), layout=self.layout, meas=["B2"], subset=None, mtd=self.mtd, use_chunks=False
            )

        self.assertTrue(np.isnan(out["B2"].values[0, 0]))
        np.testing.assert_allclose(out["B2"].values.ravel()[1:], [1.5, 2.5, 3.5])

    @patch("eoio.readers.constellr_skybee.data_io.convert_xy")
    def test_read_bands_into_dataset_roi_clip_keeps_scaling_and_mask(self, mock_convert_xy):
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box

        mock_convert_xy.side_effect = lambda xy, *_: (np.zeros((2, 2)), np.zeros((2, 2)))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "vnir02.tiff")
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=4,
                width=4,
                count=1,
                dtype="uint16",
                crs="EPSG:32733",
                transform=from_origin(0, 40, 10, 10),
                nodata=0,
            ) as dst:
                data = np.full((4, 4), 200, dtype="uint16")
                data[0, 0] = 0
                dst.write(data, 1)
                dst.scales = (0.01,)
                dst.offsets = (0.0,)
            self.layout.tif_band_files.return_value = {"B2": path}
            # the lower-left 2 x 2 pixels
            subset = MagicMock(geometries=[box(0, 0, 20, 20)])

            out = read_bands_into_dataset(
                ds=xr.Dataset(), layout=self.layout, meas=["B2"], subset=subset, mtd=self.mtd, use_chunks=False
            )

        np.testing.assert_allclose(out["B2"].values, 2.0)

    def test_get_var_geometry(self):
        self.assertEqual(get_var_geometry("B1"), 60)
        self.assertEqual(get_var_geometry("B2"), 10)

    def test_get_var_resolution(self):
        self.assertEqual(get_var_resolution("B2"), "10m")
        self.assertEqual(get_var_resolution("B1"), "60m")


if __name__ == "__main__":
    unittest.main()
