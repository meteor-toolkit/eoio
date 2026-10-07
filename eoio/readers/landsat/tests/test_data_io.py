"""eoio.readers.landsat.tests.test_data_io - tests for eoio.readers.landsat.data_io"""

import unittest
import warnings
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from eoio.readers.landsat.data_io import _nearest_resample_2d, read_bands_into_dataset


class TestNearestResample2D(unittest.TestCase):
    def test_exact_double_resolution_matches_landsat_pan_vs_30m_ratio(self):
        # Mirrors Landsat's own 30m angle grid -> 15m panchromatic band relationship.
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _nearest_resample_2d(arr, (4, 4))

        self.assertEqual(result.shape, (4, 4))
        expected = np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_non_exact_ratio_still_produces_target_shape(self):
        """An ROI-subsetted angle grid and band can differ from a clean integer ratio
        (each clipped independently on its own native pixel grid -- see
        _nearest_resample_2d's own docstring); resampling should still degrade
        gracefully to the exact target shape rather than raising."""
        arr = np.arange(9.0).reshape(3, 3)
        result = _nearest_resample_2d(arr, (5, 7))

        self.assertEqual(result.shape, (5, 7))
        # Every resampled value must come from the source array (nearest-neighbour, no
        # interpolation/extrapolation beyond the source's own value range).
        self.assertTrue(np.isin(result, arr).all())

    def test_downsampling_also_produces_target_shape(self):
        arr = np.arange(16.0).reshape(4, 4)
        result = _nearest_resample_2d(arr, (2, 2))

        self.assertEqual(result.shape, (2, 2))


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
    def test_read_bands_into_dataset_UsesPerPixelSZA_WithMismatchedGrid(self, mock_lazy_rioxarray):
        """Regression test: the angle files are always read at their own native 30m
        grid, which only matches most Landsat bands -- the panchromatic B8 (15m) used
        to hit a raw shape mismatch dividing a (2, 2) band by a (1, 1) angle array,
        caught by the broad except and silently falling back to the much less accurate
        per-scene metadata sun_elevation (with a "correction failed" warning implying
        angle data wasn't read at all, when it actually had been -- just at the wrong
        resolution). The angle grid should now be resampled onto the band's own (finer)
        grid first, so the per-pixel correction actually succeeds, with no warning."""
        # geometry_id "15m" (distinct from the angle grid's own "30m", set below) so the
        # post-correction rename step gives B1 its own x_15m/y_15m dims -- matching real
        # Landsat product metadata (see LSMetadataExtractor.get_angle_metadata's
        # hardcoded "30m") and avoiding an unrelated dimension-size clash when assigning
        # into `ds`, which already has a plain-dimensioned "solar_zenith_angle" below.
        self.mtd.get_variable_product_metadata.return_value = {
            "reflectance_mult": 1.0,
            "reflectance_add": 0.0,
            "geometry_id": "15m",
        }
        # A 2x2 "band" (mimicking B8 at double the 30m angle grid's linear resolution)
        # with a uniform value, so every pixel's expected corrected value is identical
        # regardless of which angle-grid cell nearest-neighbour resampling picked.
        da = self._make_da(np.array([[[10, 10], [10, 10]]], dtype=np.uint16))
        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        # A single-pixel (1, 1) angle grid -- shape mismatch against the 2x2 band above.
        ds["solar_zenith_angle"] = xr.DataArray(np.array([[30.0]]), dims=("y", "x"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_bands_into_dataset(ds=ds, layout=self.layout, meas_vars=["B1"], subset=None, mtd=self.mtd)

        self.assertEqual(caught, [])
        # 10 / sin(90-30) = 10 / sin(60) at every pixel -- the resampled SZA (nearest-
        # neighbour from the single source cell) is 30 everywhere, not the fallback
        # metadata sun_elevation of 45 (which would give a different result).
        expected = 10.0 / np.sin(np.deg2rad(60.0))
        self.assertEqual(ds["B1"].values.shape, (2, 2))
        np.testing.assert_allclose(ds["B1"].values, expected)

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

    @patch("eoio.readers.landsat.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_MasksNegativeReflectancePixel(self, mock_lazy_rioxarray):
        """Landsat Collection 2 reflectance products use a negative scaled value (e.g.
        -0.1) as a fill sentinel for pixels outside the actual acquired footprint, distinct
        from the encoded_nodata/zero-DN raw-DN masking above (only visible post-scaling) --
        physically implausible, so it must come out as NaN like any other missing-data
        value, not a real reading."""
        self.mtd.get_variable_product_metadata.return_value = {
            "reflectance_mult": 1.0,
            # A negative additive offset (realistic -- real Landsat metadata's own
            # REFLECTANCE_ADD_BAND_x is typically negative) pushes a low-but-nonzero,
            # non-encoded-nodata DN below zero once scaled: fill data that neither of the
            # earlier raw-DN nodata checks (encoded_nodata, mask_zero_as_nodata) catches.
            "reflectance_add": -1000.0,
        }
        da = self._make_da(np.array([[[10, 5000]]], dtype=np.uint16))
        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        read_bands_into_dataset(ds=ds, layout=self.layout, meas_vars=["B1"], subset=None, mtd=self.mtd)

        # DN=10 -> 10 - 1000 = -990 (negative, masked); DN=5000 -> 4000 (positive, kept).
        self.assertTrue(np.isnan(ds["B1"].values[0, 0]))
        self.assertFalse(np.isnan(ds["B1"].values[0, 1]))


if __name__ == "__main__":
    unittest.main()
