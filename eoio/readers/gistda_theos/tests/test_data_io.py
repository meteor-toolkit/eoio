"""eoio.readers.gistda_theos.tests.test_data_io - unit tests for eoio.readers.gistda_theos.data_io"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
from rasterio.transform import Affine, from_gcps
from rasterio.control import GroundControlPoint

from eoio.readers.gistda_theos.data_io import (
    GISTDATHEOSDataIOError,
    _band_file_position,
    georeference_from_gcps,
    get_var_geometry,
    lonlat_grid_from_gcps,
    read_bands_into_dataset,
)

# A real product's IMAGERY.TIF stores bands in this order (TIFFTAG_IMAGEDESCRIPTION
# "B3 B2 B1 B4 ") -- band index in the raw TIFF (0-indexed) for each meas name.
_TIFF_BAND_ORDER = ["B3", "B2", "B1", "B4"]

# The real product's own 4 corner GCPs (row, col, lon, lat) -- see METADATA.DIM's
# Geoposition_Points / the TIFF's embedded GDAL GCPs.
_REAL_GCPS = [
    (1.5, 1.5, 14.743870078372012, -23.137489713016514),
    (1.5, 6000.5, 15.681031070663265, -23.291976925944475),
    (6000.5, 6000.5, 15.482495497605992, -24.05062963013373),
    (6000.5, 1.5, 14.539471944936292, -23.89488328853735),
]


def _make_gcp_image(width: int = 6000, height: int = 6000) -> xr.DataArray:
    """A DataArray shaped like a real THEOS IMAGERY.TIF read: 4 bands in
    _TIFF_BAND_ORDER, bare pixel-index x/y, no CRS/transform -- and a
    ``.rio.get_gcps()``/``.rio.crs`` that report the real product's own 4 corner
    GCPs, exactly like rioxarray reading a real THEOS TIFF's embedded GDAL GCPs.
    """
    data = np.zeros((4, height, width), dtype=np.uint16)
    for pos, name in enumerate(_TIFF_BAND_ORDER):
        # distinct constant fill per physical band, so a test can tell which
        # physical band's data ended up under which meas var name
        data[pos] = {"B1": 10, "B2": 20, "B3": 30, "B4": 40}[name]
    da = xr.DataArray(data, dims=("band", "y", "x"), coords={"band": [1, 2, 3, 4]})
    gcps = [GroundControlPoint(row=r, col=c, x=x, y=y) for r, c, x, y in _REAL_GCPS]
    da = da.rio.write_gcps(gcps, "EPSG:4326")
    return da


class TestBandFilePosition(unittest.TestCase):
    def test_matches_the_real_products_tiff_band_order(self):
        # verified against a real product's TIFFTAG_IMAGEDESCRIPTION "B3 B2 B1 B4 "
        self.assertEqual(_band_file_position("B3"), 0)
        self.assertEqual(_band_file_position("B2"), 1)
        self.assertEqual(_band_file_position("B1"), 2)
        self.assertEqual(_band_file_position("B4"), 3)

    def test_unknown_band_raises_a_clear_error(self):
        with self.assertRaises(GISTDATHEOSDataIOError):
            _band_file_position("B9")


class TestGeoreferenceFromGcps(unittest.TestCase):
    def test_raises_when_no_gcps(self):
        da = xr.DataArray(
            np.zeros((1, 2, 2), dtype=np.uint8),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": [0, 1], "x": [0, 1]},
        )
        with self.assertRaises(GISTDATHEOSDataIOError):
            georeference_from_gcps(da)

    def test_writes_transform_and_crs_from_real_gcps(self):
        img = _make_gcp_image(width=10, height=10)
        out, transform, gcp_crs = georeference_from_gcps(img)

        self.assertEqual(str(gcp_crs), "EPSG:4326")
        self.assertEqual(str(out.rio.crs), "EPSG:4326")
        self.assertIsInstance(transform, Affine)
        # a real affine fit from 4 rotated corners always has nonzero shear terms
        self.assertNotEqual(transform.b, 0.0)
        self.assertNotEqual(transform.d, 0.0)
        # x/y stay bare pixel indices -- only .rio.transform()/.rio.crs change
        np.testing.assert_array_equal(out.coords["x"].values, img.coords["x"].values)
        np.testing.assert_array_equal(out.coords["y"].values, img.coords["y"].values)


class TestLonLatGridFromGcps(unittest.TestCase):
    def test_full_grid_matches_the_known_corners(self):
        gcps = [GroundControlPoint(row=r, col=c, x=x, y=y) for r, c, x, y in _REAL_GCPS]
        transform = from_gcps(gcps)

        lon, lat = lonlat_grid_from_gcps(transform, width=6000, height=6000)

        self.assertEqual(lon.shape, (6000, 6000))
        self.assertEqual(lat.shape, (6000, 6000))
        # top-left/top-right corners must resolve close to their own real GCPs --
        # this is exactly the check that catches the independent-axis bug: with x/y
        # treated as independent 1-D axes, the top-right corner's *latitude* comes
        # out equal to the top-left's (wrong for a rotated scene).
        self.assertAlmostEqual(lon[0, 0], _REAL_GCPS[0][2], delta=0.01)
        self.assertAlmostEqual(lat[0, 0], _REAL_GCPS[0][3], delta=0.01)
        self.assertAlmostEqual(lon[0, -1], _REAL_GCPS[1][2], delta=0.01)
        self.assertAlmostEqual(lat[0, -1], _REAL_GCPS[1][3], delta=0.01)
        # rotation check: latitude must change moving along a single row
        self.assertNotAlmostEqual(lat[0, 0], lat[0, -1], delta=0.01)


class TestReadBandsIntoDataset(unittest.TestCase):
    """Unit tests for read_bands_into_dataset function."""

    def _make_mtd(self):
        mtd = Mock()
        mtd.product_metadata = {"geospatial_bounds_crs": "EPSG:4326"}
        mtd.variable_product_metadata.return_value = {"band_gain": 1.0, "band_bias": 0.0}
        return mtd

    @patch("eoio.readers.gistda_theos.data_io.lazy_rioxarray")
    def test_reads_each_band_from_its_real_tiff_position_not_swapped(self, mock_lazy_rioxarray):
        """The regression test for the B1/B3 swap: each meas var's data must come
        from ITS OWN physical band, not the position it happens to be requested
        in -- exercised here with every band requested out of TIFF order."""
        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_rxr.open_rasterio.return_value = _make_gcp_image(width=4, height=4)

        layout = Mock()
        layout.image_file.return_value = "/tmp/IMAGERY.TIF"

        result = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=layout,
            meas=["B1", "B2", "B3", "B4"],
            subset=None,
            mtd=self._make_mtd(),
            use_chunks=False,
            chunks=None,
        )

        # _make_gcp_image fills B1/B2/B3/B4's *physical* TIFF positions with
        # 10/20/30/40 respectively -- each var must carry its own fill value back.
        for name, fill in {"B1": 10, "B2": 20, "B3": 30, "B4": 40}.items():
            self.assertTrue((result[name].values == fill).all(), f"{name} did not read its own physical band")

    @patch("eoio.readers.gistda_theos.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_returns_dataset(self, mock_lazy_rioxarray):
        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_rxr.open_rasterio.return_value = _make_gcp_image(width=4, height=4)

        layout = Mock()
        layout.image_file.return_value = "/tmp/IMAGERY.TIF"

        result = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=layout,
            meas=["B1"],
            subset=None,
            mtd=self._make_mtd(),
            use_chunks=False,
            chunks=None,
        )

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("B1", result.data_vars)
        self.assertIn(f"longitude_{get_var_geometry()}m", result.coords)
        self.assertIn(f"latitude_{get_var_geometry()}m", result.coords)
        # the full-image working coords must not leak into the returned dataset
        self.assertNotIn("longitude", result.coords)
        self.assertNotIn("latitude", result.coords)

    @patch("eoio.readers.gistda_theos.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset_masks_zero_values(self, mock_lazy_rioxarray):
        img = _make_gcp_image(width=2, height=2)
        img.values[_TIFF_BAND_ORDER.index("B1")] = np.array([[0, 5], [7, 0]])
        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_rxr.open_rasterio.return_value = img

        layout = Mock()
        layout.image_file.return_value = "/tmp/IMAGERY.TIF"

        result = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=layout,
            meas=["B1"],
            subset=None,
            mtd=self._make_mtd(),
            use_chunks=False,
            chunks=None,
        )

        band_da = result["B1"]
        y_dim = next(d for d in band_da.dims if d.startswith("y_"))
        x_dim = next(d for d in band_da.dims if d.startswith("x_"))

        # first element should be masked (converted to NaN), second should be 5
        self.assertTrue(np.isnan(band_da.isel({y_dim: 0, x_dim: 0}).values))
        self.assertEqual(band_da.isel({y_dim: 0, x_dim: 1}).values, 5)

    @patch("eoio.readers.gistda_theos.data_io.lazy_rioxarray")
    def test_longitude_latitude_capture_rotation(self, mock_lazy_rioxarray):
        """Regression test for the independent-x/y-axis bug: latitude must vary
        along a row (and longitude along a column) for this rotated scene."""
        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_rxr.open_rasterio.return_value = _make_gcp_image(width=50, height=50)

        layout = Mock()
        layout.image_file.return_value = "/tmp/IMAGERY.TIF"

        result = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=layout,
            meas=["B1"],
            subset=None,
            mtd=self._make_mtd(),
            use_chunks=False,
            chunks=None,
        )

        lat = result[f"latitude_{get_var_geometry()}m"].values
        lon = result[f"longitude_{get_var_geometry()}m"].values
        self.assertNotAlmostEqual(lat[0, 0], lat[0, -1], delta=0.001)
        self.assertNotAlmostEqual(lon[0, 0], lon[-1, 0], delta=0.001)


class TestReadBandsIntoDatasetRadiance(unittest.TestCase):
    """read_bands_into_dataset returns TOA radiance (DN / gain + bias); reflectance is left to the
    units processor."""

    @patch("eoio.readers.gistda_theos.data_io.lazy_rioxarray")
    def test_returns_radiance_not_reflectance(self, mock_lazy_rioxarray):
        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_rxr.open_rasterio.return_value = _make_gcp_image(width=2, height=2)

        layout = Mock()
        layout.image_file.return_value = "/tmp/IMAGERY.TIF"
        mtd = Mock()
        mtd.product_metadata = {"geospatial_bounds_crs": "EPSG:4326"}
        mtd.variable_product_metadata.return_value = {"band_gain": 2.0, "band_bias": 1.0}

        result = read_bands_into_dataset(
            ds=xr.Dataset(), layout=layout, meas=["B1"], subset=None, mtd=mtd, use_chunks=False, chunks=None
        )

        # B1's fill DN is 10: 10 / 2.0 + 1.0
        self.assertTrue(np.allclose(result["B1"].values, 6.0))


class TestReadBandsIntoDatasetShapeAndLaziness(unittest.TestCase):
    def _read(self, mock_lazy_rioxarray, image):
        mock_rxr = Mock()
        mock_lazy_rioxarray.return_value = mock_rxr
        mock_rxr.open_rasterio.return_value = image
        layout = Mock()
        layout.image_file.return_value = "/tmp/IMAGERY.TIF"
        mtd = Mock()
        mtd.product_metadata = {"geospatial_bounds_crs": "EPSG:4326"}
        mtd.variable_product_metadata.return_value = {"band_gain": 2.0, "band_bias": 1.0}
        return read_bands_into_dataset(
            ds=xr.Dataset(), layout=layout, meas=["B1"], subset=None, mtd=mtd, use_chunks=False, chunks=None
        )

    @patch("eoio.readers.gistda_theos.data_io.lazy_rioxarray")
    def test_a_one_pixel_wide_image_keeps_its_spatial_dimensions(self, mock_lazy_rioxarray):
        """Only the band axis is dropped. A bare squeeze() would also collapse an x or y dimension of
        length 1 -- what a very small ROI leaves -- and the band would lose its grid."""
        result = self._read(mock_lazy_rioxarray, _make_gcp_image(width=1, height=3))
        self.assertEqual(result["B1"].dims, ("y_15m", "x_15m"))
        self.assertEqual(result["B1"].shape, (3, 1))

    @patch("eoio.readers.gistda_theos.data_io.lazy_rioxarray")
    def test_gain_and_bias_keep_a_dask_backed_band_lazy(self, mock_lazy_rioxarray):
        """Applying the radiometric scaling must not load a chunked band into memory."""
        import dask.array as da

        result = self._read(mock_lazy_rioxarray, _make_gcp_image(width=4, height=4).chunk({"x": 2, "y": 2}))
        self.assertIsInstance(result["B1"].data, da.Array)
        self.assertTrue(np.allclose(result["B1"].values, 10 / 2.0 + 1.0))


if __name__ == "__main__":
    unittest.main()
