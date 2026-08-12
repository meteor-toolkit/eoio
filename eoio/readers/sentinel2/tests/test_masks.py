"""eoio.readers.sentinel2.tests.test_masks - tests for eoio.readers.sentinel2.masks"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from eoio.readers.sentinel2.masks import MASK_OPTIONS, MSK_CLASSI_BANDS, add_masks, read_masks

try:
    # Importing rioxarray is what registers the ``.rio`` accessor used by the
    # clipping tests below. Everything else here mocks out the IO entirely, so
    # only those two tests need it -- and rioxarray is an optional "spatial"
    # extra (it pulls in rasterio/GDAL), hence the guard rather than a hard import.
    import rioxarray  # noqa: F401

    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False


def _fake_msk_classi(ny: int = 4, nx: int = 5, crs: str = None) -> xr.DataArray:
    """Build a stand-in for MSK_CLASSI_B00.jp2: a 3-band 0/1 raster.

    Band 1 (opaque clouds) has a single flagged pixel, band 2 (cirrus) has
    two, band 3 (snow/ice) none -- so each layer is distinguishable by its
    sum in the assertions below. Pass *crs* when the test needs the ``.rio``
    accessor to do real work (clipping requires a CRS).
    """
    data = np.zeros((3, ny, nx), dtype=np.uint8)
    data[0, 0, 0] = 1
    data[1, 0, :2] = 1
    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": np.arange(ny, dtype=float), "x": np.arange(nx, dtype=float)},
    )
    if crs is not None:
        da = da.rio.write_crs(crs)
    return da


class TestReadMasks(unittest.TestCase):
    def setUp(self):
        self.layout = MagicMock()
        self.layout.msk_classi_path.return_value = "/tmp/fake_MSK_CLASSI_B00.jp2"

    def _read(self, masks, subset=None, da=None):
        rxr = MagicMock()
        rxr.open_rasterio.return_value = da if da is not None else _fake_msk_classi()
        with patch("eoio.readers.sentinel2.masks.lazy_rioxarray", return_value=rxr):
            return read_masks(ds=xr.Dataset(), masks=masks, layout=self.layout, subset=subset)

    def test_adds_requested_mask_variables(self):
        ds = self._read(["opaque_clouds", "cirrus"])
        self.assertIn("opaque_clouds", ds.data_vars)
        self.assertIn("cirrus", ds.data_vars)
        self.assertNotIn("snow_ice", ds.data_vars)

    def test_selects_the_correct_band_per_mask(self):
        """Regression guard on the 1-based -> 0-based band index conversion: the
        bands carry different numbers of flagged pixels, so picking the wrong
        one is detectable by the sum."""
        ds = self._read(MASK_OPTIONS)
        self.assertEqual(int(ds["opaque_clouds"].sum()), 1)
        self.assertEqual(int(ds["cirrus"].sum()), 2)
        self.assertEqual(int(ds["snow_ice"].sum()), 0)

    def test_masks_are_placed_on_the_60m_grid(self):
        """The classification raster is 60 m while measurement bands are 10/20 m,
        so its variables must get their own dims rather than silently colliding
        with a measurement grid of a different shape."""
        ds = self._read(["opaque_clouds"])
        self.assertEqual(ds["opaque_clouds"].dims, ("y_60m", "x_60m"))

    def test_mask_dtype_is_integral(self):
        ds = self._read(["opaque_clouds"])
        self.assertEqual(ds["opaque_clouds"].dtype, np.uint8)

    def test_flag_attrs_attached(self):
        ds = self._read(["opaque_clouds"])
        self.assertEqual(ds["opaque_clouds"].attrs["flag_values"], [0, 1])
        self.assertIn("cloud", ds["opaque_clouds"].attrs["flag_meanings"])

    def test_missing_mask_file_warns_and_returns_dataset_unchanged(self):
        """Pre-04.00 baselines ship QA60/GML instead of MSK_CLASSI. Requesting
        masks on such a product must not lose the measurement data that has
        already been read into the dataset."""
        self.layout.msk_classi_path.return_value = None
        ds_in = xr.Dataset({"B02": (("y", "x"), np.ones((2, 2)))})
        with patch("eoio.readers.sentinel2.masks.lazy_rioxarray") as mock_lazy:
            with self.assertWarns(UserWarning):
                ds = read_masks(ds=ds_in, masks=["opaque_clouds"], layout=self.layout)
        mock_lazy.assert_not_called()
        self.assertIn("B02", ds.data_vars)
        self.assertNotIn("opaque_clouds", ds.data_vars)

    def test_too_few_bands_warns_and_skips_that_mask(self):
        one_band = _fake_msk_classi()
        one_band = one_band.isel(band=slice(0, 1))
        with self.assertWarns(UserWarning):
            ds = self._read(["snow_ice"], da=one_band)
        self.assertNotIn("snow_ice", ds.data_vars)

    @unittest.skipUnless(HAS_RIOXARRAY, "rioxarray not installed")
    def test_clip_box_applied_when_subset_given(self):
        """A real (CRS-tagged) raster is clipped for real here, rather than
        asserting a mock call -- so this also catches the clip being applied
        to the wrong object or in the wrong coordinate order."""
        da = _fake_msk_classi(ny=8, nx=8, crs="EPSG:32611")
        subset = MagicMock()
        subset.clip_box = (0.0, 0.0, 3.0, 3.0)
        subset.geometries = None
        ds = self._read(["opaque_clouds"], subset=subset, da=da)

        full = da.sizes["y"] * da.sizes["x"]
        clipped = ds["opaque_clouds"].sizes["y_60m"] * ds["opaque_clouds"].sizes["x_60m"]
        self.assertLess(clipped, full)

    @unittest.skipUnless(HAS_RIOXARRAY, "rioxarray not installed")
    def test_clip_box_survives_a_sub_pixel_roi(self):
        """Regression test: a ROI smaller than one 60 m pixel (e.g. a small in-situ
        site box, or the sliver left after clipping to a single MGRS tile near a
        tile boundary) must not raise rioxarray's OneDimensionalRaster -- it should
        still yield a real result rather than dropping the mask for that matchup."""
        da = _fake_msk_classi(ny=8, nx=8, crs="EPSG:32611")
        subset = MagicMock()
        subset.clip_box = (2.0, 2.0, 2.4, 2.4)  # well under one 1.0-unit pixel here
        subset.geometries = None
        ds = self._read(["opaque_clouds"], subset=subset, da=da)

        self.assertGreater(ds["opaque_clouds"].sizes["y_60m"], 0)
        self.assertGreater(ds["opaque_clouds"].sizes["x_60m"], 0)

    @unittest.skipUnless(HAS_RIOXARRAY, "rioxarray not installed")
    def test_no_clip_when_subset_is_none(self):
        da = _fake_msk_classi(ny=8, nx=8, crs="EPSG:32611")
        ds = self._read(["opaque_clouds"], subset=None, da=da)
        self.assertEqual(ds["opaque_clouds"].sizes["y_60m"], 8)
        self.assertEqual(ds["opaque_clouds"].sizes["x_60m"], 8)


class TestAddMasks(unittest.TestCase):
    def setUp(self):
        self.layout = MagicMock()
        self.layout.msk_classi_path.return_value = "/tmp/fake_MSK_CLASSI_B00.jp2"
        self.config = MagicMock()
        self.config.read_params = {"use_chunks": False, "chunks": None}

    def test_rejects_unknown_mask_name(self):
        with self.assertRaises(ValueError) as cm:
            add_masks(ds=xr.Dataset(), masks=["not_a_mask"], layout=self.layout, subset=None, config=self.config)
        self.assertIn("not_a_mask", str(cm.exception))

    def test_passes_valid_masks_through_to_read_masks(self):
        rxr = MagicMock()
        rxr.open_rasterio.return_value = _fake_msk_classi()
        with patch("eoio.readers.sentinel2.masks.lazy_rioxarray", return_value=rxr):
            ds = add_masks(ds=xr.Dataset(), masks=["cirrus"], layout=self.layout, subset=None, config=self.config)
        self.assertIn("cirrus", ds.data_vars)


class TestMaskDefinitions(unittest.TestCase):
    def test_band_indices_are_one_based_and_contiguous(self):
        self.assertEqual(sorted(MSK_CLASSI_BANDS.values()), [1, 2, 3])

    def test_mask_options_matches_band_mapping(self):
        self.assertEqual(MASK_OPTIONS, list(MSK_CLASSI_BANDS))


if __name__ == "__main__":
    unittest.main()
