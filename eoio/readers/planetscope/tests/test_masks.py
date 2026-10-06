"""
eoio.readers.planetscope.tests.test_masks - unit tests for eoio.readers.planetscope.masks
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from eoio.readers.planetscope.layout import PlanetScopeLayout, PlanetScopeLayoutError
from eoio.readers.planetscope.masks import MASK_OPTIONS, UDM2_BANDS, add_masks, read_masks
from eoio.readers.subset.roi_subset import ResolvedROISubset

SCENE = "20250108_092601_62_251b"
X0, Y0, RES, N = 500000.0, 7000000.0, 3.0, 10


def _write_tif(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="uint8",
        crs="EPSG:32733",
        transform=from_origin(X0, Y0, RES, RES),
    ) as dst:
        dst.write(data)


def _subset(geometries, clip_box):
    return ResolvedROISubset(
        geometries=geometries, clip_box=clip_box, roi=None, roi_crs_epsg=32733, image_crs="EPSG:32733"
    )


class TestPlanetScopeMasks(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

        # image file only needs to exist for the layout; masks are read from the udm2 tif
        self.image = self.dir / f"{SCENE}_3B_AnalyticMS_8b_clip.tif"
        _write_tif(self.image, np.ones((8, N, N), dtype="uint8"))

        # clear everywhere except one cloudy row (y == 0); confidence 90; nothing unusable
        udm2 = np.zeros((8, N, N), dtype="uint8")
        udm2[UDM2_BANDS["clear"] - 1] = 1
        udm2[UDM2_BANDS["clear"] - 1, 0, :] = 0
        udm2[UDM2_BANDS["cloud"] - 1, 0, :] = 1
        udm2[UDM2_BANDS["confidence"] - 1] = 90
        self.udm2_path = self.dir / f"{SCENE}_3B_udm2_clip.tif"
        _write_tif(self.udm2_path, udm2)

        self.layout = PlanetScopeLayout(str(self.image))
        self.config = SimpleNamespace(read_params={})

    def test_mask_options_follow_udm2_band_order(self):
        self.assertEqual(
            MASK_OPTIONS, ["clear", "snow", "shadow", "haze_light", "haze_heavy", "cloud", "confidence", "unusable"]
        )

    def test_reads_requested_layers_on_the_3m_grid(self):
        ds = read_masks(ds=xr.Dataset(), masks=["clear", "cloud"], layout=self.layout)
        self.assertEqual(set(ds.data_vars), {"clear", "cloud"})
        self.assertEqual(ds["clear"].dims, ("y_3m", "x_3m"))
        self.assertEqual(ds["clear"].shape, (N, N))
        self.assertEqual(ds["clear"].values[0].sum(), 0)
        self.assertEqual(ds["clear"].values[1:].sum(), N * (N - 1))
        self.assertEqual(ds["cloud"].values[0].sum(), N)

    def test_confidence_keeps_its_values(self):
        ds = read_masks(ds=xr.Dataset(), masks=["confidence"], layout=self.layout)
        self.assertTrue((ds["confidence"].values == 90).all())

    def test_outside_roi_polygon_is_nan_not_zero(self):
        # right triangle covering the lower-left half of the first 6x6 pixels -- outside the
        # polygon must not read as "not clear" (0)
        x_min, y_max = X0 + 3 * RES, Y0 - 3 * RES
        size = 6 * RES
        tri = {
            "type": "Polygon",
            "coordinates": [[(x_min, y_max), (x_min + size, y_max), (x_min, y_max - size), (x_min, y_max)]],
        }
        subset = _subset([tri], (x_min, y_max - size, x_min + size, y_max))
        ds = read_masks(ds=xr.Dataset(), masks=["clear"], layout=self.layout, subset=subset)
        clear = ds["clear"].values
        self.assertTrue(np.isnan(clear).any())
        self.assertTrue((clear[~np.isnan(clear)] == 1).all())

    def test_box_only_subset_is_clipped(self):
        subset = _subset(None, (X0, Y0 - 5 * RES, X0 + 5 * RES, Y0))
        ds = read_masks(ds=xr.Dataset(), masks=["clear"], layout=self.layout, subset=subset)
        self.assertEqual(ds["clear"].shape, (5, 5))

    def test_add_masks_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            add_masks(ds=xr.Dataset(), masks=["not_a_mask"], layout=self.layout, subset=None, config=self.config)

    def test_add_masks_reads(self):
        ds = add_masks(ds=xr.Dataset(), masks=["clear"], layout=self.layout, subset=None, config=self.config)
        self.assertIn("clear", ds)

    def test_udm2_with_too_few_bands_raises(self):
        _write_tif(self.udm2_path, np.ones((7, N, N), dtype="uint8"))
        with self.assertRaises(ValueError):
            read_masks(ds=xr.Dataset(), masks=["unusable"], layout=self.layout)

    def test_missing_udm2_file_raises(self):
        self.udm2_path.unlink()
        with self.assertRaises(PlanetScopeLayoutError):
            read_masks(ds=xr.Dataset(), masks=["clear"], layout=self.layout)


if __name__ == "__main__":
    unittest.main()
