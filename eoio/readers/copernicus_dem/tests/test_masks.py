import unittest
from unittest.mock import Mock, patch

import xarray as xr

from eoio.readers.copernicus_dem.masks import (
    add_masks,
    read_masks,
)


class TestAddMasks(unittest.TestCase):
    @patch("eoio.readers.copernicus_dem.masks.read_masks")
    def test_add_masks_calls_read_masks(self, mock_read_masks):
        ds = xr.Dataset()

        config = Mock()
        config.vars_sel = {"mask": ["water_body_mask"]}
        config.read_params = {
            "use_chunks": True,
            "chunks": {"x": 256, "y": 256},
        }

        layout = Mock()
        subset = Mock()

        expected_ds = xr.Dataset()
        mock_read_masks.return_value = expected_ds

        result = add_masks(
            ds=ds,
            masks=["water_body_mask"],
            layout=layout,
            subset=subset,
            config=config,
        )

        mock_read_masks.assert_called_once_with(
            ds=ds,
            masks=["water_body_mask"],
            layout=layout,
            subset=subset,
            config=config,
            use_chunks=True,
            chunks={"x": 256, "y": 256},
        )

        self.assertIs(result, expected_ds)

    def test_add_masks_raises_when_no_masks_selected(self):
        ds = xr.Dataset()

        config = Mock()
        config.vars_sel = {"mask": None}
        config.read_params = {}

        with self.assertRaises(ValueError):
            add_masks(
                ds=ds,
                masks=["water_body_mask"],
                layout=Mock(),
                subset=None,
                config=config,
            )


class TestReadMasks(unittest.TestCase):
    @patch("eoio.readers.copernicus_dem.masks.lazy_rioxarray")
    def test_read_masks_returns_dataset_when_masks_empty(
        self,
        mock_lazy_rioxarray,
    ):
        ds = xr.Dataset()

        result = read_masks(
            ds=ds,
            masks=[],
            layout=Mock(),
            config=Mock(),
        )

        self.assertIs(result, ds)
        mock_lazy_rioxarray.assert_called_once()

    @patch("eoio.readers.copernicus_dem.masks.lazy_rioxarray")
    def test_read_masks_reads_mask(
        self,
        mock_lazy_rioxarray,
    ):
        ds = xr.Dataset()

        da = xr.DataArray([[1]], dims=("y", "x"))

        rxr = Mock()
        rxr.open_rasterio.return_value.squeeze.return_value = da

        mock_lazy_rioxarray.return_value = rxr

        layout = Mock()
        layout.wbm = "/tmp/wbm.tif"
        layout.flm = "/tmp/flm.tif"
        layout.edm = "/tmp/edm.tif"
        layout.hem = "/tmp/hem.tif"

        result = read_masks(
            ds=ds,
            masks=["water_body_mask"],
            layout=layout,
            config=Mock(),
        )

        rxr.open_rasterio.assert_called_once_with(
            "/tmp/wbm.tif",
            chunks=None,
        )

        self.assertIn("water_body_mask", result.data_vars)

    @patch("eoio.readers.copernicus_dem.masks.suggest_raster_chunks")
    @patch("eoio.readers.copernicus_dem.masks.lazy_rioxarray")
    def test_read_masks_computes_chunks(
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

        layout = Mock()
        layout.wbm = "/tmp/wbm.tif"
        layout.flm = "/tmp/flm.tif"
        layout.edm = "/tmp/edm.tif"
        layout.hem = "/tmp/hem.tif"

        read_masks(
            ds=ds,
            masks=["water_body_mask"],
            layout=layout,
            config=Mock(),
            use_chunks=True,
        )

        mock_suggest_chunks.assert_called_once_with(
            "/tmp/wbm.tif",
            target_mb=32.0,
        )

        rxr.open_rasterio.assert_called_once_with(
            "/tmp/wbm.tif",
            chunks={"x": 512, "y": 512},
        )

    @patch("eoio.readers.copernicus_dem.masks.lazy_rioxarray")
    def test_read_masks_uses_provided_chunks(
        self,
        mock_lazy_rioxarray,
    ):
        ds = xr.Dataset()

        da = xr.DataArray([[1]], dims=("y", "x"))

        rxr = Mock()
        rxr.open_rasterio.return_value.squeeze.return_value = da

        mock_lazy_rioxarray.return_value = rxr

        layout = Mock()
        layout.wbm = "/tmp/wbm.tif"
        layout.flm = "/tmp/flm.tif"
        layout.edm = "/tmp/edm.tif"
        layout.hem = "/tmp/hem.tif"

        read_masks(
            ds=ds,
            masks=["water_body_mask"],
            layout=layout,
            config=Mock(),
            use_chunks=True,
            chunks={"x": 128, "y": 128},
        )

        rxr.open_rasterio.assert_called_once_with(
            "/tmp/wbm.tif",
            chunks={"x": 128, "y": 128},
        )

    @patch("eoio.readers.copernicus_dem.masks.lazy_rioxarray")
    def test_read_masks_applies_subset_clip(
        self,
        mock_lazy_rioxarray,
    ):
        ds = xr.Dataset()

        clipped_da = xr.DataArray([[1]], dims=("y", "x"))

        rio_after_write = Mock()
        rio_after_write.rio.clip.return_value = clipped_da

        rio_mock = Mock()
        rio_mock.write_crs.return_value = rio_after_write

        da = Mock()
        da.rio = rio_mock

        rxr = Mock()
        rxr.open_rasterio.return_value.squeeze.return_value = da

        mock_lazy_rioxarray.return_value = rxr

        subset = Mock()
        subset.geometries = ["geom"]

        layout = Mock()
        layout.wbm = "/tmp/wbm.tif"
        layout.flm = "/tmp/flm.tif"
        layout.edm = "/tmp/edm.tif"
        layout.hem = "/tmp/hem.tif"

        result = read_masks(
            ds=ds,
            masks=["water_body_mask"],
            layout=layout,
            subset=subset,
            config=Mock(),
        )

        rio_mock.write_crs.assert_called_once_with(4326)
        rio_after_write.rio.clip.assert_called_once_with(["geom"])

        self.assertIn("water_body_mask", result.data_vars)


if __name__ == "__main__":
    unittest.main()
