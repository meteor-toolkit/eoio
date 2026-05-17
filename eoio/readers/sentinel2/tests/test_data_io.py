"""eoio.readers.sentinel2.test_data_io - tests for eoio.readers.sentinel2.data_io"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from eoio.readers.sentinel2.data_io import read_bands_into_dataset


class TestReadBandsIntoDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Mock layout + metadata extractor
        self.layout = MagicMock()
        self.mtd = MagicMock()

        # Two dummy band paths
        self.band_paths = {
            "B02": "/tmp/fake_B02.jp2",
            "B03": "/tmp/fake_B03.jp2",
            "AOT": "/tmp/fake_AOT.jp2",
            "WVP": "/tmp/fake_WVP.jp2",
            "SCL": "/tmp/fake_SCL.jp2",
            "TCI": "/tmp/fake_TCI.jp2",
            "FUTURE_VAR": "/tmp/fake_FUTURE_VAR.jp2",
        }

        def _img_jp2_paths(meas_vars, prefer_res_m=None):
            return {b: self.band_paths[b] for b in meas_vars}

        self.layout.img_jp2_paths.side_effect = _img_jp2_paths

        # Default product metadata (new API: attribute, not method)
        self.mtd.product_metadata = {"quantification_level": {"reflectance": 10000}}

        # Default per-band metadata: MUST include radiometric_offset for new code
        def _var_mtd(band: str):
            return {"radiometric_offset": 0.0}

        self.mtd.variable_product_metadata.side_effect = _var_mtd

        # Default subset: none
        self.subset_none = None

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
        # Pretend rioxarray attached attrs on coords (as can happen after clipping)
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

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_raises_on_invalid_quantification_level(self, mock_lazy_rioxarray):
        self.mtd.product_metadata = {"quantification_level": 0}
        with self.assertRaises(ValueError):
            read_bands_into_dataset(
                ds=self.ds0,
                layout=self.layout,
                meas_vars=["B02"],
                subset=None,
                mtd=self.mtd,
            )

        self.mtd.product_metadata = {}
        with self.assertRaises(ValueError):
            read_bands_into_dataset(
                ds=self.ds0,
                layout=self.layout,
                meas_vars=["B02"],
                subset=None,
                mtd=self.mtd,
            )

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_reads_bands_applies_offset_scaling_and_masks_zero(self, mock_lazy_rioxarray):
        # Default radiometric_offset=0.0 via setUp.
        # B02 has a zero that should become NaN; B03 all non-zero
        da_b02 = self._make_da(np.array([[[0, 10000], [20000, 30000]]], dtype=np.uint16))
        da_b03 = self._make_da(np.array([[[10000, 20000], [30000, 40000]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.side_effect = [da_b02, da_b03]
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["B02", "B03"],
            subset=None,
            mtd=self.mtd,
            mask_zero_as_nodata=True,
        )

        self.assertIn("B02", ds.data_vars)
        self.assertIn("B03", ds.data_vars)

        # band dim dropped
        self.assertNotIn("band", ds["B02"].dims)

        # scaling: (raw + offset)/10000 -> float32; zero masked
        self.assertEqual(ds["B02"].dtype, np.dtype("float32"))
        self.assertTrue(np.isnan(ds["B02"].values[0, 0]))  # zero masked
        self.assertAlmostEqual(float(ds["B02"].values[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(ds["B02"].values[1, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(ds["B03"].values[1, 1]), 4.0, places=6)

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_applies_radiometric_offset(self, mock_lazy_rioxarray):
        # Make offset non-zero so we can validate it is applied before scaling.
        def _var_mtd(band: str):
            return {"radiometric_offset": 100.0}

        self.mtd.variable_product_metadata.side_effect = _var_mtd
        self.mtd.product_metadata = {"quantification_level": {"reflectance": 10000}}

        # raw=10000 -> (10000 + 100)/10000 = 1.01
        da = self._make_da(np.array([[[10000, 0]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["B02"],
            subset=None,
            mtd=self.mtd,
            mask_zero_as_nodata=False,
        )

        self.assertAlmostEqual(float(ds["B02"].values[0, 0]), 1.01, places=6)
        self.assertAlmostEqual(float(ds["B02"].values[0, 1]), 0.01, places=6)

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_does_not_mask_zero_when_disabled(self, mock_lazy_rioxarray):
        da = self._make_da(np.array([[[0, 10000]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["B02"],
            subset=None,
            mtd=self.mtd,
            mask_zero_as_nodata=False,
        )

        # zero should remain zero after (raw+offset)/ql (offset defaults to 0.0)
        self.assertEqual(float(ds["B02"].values[0, 0]), 0.0)

    @patch("eoio.readers.sentinel2.data_io.suggest_raster_chunks")
    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_use_chunks_calls_suggest_and_passes_chunks(self, mock_lazy_rioxarray, mock_suggest):
        mock_suggest.return_value = {"x": 256, "y": 256}

        da_b02 = self._make_da(np.array([[[10000]]], dtype=np.uint16))
        da_b03 = self._make_da(np.array([[[20000]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.side_effect = [da_b02, da_b03]
        mock_lazy_rioxarray.return_value = rxr

        _ = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["B02", "B03"],
            subset=None,
            mtd=self.mtd,
            use_chunks=True,
            chunks=None,
        )

        # called once (from first path)
        mock_suggest.assert_called_once()
        # open_rasterio called with derived chunks for both bands
        calls = rxr.open_rasterio.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs.get("chunks"), {"x": 256, "y": 256})
        self.assertEqual(calls[1].kwargs.get("chunks"), {"x": 256, "y": 256})

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_passes_preferred_resolution_to_layout(self, mock_lazy_rioxarray):
        da = self._make_da(np.array([[[10000]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        _ = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["B02"],
            subset=None,
            mtd=self.mtd,
            preferred_resolution=20,
        )

        self.layout.img_jp2_paths.assert_called_once_with(["B02"], prefer_res_m=20)

    @patch("eoio.readers.sentinel2.data_io.suggest_raster_chunks")
    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_explicit_chunks_override_suggest(self, mock_lazy_rioxarray, mock_suggest):
        da_b02 = self._make_da(np.array([[[10000]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da_b02
        mock_lazy_rioxarray.return_value = rxr

        _ = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["B02"],
            subset=None,
            mtd=self.mtd,
            use_chunks=True,
            chunks={"x": 1024, "y": 1024},
        )

        # suggest should not be called if chunks explicitly provided
        mock_suggest.assert_not_called()
        rxr.open_rasterio.assert_called_once()
        self.assertEqual(rxr.open_rasterio.call_args.kwargs.get("chunks"), {"x": 1024, "y": 1024})

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_decodes_aot_with_quantification(self, mock_lazy_rioxarray):
        """AOT should be scaled by quantification value when present."""
        self.mtd.product_metadata = {"quantification_level": {"aot": 1000}}

        # AOT raw values: 100, 500
        da = self._make_da(np.array([[[100, 500]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["AOT"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("AOT", ds.data_vars)
        self.assertEqual(ds["AOT"].dtype, np.dtype("float32"))
        # 100 / 1000 = 0.1, 500 / 1000 = 0.5
        self.assertAlmostEqual(float(ds["AOT"].values[0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(ds["AOT"].values[0, 1]), 0.5, places=6)

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_decodes_aot_without_quantification(self, mock_lazy_rioxarray):
        """AOT should not be scaled if quantification is missing or 0."""
        self.mtd.product_metadata = {"quantification_level": {}}

        # AOT raw values
        da = self._make_da(np.array([[[100, 200]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["AOT"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("AOT", ds.data_vars)
        self.assertEqual(ds["AOT"].dtype, np.dtype("float32"))
        # No scaling applied
        self.assertEqual(float(ds["AOT"].values[0, 0]), 100.0)
        self.assertEqual(float(ds["AOT"].values[0, 1]), 200.0)

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_decodes_wvp_with_quantification(self, mock_lazy_rioxarray):
        """WVP should be scaled by quantification value when present."""
        self.mtd.product_metadata = {"quantification_level": {"wvp": 500}}

        # WVP raw values: 250, 750
        da = self._make_da(np.array([[[250, 750]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["WVP"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("WVP", ds.data_vars)
        self.assertEqual(ds["WVP"].dtype, np.dtype("float32"))
        # 250 / 500 = 0.5, 750 / 500 = 1.5
        self.assertAlmostEqual(float(ds["WVP"].values[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(ds["WVP"].values[0, 1]), 1.5, places=6)

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_decodes_wvp_without_quantification(self, mock_lazy_rioxarray):
        """WVP should not be scaled if quantification is missing or 0."""
        self.mtd.product_metadata = {"quantification_level": {"wvp": 0}}

        # WVP raw values
        da = self._make_da(np.array([[[100, 200]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["WVP"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("WVP", ds.data_vars)
        self.assertEqual(ds["WVP"].dtype, np.dtype("float32"))
        # No scaling applied (q=0)
        self.assertEqual(float(ds["WVP"].values[0, 0]), 100.0)
        self.assertEqual(float(ds["WVP"].values[0, 1]), 200.0)

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_decodes_scl_as_int16(self, mock_lazy_rioxarray):
        """SCL should be converted to int16 (categorical classification layer)."""
        self.mtd.product_metadata = {"quantification_level": {"reflectance": 10000}}

        # SCL categorical values: 1, 2, 3, 4 (nodata, saturated, dark, vegetation, etc.)
        da = self._make_da(np.array([[[1, 2], [3, 4]]], dtype=np.uint8))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["SCL"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("SCL", ds.data_vars)
        # Verify dtype is int16
        self.assertEqual(ds["SCL"].dtype, np.dtype("int16"))
        # Values should be preserved
        self.assertEqual(int(ds["SCL"].values[0, 0]), 1)
        self.assertEqual(int(ds["SCL"].values[0, 1]), 2)
        self.assertEqual(int(ds["SCL"].values[1, 0]), 3)
        self.assertEqual(int(ds["SCL"].values[1, 1]), 4)

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_decodes_tci_rgb_three_band(self, mock_lazy_rioxarray):
        """TCI should keep 3-band RGB data as uint (not scaled)."""
        self.mtd.product_metadata = {"quantification_level": {"reflectance": 10000}}

        # TCI has 3 bands (R, G, B)
        da = xr.DataArray(
            np.array(
                [
                    [[255, 200], [150, 100]],  # R band
                    [[200, 180], [160, 140]],  # G band
                    [[180, 160], [140, 120]],  # B band
                ],
                dtype=np.uint8,
            ),
            dims=("band", "y", "x"),
            coords={
                "band": [1, 2, 3],
                "y": np.arange(2),
                "x": np.arange(2),
            },
        )
        da.coords["x"].attrs = {}
        da.coords["y"].attrs = {}

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["TCI"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("TCI", ds.data_vars)
        # Should have rgb dimension instead of band
        self.assertIn("rgb", ds["TCI"].dims)
        self.assertNotIn("band", ds["TCI"].dims)
        # Data type should be uint8 (not converted to float32)
        self.assertEqual(ds["TCI"].dtype, np.dtype("uint8"))
        # RGB coords should be set correctly
        np.testing.assert_array_equal(ds["TCI"].coords["rgb"].values, ["R", "G", "B"])

    @patch("eoio.readers.sentinel2.data_io.lazy_rioxarray")
    def test_decodes_unknown_layer_unchanged(self, mock_lazy_rioxarray):
        """Unknown/future layer types should be left as read (no conversion)."""
        self.mtd.product_metadata = {"quantification_level": {"reflectance": 10000}}

        # Mock a hypothetical future variable with data as uint32
        da = self._make_da(np.array([[[1000, 2000]]], dtype=np.uint32))
        # Remove band dimension to simulate single-band unknown layer
        da = da.isel(band=0, drop=True)

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = read_bands_into_dataset(
            ds=xr.Dataset(),
            layout=self.layout,
            meas_vars=["FUTURE_VAR"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("FUTURE_VAR", ds.data_vars)
        # Should keep original dtype (uint32)
        self.assertEqual(ds["FUTURE_VAR"].dtype, np.dtype("uint32"))
        # Values should be unchanged
        self.assertEqual(int(ds["FUTURE_VAR"].values[0, 0]), 1000)
        self.assertEqual(int(ds["FUTURE_VAR"].values[0, 1]), 2000)


if __name__ == "__main__":
    unittest.main()
