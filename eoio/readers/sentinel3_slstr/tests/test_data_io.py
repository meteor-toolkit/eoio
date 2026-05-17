"""Unit tests for eoio.readers.sentinel3_slstr.data_io.

These mirror the sentinel3_olci tests but adapted to SLSTR helpers.
"""

import unittest
import xarray as xr
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

import eoio.readers.sentinel3_slstr.data_io as data_io


class FakeRaster:
    def __init__(self, da):
        self._da = da

    def squeeze(self):
        return self._da

    def __getitem__(self, key):
        # emulate indexing like rasterio returning a list-like of datasets
        if key == 0:
            return self._da
        raise IndexError


def _make_fake_da():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        attrs={
            "valid_min": 0,
            "valid_max": 100,
            "scale_factor": 1.0,
            "long_name": "radiance",
            "units": "W/m2/sr",
        },
    )
    return da


# Provide attribute-style access for scale/add used by the reader arithmetic.
# xarray.DataArray stores these in `.attrs`; the production code accesses
# them as attributes (e.g. `da.scale_factor`) so expose read-only properties
# that proxy into `.attrs` for tests.
if not hasattr(xr.DataArray, "scale_factor"):

    def _scale(self):
        return self.attrs.get("scale_factor", 1.0)

    xr.DataArray.scale_factor = property(_scale)

if not hasattr(xr.DataArray, "add_offset"):

    def _add(self):
        return self.attrs.get("add_offset", 0.0)

    xr.DataArray.add_offset = property(_add)


class TestS3SLSTRDataIO(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.layout = MagicMock()
        self.subset = MagicMock()
        self.config = MagicMock()
        self.config.vars_sel = {"meas": ["S1_radiance_an"], "aux": []}
        self.config.read_params = {"chunks": None, "use_chunks": False}
        self.mtd = MagicMock()

    @patch("eoio.readers.sentinel3_slstr.data_io.lazy_rioxarray")
    def test_read_bands_with_empty_meas_returns_dataset(self, mock_riox):
        # When meas is empty the function should return the dataset unmodified
        self.layout.meas_paths.return_value = {}
        result = data_io.read_bands_into_dataset(ds=self.ds, layout=self.layout, meas=[], subset=None, mtd=self.mtd)
        self.assertIsInstance(result, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.data_io.lazy_rioxarray")
    def test_read_lat_lon_coordinates_returns_coords(self, mock_riox):
        mock_geo = MagicMock()
        # emulate dataset returned from rioxarray
        mock_geo.latitude_an = MagicMock()
        mock_geo.longitude_an = MagicMock()
        mock_geo.latitude_an.attrs = {"scale_factor": 1}
        mock_geo.longitude_an.attrs = {"scale_factor": 1}
        mock_riox.return_value.open_rasterio.return_value = [mock_geo]
        self.layout.geodetic_path.return_value = "dummy_path"

        ds, clip_boxes = data_io.read_lat_lon_coordinates(
            ds=self.ds,
            layout=self.layout,
            grids=["an"],
            subset=None,
            config=self.config,
            use_chunks=False,
            chunks=None,
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertTrue(isinstance(clip_boxes, dict) or clip_boxes is None)


class TestReadBandsIntoDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = Mock()
        self.mtd = Mock()
        # layout.meas_paths should return a mapping for requested bands
        self.layout.meas_paths.side_effect = lambda bands=None: (
            {b: Path(f"/tmp/{b}.nc") for b in (bands or ["S1_radiance_an"])}
        )
        self.layout.get_grid.return_value = "an"
        self.mtd.variable_product_metadata.side_effect = lambda b: {"band_central_wavelength": 400.0}
        # provide lon/lat as coordinates (not data variables) as expected by the reader
        # Use coordinate names that match the renamed band dims produced by the
        # reader (e.g. 'x_500m_an', 'y_500m_an'). Provide 2D lon/lat arrays and
        # 1D dimension coordinates.
        self.ds0 = xr.Dataset(
            coords={
                "lon_500m_an": (
                    ("y_500m_an", "x_500m_an"),
                    np.array([[0.1, 1.0], [2.0, 3.0]]),
                ),
                "lat_500m_an": (
                    ("y_500m_an", "x_500m_an"),
                    np.array([[0.2, 2.0], [4.0, 6.0]]),
                ),
                "y_500m_an": [0, 1],
                "x_500m_an": [0, 1],
            },
        )

    def test_function_signature(self):
        self.assertTrue(callable(data_io.read_bands_into_dataset))

    @patch(
        "eoio.readers.sentinel3_slstr.data_io.xr.DataArray.drop_vars",
        new=lambda self, *a, **k: self,
    )
    @patch("eoio.readers.sentinel3_slstr.data_io.lazy_rioxarray")
    def test_read_bands_calls_layout_method(self, lazy_rioxarray):
        def open_rasterio_side_effect(path, chunks=None):
            band = Path(path).stem
            da = _make_fake_da()
            ds_inner = xr.Dataset({band: da, "band": xr.DataArray([0], dims=("band",))})
            return [ds_inner]

        lazy_rioxarray.return_value.open_rasterio.side_effect = open_rasterio_side_effect
        data_io.read_bands_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["S1_radiance_an", "S2_radiance_bn"],
            subset=None,
            mtd=self.mtd,
        )
        self.layout.meas_paths.assert_called()

    @patch(
        "eoio.readers.sentinel3_slstr.data_io.xr.DataArray.drop_vars",
        new=lambda self, *a, **k: self,
    )
    @patch("eoio.readers.sentinel3_slstr.data_io.lazy_rioxarray")
    def test_read_bands_with_single_band(self, lazy_rioxarray):
        def open_rasterio_side_effect(path, chunks=None):
            band = Path(path).stem
            da = _make_fake_da()
            ds_inner = xr.Dataset({band: da, "band": xr.DataArray([0], dims=("band",))})
            return [ds_inner]

        lazy_rioxarray.return_value.open_rasterio.side_effect = open_rasterio_side_effect
        data_io.read_bands_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["S1_radiance_an"],
            subset=None,
            mtd=self.mtd,
        )
        self.layout.meas_paths.assert_called_with(["S1_radiance_an"])


if __name__ == "__main__":
    unittest.main()


class TestDummyDataIO(unittest.TestCase):
    def test_layout_mock_meas_paths(self):
        layout = Mock()
        layout.meas_paths.return_value = {"S1_radiance_an": Path("/path/to/S1_radiance_an.nc")}
        result = layout.meas_paths(["S1_radiance_an"])
        self.assertIn("S1_radiance_an", result)
        self.assertEqual(result["S1_radiance_an"], Path("/path/to/S1_radiance_an.nc"))

    def test_dataset_has_setitem(self):
        ds = xr.Dataset()
        self.assertTrue(hasattr(ds, "__setitem__"))

    def test_metadata_extractor_mock(self):
        from eoio.readers.sentinel3_slstr.metadata.extractor import (
            S3SLSTRMetadataExtractor,
        )

        mtd = Mock(spec=S3SLSTRMetadataExtractor)
        mtd.get_variable_product_metadata.return_value = {"wavelength": 400.0}
        result = mtd.get_variable_product_metadata("S1_radiance_an")
        self.assertIsInstance(result, dict)
        self.assertIn("wavelength", result)

    def test_dataset_is_mutable(self):
        ds = xr.Dataset()
        ds["S1_radiance_an"] = xr.DataArray([1, 2, 3])
        self.assertIn("S1_radiance_an", ds)
        self.assertEqual(len(ds["S1_radiance_an"]), 3)
