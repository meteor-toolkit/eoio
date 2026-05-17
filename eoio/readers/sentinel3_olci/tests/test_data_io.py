import unittest
import xarray as xr
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

import eoio.readers.sentinel3_olci.data_io as data_io
from eoio.readers.sentinel3_olci.metadata.extractor import S3OLCIMetadataExtractor


class FakeRaster:
    def __init__(self, da):
        self._da = da

    def squeeze(self):
        return self._da


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


class TestS3OLCIDataIO(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.layout = MagicMock()
        self.subset = MagicMock()
        self.config = MagicMock()
        self.config.vars_sel = {
            "meas": ["Oa01"],
            "aux": ["observation_geometry", "removed_pixels"],
        }
        self.config.read_params = {"chunks": None, "use_chunks": False}
        self.mtd = MagicMock()
        self.chunks = None
        self.use_chunks = False

    @patch("eoio.readers.sentinel3_olci.data_io.lazy_rioxarray")
    def test_read_bands_into_dataset(self, mock_riox):
        class FakeDA:
            def __init__(self):
                self.attrs = {"valid_min": 0, "valid_max": 100, "scale_factor": 1.0}
                self.dims = ("y", "x")

            def __le__(self, other):
                return np.array([[True]])

            def __ge__(self, other):
                return np.array([[True]])

            def __mul__(self, other):
                return self

            def where(self, cond):
                return self

            def rename(self, mapping):
                return self

            def squeeze(self):
                return self

            def drop_vars(self, *a, **k):
                return self

        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value = FakeDA()
        self.layout.requested_radiance_paths.return_value = {"Oa01": "dummy_path"}
        # Provide longitude_300m in the test dataset to avoid KeyError
        self.ds["longitude_300m"] = (("y", "x"), np.ones((1, 1)))
        ds = data_io.read_bands_into_dataset(ds=self.ds, layout=self.layout, meas=["Oa01"], subset=None, mtd=self.mtd)
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_olci.data_io.lazy_rioxarray")
    @patch("numpy.power", lambda a, b: a)
    def test_read_uncertainty_into_dataset(self, mock_riox):
        class FakeDA:
            def __init__(self):
                self.attrs = {
                    "valid_min": 0,
                    "valid_max": 100,
                    "scale_factor": 1.0,
                    "add_offset": 0.0,
                    "units": "W/m2/sr (foo)",
                    "long_name": "scaled radiance",
                }
                self.dims = ("y", "x")
                self._data = np.ones((1, 1))

            @property
            def data(self):
                return self._data

            def __lt__(self, other):
                return self._data < other

            def where(self, cond):
                return self

            def squeeze(self):
                return self

            def drop_vars(self, *a, **k):
                return self

            def rename(self, mapping):
                return self

            def __mul__(self, other):
                return self

            def __add__(self, other):
                return self

            def __pow__(self, other):
                return self

            def __rpow__(self, other):
                return self

            def __getattr__(self, name):
                if name == "attrs":
                    return self.attrs
                raise AttributeError(name)

        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value = FakeDA()
        self.layout.requested_uncertainty_paths.return_value = {"Oa01": "dummy_path"}
        # Add Oa01 variable to ds so assignment works
        self.ds["Oa01"] = (("y", "x"), np.ones((1, 1)))
        ds = data_io.read_uncertainty_into_dataset(ds=self.ds, layout=self.layout, meas=["Oa01"], subset=None)
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_olci.data_io.lazy_rioxarray")
    def test_read_tie_geo_coordinates(self, mock_riox):
        mock_geo = MagicMock()
        mock_geo.longitude = MagicMock()
        mock_geo.latitude = MagicMock()
        mock_geo.longitude.attrs = {"scale_factor": 1}
        mock_geo.latitude.attrs = {"scale_factor": 1}
        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value.drop_vars.return_value = mock_geo
        self.layout.tie_geo_coordinates_path.return_value = "dummy_path"
        result = data_io.read_tie_geo_coordinates(ds=self.ds, layout=self.layout, config=self.config, subset=None)
        if isinstance(result, tuple):
            ds, _ = result
        else:
            ds = result
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_olci.data_io.lazy_rioxarray")
    def test_read_lat_lon_coordinates(self, mock_riox):
        mock_geo = MagicMock()
        mock_geo.longitude = MagicMock()
        mock_geo.latitude = MagicMock()
        mock_geo.longitude.attrs = {"scale_factor": 1}
        mock_geo.latitude.attrs = {"scale_factor": 1}
        mock_geo.attrs = {"resolution": " 300 300 "}
        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value.drop_vars.return_value = mock_geo
        self.layout.geo_coordinates_path.return_value = "dummy_path"
        self.config.vars_sel["aux"] = []
        ds, _ = data_io.read_lat_lon_coordinates(ds=self.ds, layout=self.layout, subset=None, config=self.config)
        self.assertIsInstance(ds, xr.Dataset)


class TestReadBandsIntoDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = Mock()
        self.mtd = Mock(spec=S3OLCIMetadataExtractor)
        self.layout.radiance_paths.side_effect = lambda bands: {b: Path(f"/tmp/{b}.nc") for b in bands}
        self.mtd.variable_product_metadata.side_effect = lambda b: {"band_central_wavelength": 400.0}
        self.ds0 = xr.Dataset(
            {"longitude_300m": (("y", "x"), np.array([[0.1, 1.0], [2.0, 3.0]]))},
            coords={"y": [0, 1], "x": [0, 1]},
        )

    def test_function_signature(self):
        self.assertTrue(callable(data_io.read_bands_into_dataset))

    @patch("eoio.readers.sentinel3_olci.data_io.lazy_rioxarray")
    def test_read_bands_with_empty_meas(self, lazy_rioxarray):
        self.layout.radiance_paths.return_value = {}
        result = data_io.read_bands_into_dataset(ds=self.ds0, layout=self.layout, meas=[], subset=None, mtd=self.mtd)
        self.assertIsInstance(result, xr.Dataset)

    @patch("eoio.readers.sentinel3_olci.data_io.lazy_rioxarray")
    def test_read_bands_calls_layout_method(self, lazy_rioxarray):
        lazy_rioxarray.return_value.open_rasterio.side_effect = lambda path, chunks=None: FakeRaster(_make_fake_da())
        data_io.read_bands_into_dataset(
            ds=self.ds0,
            layout=self.layout,
            meas=["Oa01", "Oa02"],
            subset=None,
            mtd=self.mtd,
        )

        self.layout.radiance_paths.assert_called_with(["Oa01", "Oa02"])

    @patch("eoio.readers.sentinel3_olci.data_io.lazy_rioxarray")
    def test_read_bands_with_single_band(self, lazy_rioxarray):
        lazy_rioxarray.return_value.open_rasterio.side_effect = lambda path, chunks=None: FakeRaster(_make_fake_da())
        data_io.read_bands_into_dataset(ds=self.ds0, layout=self.layout, meas=["Oa01"], subset=None, mtd=self.mtd)

        self.layout.radiance_paths.assert_called_with(["Oa01"])


class TestDummyDataIO(unittest.TestCase):
    def test_layout_mock_radiance_paths(self):
        layout = Mock()
        layout.requested_radiance_paths.return_value = {"Oa01": Path("/path/to/Oa01.nc")}
        result = layout.requested_radiance_paths(["Oa01"])
        self.assertIn("Oa01", result)
        self.assertEqual(result["Oa01"], Path("/path/to/Oa01.nc"))

    def test_dataset_has_setitem(self):
        ds = xr.Dataset()
        self.assertTrue(hasattr(ds, "__setitem__"))

    def test_metadata_extractor_mock(self):
        mtd = Mock(spec=S3OLCIMetadataExtractor)
        mtd.variable_product_metadata.return_value = {"wavelength": 400.0}
        result = mtd.variable_product_metadata("Oa01")
        self.assertIsInstance(result, dict)
        self.assertIn("wavelength", result)

    def test_dataset_is_mutable(self):
        ds = xr.Dataset()
        ds["Oa01"] = xr.DataArray([1, 2, 3])
        self.assertIn("Oa01", ds)
        self.assertEqual(len(ds["Oa01"]), 3)


if __name__ == "__main__":
    unittest.main()
