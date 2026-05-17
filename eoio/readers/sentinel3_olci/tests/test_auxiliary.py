import unittest
import xarray as xr
from unittest.mock import MagicMock, patch
import numpy as np

from eoio.readers.sentinel3_olci import auxiliary


class TestS3OLCIAuxiliary(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.layout = MagicMock()
        self.config = MagicMock()
        self.config.vars_sel = {
            "aux": ["observation_geometry", "removed_pixels"],
            "meas": ["Oa01"],
        }
        self.config.read_params = {"chunks": None, "use_chunks": False}

    @patch("eoio.readers.sentinel3_olci.auxiliary.lazy_rioxarray")
    @patch("eoio.readers.sentinel3_olci.auxiliary.read_obs_geometry")
    @patch("eoio.readers.sentinel3_olci.auxiliary.read_removed_pixels")
    def test_add_aux_calls_readers(self, mock_removed, mock_obs, mock_lazy_riox):
        # Patch all file I/O to return a dataset with band and spatial_ref
        mock_obs.return_value = self.ds
        mock_removed.return_value = self.ds

        class MockRio:
            def clip_box(self, x_min, y_min, x_max, y_max):
                return self

            def write_crs(self, crs):
                return self

            @property
            def rio(self):
                return self

        class MockObsVar:
            def __init__(self):
                self.dims = ("y", "x")
                self.data_vars = {
                    "OAA": (("y", "x"), np.zeros((1, 1))),
                    "OZA": (("y", "x"), np.zeros((1, 1))),
                    "SAA": (("y", "x"), np.zeros((1, 1))),
                    "SZA": (("y", "x"), np.zeros((1, 1))),
                }
                self.attrs = {}
                self.longitude = MagicMock()
                self.longitude.attrs = {"scale_factor": 1}
                self.latitude = MagicMock()
                self.latitude.attrs = {"scale_factor": 1}
                self.altitude = MagicMock()
                self.altitude.attrs = {"scale_factor": 1}

            def items(self):
                return self.data_vars.items()

            def __iter__(self):
                return iter([])

            def drop_vars(self, names):
                return self

            def rename(self, mapping):
                return self

            @property
            def rio(self):
                return MockRio()

        mock_lazy_riox.return_value.open_rasterio.return_value.squeeze.return_value = MockObsVar()
        _ds = auxiliary.add_aux(ds=self.ds, layout=self.layout, subset=None, config=self.config)
        mock_obs.assert_called_once()
        mock_removed.assert_called_once()

    @patch("eoio.readers.sentinel3_olci.auxiliary.lazy_rioxarray")
    def test_read_obs_geometry(self, mock_riox):
        mock_obs = MagicMock()
        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value.drop_vars.return_value = mock_obs
        self.layout.observation_geometry_path.return_value = "dummy_path"
        ds = auxiliary.read_obs_geometry(ds=self.ds, layout=self.layout, config=self.config, subset=None)
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_olci.auxiliary.lazy_rioxarray")
    def test_read_removed_pixels(self, mock_riox):
        mock_dims = MagicMock()
        mock_dims.longitude = MagicMock()
        mock_dims.latitude = MagicMock()
        mock_dims.longitude.attrs = {"scale_factor": 1}
        mock_dims.latitude.attrs = {"scale_factor": 1}
        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value.drop_vars.return_value = mock_dims
        self.layout.removed_pixels_path.return_value = "dummy_path"
        self.config.vars_sel["meas"] = ["Oa01"]
        ds = auxiliary.read_removed_pixels(ds=self.ds, layout=self.layout, config=self.config, subset=None)
        self.assertIsInstance(ds, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
