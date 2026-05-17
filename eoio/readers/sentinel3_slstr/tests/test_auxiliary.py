"""Unit tests for eoio.readers.sentinel3_slstr.auxiliary.

Mirrors sentinel3_olci auxiliary tests but adapted for SLSTR helpers.
"""

import unittest
import xarray as xr
from unittest.mock import MagicMock, patch

from eoio.readers.sentinel3_slstr import auxiliary


class TestS3SLSTRAuxiliary(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.layout = MagicMock()
        self.config = MagicMock()
        self.config.vars_sel = {"aux": None, "meas": ["S1_radiance_an"]}
        self.config.read_params = {"chunks": None, "use_chunks": False}

    def test_add_aux_returns_unmodified_when_no_aux_requested(self):
        self.config.vars_sel = {"aux": None, "meas": ["S1_radiance_an"]}
        out = auxiliary.add_aux(ds=self.ds, layout=self.layout, grids={}, subset=None, config=self.config)
        self.assertIs(out, self.ds)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_met")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.xr.open_dataset")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.lazy_rioxarray")
    def test_add_aux_calls_subreaders_when_aux_present(self, mock_riox, mock_xr_open, mock_read_met):
        # Request a couple of auxiliary groups; patch internals to avoid I/O
        self.config.vars_sel = {"aux": ["cartesian", "met"], "meas": ["S1_radiance_an"]}
        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value = MagicMock()
        # patch xr.open_dataset used by read_time/read_viscal and patch read_met
        mock_xr_open.return_value = xr.Dataset()
        mock_read_met.return_value = xr.Dataset()

        # Call add_aux and verify it returns a dataset (no exceptions)
        ds = auxiliary.add_aux(
            ds=self.ds,
            layout=self.layout,
            grids={"tx": 16000},
            subset=None,
            clip_boxes={"tx": (0, 0, 1, 1)},
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_cartesian_geometry")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_indices")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_met")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_time")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_viscal")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_observation_geometry")
    @patch("eoio.readers.sentinel3_slstr.auxiliary.read_orphan")
    def test_add_aux_delegates_to_each_reader(
        self,
        mock_orphan,
        mock_obs_geometry,
        mock_viscal,
        mock_time,
        mock_met,
        mock_indices,
        mock_cartesian,
    ):
        # Request many aux groups and verify delegation
        self.config.vars_sel = {
            "aux": [
                "cartesian",
                "indices",
                "met",
                "time",
                "viscal",
                "observation_geometry",
                "orphan",
            ],
            "meas": ["S1_radiance_an"],
        }

        grids = {"an": 500}
        clip_boxes = {"an": (0, 0, 1, 1)}

        # Ensure each mocked reader returns a dataset so add_aux returns an xr.Dataset
        mock_cartesian.return_value = xr.Dataset()
        mock_indices.return_value = xr.Dataset()
        mock_met.return_value = xr.Dataset()
        mock_time.return_value = xr.Dataset()
        mock_viscal.return_value = xr.Dataset()
        mock_obs_geometry.return_value = xr.Dataset()
        mock_orphan.return_value = xr.Dataset()

        ds = auxiliary.add_aux(
            ds=self.ds,
            layout=self.layout,
            grids=grids,
            subset=None,
            clip_boxes=clip_boxes,
            config=self.config,
        )

        self.assertIsInstance(ds, xr.Dataset)
        mock_cartesian.assert_called_once()
        mock_indices.assert_called_once()
        mock_met.assert_called_once()
        mock_time.assert_called_once()
        mock_viscal.assert_called_once()
        mock_obs_geometry.assert_called_once()
        mock_orphan.assert_called_once()

        # Provide minimal grids and clip_boxes
        grids = {"an": 500}
        clip_boxes = {"an": (0, 0, 1, 1)}

        ds = auxiliary.add_aux(
            ds=self.ds,
            layout=self.layout,
            grids=grids,
            subset=None,
            clip_boxes=clip_boxes,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.lazy_rioxarray")
    def test_read_cartesian_geometry(self, mock_riox):
        # ensure aux selection is a container so function can check for 'orphan'
        self.config.vars_sel = {"aux": []}
        # Return a mock that supports the chained calls used in production
        mock_obj = MagicMock()
        mock_obj.squeeze.return_value.drop_vars.return_value = mock_obj
        mock_obj.rename.return_value = mock_obj
        mock_riox.return_value.open_rasterio.return_value = mock_obj
        self.layout.cartesian_path.return_value = "dummy"
        grids = {"tx": 16000}
        clip_boxes = {"tx": (0, 0, 1, 1)}
        ds = auxiliary.read_cartesian_geometry(
            ds=self.ds,
            layout=self.layout,
            grids=grids,
            subset=None,
            clip_boxes=clip_boxes,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.lazy_rioxarray")
    def test_read_indices(self, mock_riox):
        import numpy as np

        # ensure aux selection is a container so function can check for 'orphan'
        self.config.vars_sel = {"aux": []}
        # open_rasterio returns a sequence whose first element is the dataset
        mock_riox.return_value.open_rasterio.return_value = [xr.Dataset({"idx": (("y", "x"), np.zeros((1, 1)))})]
        self.layout.indices_path.return_value = "dummy"
        grids = {"an": 500}
        ds = auxiliary.read_indices(
            ds=self.ds,
            layout=self.layout,
            grids=grids,
            subset=None,
            clip_boxes=None,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.xr.open_dataset")
    def test_read_met(self, mock_xr_open):
        mock_xr_open.return_value = xr.Dataset({"met": ((), 1)})
        self.layout.met_path.return_value = "dummy"
        ds = auxiliary.read_met(
            ds=self.ds,
            layout=self.layout,
            subset=None,
            clip_boxes=None,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.xr.open_dataset")
    def test_read_time(self, mock_xr_open):
        # time datasets provide a 'rows' variable that get renamed
        mock_xr_open.return_value = xr.Dataset({"rows": (("rows",), [1])})
        self.layout.time_path.return_value = "dummy"
        grids = {"an": 500}
        ds = auxiliary.read_time(
            ds=self.ds,
            layout=self.layout,
            grids=grids,
            subset=None,
            clip_boxes=None,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.xr.open_dataset")
    def test_read_viscal(self, mock_xr_open):
        mock_xr_open.return_value = xr.Dataset({"v": ((), 1)})
        self.layout.viscal_path.return_value = "dummy"
        ds = auxiliary.read_viscal(
            ds=self.ds,
            layout=self.layout,
            subset=None,
            clip_boxes=None,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.lazy_rioxarray")
    def test_read_obs_geometry(self, mock_riox):

        # Make the chained calls return an xarray.Dataset-like object
        mock_obj = MagicMock()
        mock_obj.squeeze.return_value.drop_vars.return_value = mock_obj
        mock_obj.rename.return_value = mock_obj
        mock_obj.attrs = {}
        mock_riox.return_value.open_rasterio.return_value = mock_obj
        self.layout.geometry_tn_path.return_value = "dummy"
        clip_boxes = {"tx": (0, 0, 1, 1)}
        # supply minimal grids mapping required by API
        ds = auxiliary.read_observation_geometry(
            ds=self.ds,
            layout=self.layout,
            grids={"an": 500},
            subset=None,
            clip_boxes=clip_boxes,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.auxiliary.lazy_rioxarray")
    def test_read_orphan(self, mock_riox):

        # layout.meas_paths should return mapping of band->path
        bnd = "S1_radiance_an"
        self.layout.meas_paths.return_value = {bnd: "p1"}
        # compute expected orphan var name using same logic as production
        _orphan_var = bnd[:-3] + "_orphan_" + bnd[-2:]
        # open_rasterio returns a sequence where [1] contains an object that
        # yields the orphan variable when indexed and supports chained calls
        second = MagicMock()
        mock_orphan_da = MagicMock()
        mock_orphan_da.squeeze.return_value.drop_vars.return_value = mock_orphan_da
        mock_orphan_da.rename.return_value = mock_orphan_da
        second.__getitem__.return_value = mock_orphan_da
        mock_riox.return_value.open_rasterio.return_value = [MagicMock(), second]
        ds = auxiliary.read_orphan(
            ds=self.ds,
            layout=self.layout,
            subset=None,
            clip_boxes=None,
            config=self.config,
        )
        self.assertIsInstance(ds, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
