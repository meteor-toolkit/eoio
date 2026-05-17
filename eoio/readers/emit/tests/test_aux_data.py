"""Tests for eoio.readers.emit.aux_data"""

import unittest
import xarray as xr
import numpy as np
from types import SimpleNamespace
from eoio.readers.emit import aux_data


class DummyConfig:
    def __init__(self, aux_vars):
        self.vars_sel = {"aux": aux_vars}


class TestAuxData(unittest.TestCase):
    def test_add_aux_no_config(self):
        ds = xr.Dataset()
        src = xr.Dataset()
        layout = SimpleNamespace(path="dummy_RAD")
        result = aux_data.add_aux(ds, src, layout, config=None)
        self.assertIs(result, ds)

    def test_add_misc_basic(self):
        ds = xr.Dataset()
        obs_ds = xr.Dataset({"Slope": (["y", "x"], np.ones((2, 2)))})
        misc_aux = ["Slope"]
        result = aux_data.add_misc(ds, obs_ds, misc_aux)
        self.assertIn("Slope", result)

    def test_add_elev_basic(self):
        ds = xr.Dataset()
        elev_ds = xr.Dataset({"elev": (["y", "x"], np.ones((2, 2)))})
        result = aux_data.add_elev(ds, elev_ds)
        self.assertIn("elev", result)


if __name__ == "__main__":
    unittest.main()
