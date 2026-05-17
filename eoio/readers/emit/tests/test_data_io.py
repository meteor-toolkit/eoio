"""Tests for eoio.readers.emit.data_io"""

import unittest
import xarray as xr
import numpy as np
from eoio.readers.emit import data_io
from eoio.readers.emit.subset import EMITSubset


class TestDataIO(unittest.TestCase):
    def test_attach_coords_from_groups_noop(self):
        arr = xr.DataArray(np.ones((2, 2)), dims=["y", "x"])
        result = data_io.attach_coords_from_groups(arr, "nonexistent_file.nc")
        self.assertIsInstance(result, xr.DataArray)

    def test_read_dataset_basic(self):
        ds = xr.Dataset()
        src = xr.Dataset({"radiance": (["y", "x"], np.ones((2, 2)))})
        meas_vars = ["radiance"]
        subset = EMITSubset()
        result = data_io.read_dataset(ds, src, meas_vars, subset)
        self.assertIn("radiance", result)


if __name__ == "__main__":
    unittest.main()
