"""eoio.readers.generic_netcdf.tests.test_data_io - unit tests for eoio.readers.generic_netcdf.data_io"""

import unittest
import xarray as xr
import numpy as np
from eoio.readers.generic_netcdf.data_io import read_dataset
from eoio.readers.generic_netcdf.subset import GENERIC_NETCDFSubset


class TestReadDataset(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset(
            {
                "var1": ("series", np.arange(5)),
                "var2": ("wavelength", np.arange(3)),
                "wavelength": ("wavelength", np.arange(3)),
            }
        )

    def test_read_dataset_include_vars(self):
        result = read_dataset(ds=self.ds, include_vars=["var1"])
        self.assertIn("var1", result)
        self.assertNotIn("var2", result)

    def test_read_dataset_with_subset(self):
        subset = GENERIC_NETCDFSubset(wavelength_indices=np.array([1]))
        result = read_dataset(ds=self.ds, include_vars=["var2"], subset=subset)
        self.assertIn("var2", result)
        self.assertEqual(result["var2"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
