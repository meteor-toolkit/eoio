"""eoio.readers.generic_netcdf.tests.test_subset - unit tests for eoio.readers.generic_netcdf.subset"""

import unittest
import xarray as xr
import numpy as np
from eoio.readers.generic_netcdf.subset import (
    build_subset,
    GENERIC_NETCDFSubset,
)


class TestGENERIC_NETCDFSubset(unittest.TestCase):
    def test_build_subset_basic(self):
        ds = xr.Dataset(
            {
                "var1": ("series", np.arange(5)),
                "var2": ("wavelength", np.arange(3)),
            }
        )
        subset = {"wavelength": {"min": 1, "max": 2}}
        result = build_subset(ds, subset)
        self.assertIsInstance(result, GENERIC_NETCDFSubset)
        self.assertTrue(np.array_equal(result.wavelength_indices, np.array([1, 2])))


if __name__ == "__main__":
    unittest.main()
