"""eoio.readers.generic_netcdf.tests.test_reader - unit tests for eoio.readers.generic_netcdf.reader"""

import unittest
from pathlib import Path
import xarray as xr
from eoio.readers.generic_netcdf.reader import NetCDFReader


class TestNetCDFReader(unittest.TestCase):
    def setUp(self):
        # Create a simple NetCDF file for testing
        self.test_path = "test.nc"
        self.ds = xr.Dataset(
            {
                "var1": ("series", [1, 2, 3]),
                "var2": ("wavelength", [4, 5, 6]),
            }
        )
        self.ds.to_netcdf(self.test_path)

    def tearDown(self):
        Path(self.test_path).unlink(missing_ok=True)

    def test_init_and_open_dataset(self):
        reader = NetCDFReader(self.test_path)
        ds = reader.open_dataset()
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("var1", ds)
        self.assertIn("var2", ds)


if __name__ == "__main__":
    unittest.main()
