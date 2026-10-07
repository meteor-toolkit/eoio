"""eoio.readers.generic_netcdf.tests.test_reader - unit tests for eoio.readers.generic_netcdf.reader"""

import unittest
from pathlib import Path
import numpy as np
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

    def test_open_dataset_with_a_datetime_variable(self):
        """A file carrying a `datetime` variable reads without raising.

        Regression test: build_subset used to index subset["roi"] unguarded,
        so any netCDF with a `datetime` variable raised KeyError('roi') --
        NetCDFReader declares no default_subset, so that key is never present.
        """
        path = "test_datetime.nc"
        xr.Dataset(
            {"var1": ("datetime", [1, 2, 3])},
            coords={"datetime": np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")},
        ).to_netcdf(path)
        try:
            ds = NetCDFReader(path).open_dataset()
            self.assertIsInstance(ds, xr.Dataset)
            self.assertIn("var1", ds)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reader_accepts_no_subsetting_parameters(self):
        """The generic reader advertises no subset vocabulary, and says so clearly.

        An arbitrary netCDF cannot be assumed to name its coordinates the way
        eoio does, so subsetting belongs to readers that know their product
        (see eoio.readers.ecmwf.subset).
        """
        self.assertEqual(NetCDFReader(self.test_path).all_options["subset"], {})
        with self.assertRaises(ValueError) as ctx:
            NetCDFReader(self.test_path, subset={"roi": (0, 0, 1, 1)})
        self.assertIn("Unknown subsetting parameter(s): ['roi']", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
