import unittest
import numpy as np
import xarray as xr
from eoio.readers.radcalnet.data_io import read_dataset


class DummySubset:
    def __init__(self, series_indices, wavelength_indices):
        self.series_indices = series_indices
        self.wavelength_indices = wavelength_indices


class testReadDataset(unittest.TestCase):
    def test_read_dataset(self):
        ds = xr.Dataset(
            {
                "var1": (["time", "wavelength"], [[1, 2], [3, 4]]),
                "var2": (["time", "wavelength"], [[5, 6], [7, 8]]),
            }
        )

        include_vars = ["var1"]
        subset = DummySubset(series_indices=[0], wavelength_indices=[1])
        result = read_dataset(ds=ds, include_vars=include_vars, subset=subset)
        # RadCalNet does not have an include_vars option because only one var exists
        assert "var1" in result
        assert "var2" not in result
        assert result["var1"].shape == (1, 1)

    def test_read_bands_into_dataset_basic(self):
        ds = xr.Dataset(
            {
                "var1": (["time", "wavelength"], np.arange(6).reshape(2, 3)),
                "var2": (["time", "wavelength"], np.arange(6, 12).reshape(2, 3)),
            }
        )

        include_vars = ["var1"]
        subset = DummySubset(series_indices=[0], wavelength_indices=[1, 2])
        result = read_dataset(ds=ds, include_vars=include_vars, subset=subset)
        assert "var1" in result
        assert "var2" not in result
        assert result["var1"].shape == (1, 2)
        assert np.all(result["var1"].values == np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
