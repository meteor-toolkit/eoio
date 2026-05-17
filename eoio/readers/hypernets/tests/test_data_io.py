import xarray as xr
import numpy as np
from eoio.readers.hypernets.data_io import read_dataset
import unittest


class DummySubset:
    def __init__(self, series_indices, wavelength_indices):
        self.series_indices = series_indices
        self.wavelength_indices = wavelength_indices


class TestReadDataset(unittest.TestCase):
    def test_read_bands_into_dataset_basic(self):
        ds = xr.Dataset(
            {
                "var1": (["series", "wavelength"], [[1, 2], [3, 4]]),
                "var2": (["series", "wavelength"], [[5, 6], [7, 8]]),
            }
        )
        include_vars = ["var1"]
        subset = DummySubset(series_indices=[0], wavelength_indices=[1])
        result = read_dataset(ds=ds, include_vars=include_vars, subset=subset)
        assert "var1" in result
        assert "var2" not in result
        assert result["var1"].shape == (1, 1)

    def test_read_dataset(self):
        ds = xr.Dataset(
            {
                "var1": (["series", "wavelength"], np.arange(6).reshape(2, 3)),
                "var2": (["series", "wavelength"], np.arange(6, 12).reshape(2, 3)),
            }
        )
        include_vars = ["var1"]
        subset = DummySubset(series_indices=[0], wavelength_indices=[1, 2])
        result = read_dataset(ds=ds, include_vars=include_vars, subset=subset)
        assert "var1" in result
        assert "var2" not in result
        assert result["var1"].shape == (1, 2)
        assert np.all(result["var1"].values == np.array([[1, 2]]))
