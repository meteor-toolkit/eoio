import unittest

import numpy as np
import xarray as xr
from eoio.readers.hypernets.subset import build_subset, HYPERNETSSubset


class TestSubset(unittest.TestCase):
    def test_build_subset_mask_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "quality_flag": ("series", np.array([0, 1, 0])),
            }
        )
        subset = {
            "mask": True,
            "wavelength": {"min": 380, "max": 1700},
            "angle": None,
            "datetime": None,
            "time_of_day_utc": None,
        }
        result = build_subset(ds, subset)
        assert isinstance(result, HYPERNETSSubset)
        assert np.all(result.series_indices == np.array([0, 2]))

    def test_build_subset_wavelength_only(self):
        ds = xr.Dataset({"wavelength": ("wavelength", np.array([400, 500, 600]))})
        subset = {
            "mask": None,
            "wavelength": {"min": 450, "max": 650},
            "angle": None,
            "datetime": None,
            "time_of_day_utc": None,
        }
        result = build_subset(ds, subset)
        assert np.all(result.wavelength_indices == np.array([1, 2]))

    def test_build_subset_angle_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "viewing_zenith_angle": ("series", np.array([10, 20, 30])),
            }
        )
        subset = {
            "mask": None,
            "wavelength": {"min": 380, "max": 1700},
            "angle": {"vza": {"min": 15, "max": 25}},
            "datetime": None,
            "time_of_day_utc": None,
        }
        result = build_subset(ds, subset)
        assert np.all(result.series_indices == np.array([1]))

    def test_build_subset_datetime_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "acquisition_time": (
                    "series",
                    np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64"),
                ),
            }
        )
        subset = {
            "mask": None,
            "wavelength": {"min": 380, "max": 1700},
            "angle": None,
            "datetime": {"min": "2020-01-02", "max": "2020-01-03"},
            "time_of_day_utc": None,
        }
        result = build_subset(ds, subset)
        assert np.all(result.series_indices == np.array([1, 2]))

    def test_build_subset_time_of_day_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "acquisition_time": (
                    "series",
                    np.array(
                        ["2020-01-01T10:00", "2020-01-01T12:00", "2020-01-01T14:00"],
                        dtype="datetime64",
                    ),
                ),
            }
        )
        subset = {
            "mask": None,
            "wavelength": {"min": 380, "max": 1700},
            "angle": None,
            "datetime": None,
            "time_of_day_utc": {"min": "11:00", "max": "13:00"},
        }
        result = build_subset(ds, subset)
        assert np.all(result.series_indices == np.array([1]))

    def test_build_subset_combined(self):
        ds = xr.Dataset(
            {
                "quality_flag": ("series", np.array([0, 0, 1])),
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "viewing_zenith_angle": ("series", np.array([10, 20, 30])),
                "acquisition_time": (
                    "series",
                    np.array(
                        ["2020-01-01T10:00", "2020-01-01T12:00", "2020-01-01T14:00"],
                        dtype="datetime64",
                    ),
                ),
            }
        )
        subset = {
            "mask": True,
            "wavelength": {"min": 450, "max": 650},
            "angle": {"vza": {"min": 15, "max": 25}},
            "datetime": {"min": "2020-01-01T00:00", "max": "2020-01-03T00:00"},
            "time_of_day_utc": {"min": "11:00", "max": "13:00"},
        }
        result = build_subset(ds, subset)
        assert np.all(result.series_indices == np.array([1]))
        assert np.all(result.wavelength_indices == np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
