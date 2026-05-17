import unittest
import numpy as np
import xarray as xr
from datetime import datetime
from eoio.readers.radcalnet.subset import build_subset

utc_times = np.array(
    [
        datetime(2022, 7, 10, 8, 0),
        datetime(2022, 7, 10, 8, 30),
        datetime(2022, 7, 10, 9, 0),
        datetime(2022, 7, 10, 9, 30),
        datetime(2022, 7, 10, 10, 0),
        datetime(2022, 7, 10, 10, 30),
        datetime(2022, 7, 10, 11, 0),
        datetime(2022, 7, 10, 11, 30),
        datetime(2022, 7, 10, 12, 0),
        datetime(2022, 7, 10, 12, 30),
    ]
)


local_times = np.array(
    [
        datetime(2022, 7, 10, 9, 0),
        datetime(2022, 7, 10, 9, 30),
        datetime(2022, 7, 10, 10, 0),
        datetime(2022, 7, 10, 10, 30),
        datetime(2022, 7, 10, 11, 0),
        datetime(2022, 7, 10, 11, 30),
        datetime(2022, 7, 10, 12, 0),
        datetime(2022, 7, 10, 12, 30),
        datetime(2022, 7, 10, 13, 0),
        datetime(2022, 7, 10, 13, 30),
    ]
)


class testBuildRadCalNetSubset(unittest.TestCase):
    def test_build_subset_wavelength_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "time": ("time", utc_times),
            }
        )
        subset = {
            "mask": None,
            "wavelength": {"min": 450, "max": 650},
            "datetime": None,
            "angle": None,
            "time_of_day_local": None,
            "time_of_day_utc": None,
        }
        result = build_subset(ds=ds, subset=subset)
        assert np.all(result.wavelength_indices == np.array([1, 2]))

    def test_build_subset_time_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "time": ("time", utc_times),
            }
        )
        subset = {
            "mask": None,
            "wavelength": {"min": 400, "max": 2500},
            "datetime": None,
            "angle": None,
            "time_of_day_local": None,
            "time_of_day_utc": {"min": "10:00", "max": "12:00"},
        }
        result = build_subset(ds=ds, subset=subset)
        assert np.all(result.series_indices == np.array([4, 5, 6, 7, 8]))

    def test_build_subset_local_time_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "time": ("time", utc_times),
                "local_time": ("time", local_times),
            }
        )
        subset = {
            "wavelength": {"min": 400, "max": 2500},
            "mask": None,
            "datetime": None,
            "angle": None,
            "time_of_day_local": {"min": "11:00", "max": "13:00"},
            "time_of_day_utc": None,
        }
        result = build_subset(ds=ds, subset=subset)
        assert np.all(result.series_indices == np.array([4, 5, 6, 7, 8]))

    def test_build_subset_datetime_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "time": ("time", utc_times),
                "local_time": ("time", local_times),
            }
        )
        subset = {
            "wavelength": {"min": 400, "max": 2500},
            "mask": None,
            "datetime": {
                "min": utc_times[0],
                "max": utc_times[-1],
            },  # should get the whole day
            "angle": None,
            "time_of_day_local": None,
            "time_of_day_utc": None,
        }
        result = build_subset(ds=ds, subset=subset)
        assert np.all(result.series_indices == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        subset = {
            "wavelength": {"min": 400, "max": 2500},
            "mask": None,
            "datetime": {
                "min": "2022-07-10",
                "max": "2022-07-11",
            },  # should get the whole day
            "angle": None,
            "time_of_day_local": None,
            "time_of_day_utc": None,
        }
        result = build_subset(ds=ds, subset=subset)
        assert np.all(result.series_indices == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_build_subset_angle_only(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "time": ("time", utc_times),
                "solar_zenith_angle": (
                    "time",
                    np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                ),
            }
        )
        subset = {
            "wavelength": {"min": 400, "max": 2500},
            "mask": None,
            "datetime": None,
            "angle": {"sza": {"min": 15, "max": 45}},
            "time_of_day_local": None,
            "time_of_day_utc": None,
        }
        result = build_subset(ds=ds, subset=subset)
        assert np.all(result.series_indices == np.array([1, 2, 3]))

    def test_build_subset_combined(self):
        ds = xr.Dataset(
            {
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "time": ("time", utc_times),
                "local_time": ("time", local_times),
                "solar_zenith_angle": (
                    "time",
                    np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                ),
                "quality_flag": ("time", np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])),
            }
        )
        subset = {
            "wavelength": {"min": 450, "max": 650},
            "mask": True,
            "datetime": {"min": utc_times[2], "max": utc_times[7]},
            "angle": {"sza": {"min": 15, "max": 45}},
            "time_of_day_local": {"min": "10:00", "max": "13:00"},
            "time_of_day_utc": {"min": "09:00", "max": "12:00"},
        }
        result = build_subset(ds=ds, subset=subset)
        assert np.all(result.series_indices == np.array([2, 3]))


if __name__ == "__main__":
    unittest.main()
