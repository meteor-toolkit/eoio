import os.path
from os import pardir
import sys
import unittest
import numpy as np
import xarray as xr
import configparser
import datetime
from eoio.readers.radcalnet.reader import (
    RadCalNetReader,
    RadCalNetInputReader,
    RADCALNETFileTypeError,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    pardir,
    pardir,
    # pardir,
    # "eoio",
    "etc",
    "test_file_paths.config",
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# Assume config has a section [hypernets] and key 'l1b_file'
test_data_path = config.get("RadCalNet", "output_file", fallback=None)
inpt_data_path = config.get("RadCalNet", "input_file", fallback=None)
ncdf_data_path = config.get("RadCalNet", "netcdf_file", fallback=None)
if sys.platform == "linux":
    inpt_data_path = inpt_data_path.replace("T:\\ECO\\EOServer\\", r"/mnt/t/").replace("\\", "/")
    test_data_path = test_data_path.replace("T:\\ECO\\EOServer\\", r"/mnt/t/").replace("\\", "/")
    ncdf_data_path = ncdf_data_path.replace("T:\\ECO\\EOServer\\", r"/mnt/t/").replace("\\", "/")


class DummyReader(RadCalNetReader):
    meas_def = {"all": ["var1"]}
    aux_def = {"all": ["aux1"]}
    mask_def = {"all": ["mask1"]}
    uncertainty_vars = ["unc1"]

    def open_dataset(self):
        return xr.Dataset(
            {
                "var1": (["series"], [1]),
                "aux1": (["series"], [2]),
                "mask1": (["series"], [3]),
                "unc1": (["series"], [4]),
            }
        )

    def extract_metadata(self, level=None):
        return {}, {}, {}


class testSiteDaylightWindowUTC(unittest.TestCase):
    def test_gona_matches_old_hardcoded_default(self):
        # GONA (15.119215 degE) is the site the old fixed 08:00-14:00 UTC
        # default was tuned for -- the dynamic window should reproduce it.
        # This also regression-tests a rounding bug found during development:
        # independently rounding hour/minute parts gave "07:60" for this exact
        # longitude (int(7.9920...)=7, round(0.9920...*60)=60, no carry).
        window = RadCalNetReader._site_daylight_window_utc(15.119215)
        assert window == {"min": "08:00", "max": "14:00"}

    def test_rvus_gets_a_daytime_not_nighttime_window(self):
        # RVUS (-115.69 degW) is many hours behind GONA in UTC -- the old
        # hardcoded 08:00-14:00 UTC window fell in the middle of RVUS's
        # local night, producing an empty dataset after subsetting.
        window = RadCalNetReader._site_daylight_window_utc(-115.69)
        assert window == {"min": "16:43", "max": "22:43"}


class testTimeOfDayUTCExplicitlySet(unittest.TestCase):
    def test_not_set_when_subset_omitted(self):
        reader = DummyReader(path=".", vars_sel=None, subset=None, read_params=None)
        assert reader._time_of_day_utc_explicitly_set is False

    def test_not_set_when_subset_given_without_the_key(self):
        reader = DummyReader(
            path=".", vars_sel=None, subset={"wavelength": {"min": 400, "max": 2500}}, read_params=None
        )
        assert reader._time_of_day_utc_explicitly_set is False

    def test_set_when_key_present_even_if_value_is_none(self):
        # An explicit {"time_of_day_utc": None} means "no time-of-day
        # subsetting at all" and must survive -- it must not be mistaken
        # for "caller didn't specify, fill in the per-site default".
        reader = DummyReader(path=".", vars_sel=None, subset={"time_of_day_utc": None}, read_params=None)
        assert reader._time_of_day_utc_explicitly_set is True

    def test_set_when_key_present_with_a_value(self):
        reader = DummyReader(
            path=".", vars_sel=None, subset={"time_of_day_utc": {"min": "09:00", "max": "12:00"}}, read_params=None
        )
        assert reader._time_of_day_utc_explicitly_set is True


class testRadCalNetReader(unittest.TestCase):
    utc = [
        "08:00",
        "08:30",
        "09:00",
        "09:30",
        "10:00",
        "10:30",
        "11:00",
        "11:30",
        "12:00",
        "12:30",
        "13:00",
        "13:30",
        "14:00",
    ]
    local = [
        "09:00",
        "09:30",
        "10:00",
        "10:30",
        "11:00",
        "11:30",
        "12:00",
        "12:30",
        "13:00",
        "13:30",
        "14:00",
        "14:30",
        "15:00",
    ]

    expected_attrs = {
        "Site": ["GONA01"],
        "Lattitude": [-23.59999],
        "Longitude": [15.119215],
        "Altitude": [510.0],
    }
    expected = {
        "P": [
            963.15,
            963.32,
            963.50,
            963.49,
            963.21,
            962.92,
            962.64,
            962.36,
            962.09,
            961.93,
            961.78,
            961.63,
            961.48,
        ],
        "T": [
            297.1,
            300.5,
            303.1,
            305.3,
            306.6,
            307.1,
            307.4,
            307.5,
            307.7,
            307.6,
            307.9,
            308.0,
            308.0,
        ],
        "WV": [
            0.62,
            0.63,
            0.64,
            0.64,
            0.63,
            0.64,
            0.66,
            0.67,
            0.66,
            0.64,
            0.64,
            0.63,
            0.63,
        ],
        "O3": [
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
            312.0,
        ],
        "AOD": [
            0.032,
            0.031,
            0.03,
            0.03,
            0.028,
            0.027,
            0.03,
            0.031,
            0.031,
            0.033,
            0.033,
            0.034,
            0.033,
        ],
        "Ang": [
            0.952,
            0.963,
            0.974,
            0.985,
            0.996,
            1.007,
            1.107,
            1.076,
            1.043,
            1.011,
            0.978,
            0.946,
            0.913,
        ],
        "Type": ["R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R"],
        "P_unc": [
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
        ],
        "T_unc": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "WV_unc": [
            0.12,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
            0.13,
        ],
        "O3_unc": [6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2],
        "AOD_unc": [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
        ],
        "Ang_unc": [
            0.017,
            0.017,
            0.017,
            0.017,
            0.017,
            0.018,
            0.019,
            0.019,
            0.018,
            0.018,
            0.017,
            0.017,
            0.016,
        ],
    }

    expected_TOA = {
        "esd": [
            1.01612,
            1.01612,
            1.01612,
            1.01612,
            1.01612,
            1.01612,
            1.01611,
            1.01611,
            1.01611,
            1.01611,
            1.01611,
            1.01611,
            1.01610,
        ],
    }

    def test_all_options(self):
        reader = DummyReader(path=".", vars_sel=None, subset=None, read_params=None)
        opts = reader.all_options
        assert isinstance(opts, dict)

    def test_open_dataset(self):
        reader = DummyReader(path=".", vars_sel=None, subset=None, read_params=None)
        ds = reader.open_dataset()
        assert "var1" in ds
        assert "aux1" in ds
        assert "mask1" in ds
        assert "unc1" in ds

    # def test_read_netcdf(self):
    # if not ncdf_data_path.exists():
    #         self.skipTest(f"Test .nc data not found at {ncdf_data_path}")
    # # Unfortunately the netcdf reader requires a bit more work.
    #     nc = RadCalNetInputReader(ncdf_data_path)
    #     ds = nc.open_dataset()

    def test_read_BOA_product(self):
        # skip test if data is not available
        if not os.path.exists(inpt_data_path):
            self.skipTest(f"Test .input data not found at {inpt_data_path}")

        inpt = RadCalNetInputReader(inpt_data_path)
        ds = inpt.open_dataset()

        for row in self.expected:
            assert ds[row].values.tolist() == self.expected[row]

        # test that dimensions are correct
        assert ds.wavelength.values.shape[0] == 211
        assert ds.time.values.shape[0] == 13

    def test_read_TOA_product(self):
        # skip test if data not avaliable
        if not os.path.exists(test_data_path):
            self.skipTest(f"Test .output data not found at {test_data_path}")

        # use default subset
        output = RadCalNetReader(test_data_path)
        ds = output.open_dataset()

        expected = self.expected.copy()
        expected.update(self.expected_TOA)

        for row in expected:
            assert ds[row].values.tolist() == expected[row]

        # test that dimensions are correct
        assert ds.wavelength.values.shape[0] == 211
        assert ds.time.values.shape[0] == 13

    def test_wrong_file_type(self):
        if not os.path.exists(test_data_path):
            self.skipTest(f"Test .output data not found at {test_data_path}")
        output = RadCalNetReader(test_data_path)
        output.path = test_data_path.replace(".output", ".csv")  # naughty changing of class type in non defined way
        self.assertRaises(RADCALNETFileTypeError, output.open_dataset)
        # test open_dataset raises error if file is not correct type -- RADCALNETFileTypeError

    def test_read_into_dataset(self):
        # skip test if no data
        if not os.path.exists(test_data_path):
            self.skipTest(f"Test .output data not found at {test_data_path}")
        # use default subset
        output = RadCalNetReader(test_data_path, read_params={"include_uncertainties": False})
        # force rcn reader to only take reflectance
        test_ds = output.open_dataset()

        assert test_ds.reflectance.shape == (211, 13)
        assert not hasattr(test_ds, "reflectance_uncertainty")

    def test_read_with_subsetting_params(self):
        # skip if data = no
        if not os.path.exists(test_data_path):
            self.skipTest(f"Test .output data not found at {test_data_path}")

        SUBSET = {
            "wavelength": {"min": 400, "max": 2500},
            "datetime": {
                "min": datetime.datetime(2017, 7, 19, 8, 0),
                "max": datetime.datetime(2017, 7, 19, 12, 30),
            },  # should get the whole day
            "angle": None,
            "time_of_day_local": None,
            "time_of_day_utc": None,
        }

        output = RadCalNetReader(test_data_path, subset=SUBSET)
        ds = output.open_dataset()

        assert ds.reflectance.shape == (211, 10)
        vals = np.array(
            [
                ds.reflectance.values[:, i][~np.isnan(ds.reflectance.values[:, i])]
                for i in range(ds.reflectance.values.shape[1])
            ]
        )
        assert vals.shape == (10, 181)

    def test_explicit_time_of_day_utc_overrides_site_default(self):
        # skip if no data
        if not os.path.exists(test_data_path):
            self.skipTest(f"Test .output data not found at {test_data_path}")

        # GONA's dynamic per-site window (08:00-14:00 UTC) covers all 13
        # samples in self.utc; a narrower explicit override must be honoured
        # instead of being replaced by the per-site default.
        output = RadCalNetReader(test_data_path, subset={"time_of_day_utc": {"min": "08:00", "max": "10:00"}})
        ds = output.open_dataset()

        assert ds.time.values.shape[0] == 5


if __name__ == "__main__":
    unittest.main()
