import os.path
from os import pardir
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
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
        "air_pressure": [
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
        "air_temperature": [
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
        "water_vapour": [
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
        "ozone": [
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
        "aerosol_optical_depth": [
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
        "angstrom_exponent": [
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
        "aerosol_type": ["R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R"],
        "air_pressure_uncertainty": [
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
        "air_temperature_uncertainty": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "water_vapour_uncertainty": [
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
        "ozone_uncertainty": [6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2],
        "aerosol_optical_depth_uncertainty": [
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
        "angstrom_exponent_uncertainty": [
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
        "earth_sun_distance": [
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

    def test_default_subset_has_no_universal_time_of_day_utc_filter(self):
        """Regression test: default_subset["time_of_day_utc"] used to be
        hardcoded to 08:00-14:00 UTC for every site, which only worked by
        coincidence for sites whose local daylight happens to fall in that
        range (e.g. GONA). For a site whose real daylight window is
        elsewhere in UTC (e.g. RVUS, ~15:00-21:00 UTC), a caller that reads
        without an explicit time_of_day_utc override -- as
        eomatch.datatree.BuildMUDT.read_products does, passing only
        roi/roi_crs -- would have every row filtered out and
        eoio.readers.radcalnet.subset.build_subset would raise "The subset
        criteria resulted in an empty dataset.", even though the file's own
        recorded rows are already correctly restricted to that site's actual
        daylight window (see scrappi's own per-site fix,
        RadcalnetCallHandler._daylight_window_hours)."""
        self.assertIsNone(RadCalNetReader.default_subset["time_of_day_utc"])

        # Confirm this holds through the real merge path too (subset=None
        # means "use the reader's own defaults", exactly as eomatch's
        # BuildMUDT effectively does for any key it doesn't explicitly set).
        reader = DummyReader(path=".", vars_sel=None, subset=None, read_params=None)
        self.assertIsNone(reader.config.subset["time_of_day_utc"])

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

    def test_metadata_level_true_is_treated_as_all(self):
        """metadata_level=True (bool) must attach metadata the same as metadata_level='all'.

        Passing the Python bool True (as eomatch's own config does) previously fell through
        both the "original" and ("all", "basic") checks in open_dataset(), so clear_metadata()
        ran but attach_metadata() never did -- silently returning a dataset with every attrs
        dict wiped to {}.
        """
        ds = xr.Dataset(
            {
                "reflectance": (["wavelength", "time"], np.zeros((2, 1))),
                "reflectance_uncertainty": (["wavelength", "time"], np.zeros((2, 1))),
            },
            coords={"wavelength": [400.0, 410.0], "time": [np.datetime64("2021-02-08")]},
        )
        with tempfile.NamedTemporaryFile(suffix=".output") as tmp:
            with (
                patch("eoio.readers.radcalnet.reader.read_file", return_value=ds),
                patch("eoio.readers.radcalnet.reader.build_subset", return_value=None),
                patch("eoio.readers.radcalnet.reader.read_dataset", side_effect=lambda ds, include_vars, subset: ds),
                patch("eoio.readers.radcalnet.reader.RadCalNetMetadataExtractor") as mock_extractor_cls,
            ):
                mock_extractor = mock_extractor_cls.return_value
                mock_extractor.clear_metadata.side_effect = lambda ds: ds
                mock_extractor.attach_metadata.side_effect = lambda ds, level: ds

                reader = RadCalNetReader(tmp.name, read_params={"metadata_level": True})
                reader.open_dataset()

                mock_extractor.attach_metadata.assert_called_once()
                self.assertEqual(mock_extractor.attach_metadata.call_args.kwargs["level"], "all")

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


def _write_synthetic_ascii_file(suffix: str) -> str:
    """Build a minimal but structurally-valid RadCalNet ascii file (self-contained, unlike
    the other tests in this module which need T:\\ drive access) so the variable naming and
    attrs behaviour of read_file()/open_dataset() can be regression-tested without real data.

    ``esd`` (earth_sun_distance) is only written for ``.output`` (TOA) files: real ``.input``
    (BOA) files don't carry it either -- RadCalNetInputReader.aux_def never selects it, so an
    unselected "esd" row would otherwise fall through read_file()'s generic per-wavelength
    fallback loop and fail float("esd")."""
    n = 13

    def row(v):
        return "\t".join([str(v)] * n)

    utc_times = [
        (datetime.datetime(2021, 1, 1, 8, 0) + datetime.timedelta(minutes=30 * i)).strftime("%H%M") for i in range(n)
    ]
    local_times = [
        (datetime.datetime(2021, 1, 1, 9, 0) + datetime.timedelta(minutes=30 * i)).strftime("%H%M") for i in range(n)
    ]

    lines = [
        "Site:\tGONA01",
        "Lat:\t-23.59999",
        "Lon:\t15.119215",
        "Alt:\t510.0",
        "Year\t" + row("2021"),
        "DOY(U)\t" + row("286"),
        "UTC\t" + "\t".join(utc_times),
        "DOY(L)\t" + row("286"),
        "Local\t" + "\t".join(local_times),
        "400\t" + row("0.30"),
        "400\t" + row("0.01"),
        "P\t" + row("963.15"),
        "P\t" + row("2.5"),
        "T\t" + row("297.1"),
        "T\t" + row("0.5"),
        "WV\t" + row("0.62"),
        "WV\t" + row("0.13"),
        "O3\t" + row("312.0"),
        "O3\t" + row("6.2"),
        "AOD\t" + row("0.032"),
        "AOD\t" + row("0.001"),
        "Ang\t" + row("0.952"),
        "Ang\t" + row("0.017"),
        "Zen\t" + row("40.1"),
        "Azi\t" + row("120.5"),
        "Type\t" + row("R"),
    ]
    if suffix == ".output":
        lines.append("esd\t" + row("1.0161"))
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    return path


def _write_ascii_with_reflectance(suffix: str, refl_pairs) -> str:
    """Like :py:func:`_write_synthetic_ascii_file` but with caller-supplied reflectance
    rows, for testing fill-value masking.

    :param refl_pairs: list of ``(wavelength, reflectance_values, uncertainty_values)``,
        each ``*_values`` a length-13 list of strings written as that wavelength's
        reflectance row and (repeated key) its uncertainty row.
    """
    n = 13

    def row(v):
        return "\t".join([str(v)] * n)

    utc_times = [
        (datetime.datetime(2021, 1, 1, 8, 0) + datetime.timedelta(minutes=30 * i)).strftime("%H%M") for i in range(n)
    ]
    local_times = [
        (datetime.datetime(2021, 1, 1, 9, 0) + datetime.timedelta(minutes=30 * i)).strftime("%H%M") for i in range(n)
    ]

    lines = [
        "Site:\tGONA01",
        "Lat:\t-23.59999",
        "Lon:\t15.119215",
        "Alt:\t510.0",
        "Year\t" + row("2021"),
        "DOY(U)\t" + row("286"),
        "UTC\t" + "\t".join(utc_times),
        "DOY(L)\t" + row("286"),
        "Local\t" + "\t".join(local_times),
    ]
    for wavelength, refl_values, unc_values in refl_pairs:
        lines.append(f"{wavelength}\t" + "\t".join(str(v) for v in refl_values))
        lines.append(f"{wavelength}\t" + "\t".join(str(v) for v in unc_values))
    for aux in ("P", "T", "WV", "O3", "AOD", "Ang"):
        lines.append(f"{aux}\t" + row("1.0"))
        lines.append(f"{aux}\t" + row("0.1"))
    lines += ["Zen\t" + row("40.1"), "Azi\t" + row("120.5"), "Type\t" + row("R")]
    if suffix == ".output":
        lines.append("esd\t" + row("1.0161"))
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    return path


class testReflectanceFillMasking(unittest.TestCase):
    """RadCalNet flags an unavailable reflectance sample with a large out-of-range
    sentinel (documented ~9999, but ERUS files use ~1000) rather than omitting the row.
    open_dataset() must convert those -- and the matching uncertainty -- to NaN, like
    every other eoio reader's missing-data handling."""

    def _open(self, refl_pairs):
        path = _write_ascii_with_reflectance(".output", refl_pairs)
        self.addCleanup(os.unlink, path)
        # time_of_day_utc=None keeps all 13 columns (no per-site daylight window)
        return RadCalNetReader(path, subset={"time_of_day_utc": None}).open_dataset()

    def test_reflectance_at_or_above_1000_is_masked(self):
        ds = self._open(
            [("500", ["0.30"] * 11 + ["1000.0", "9999.0"], ["0.01"] * 13)],
        )
        refl = ds["reflectance"].sel(wavelength=500).values
        self.assertTrue(np.isnan(refl[11]))
        self.assertTrue(np.isnan(refl[12]))
        self.assertFalse(np.isnan(refl[0]))
        self.assertAlmostEqual(float(refl[0]), 0.30, places=6)

    def test_matching_uncertainty_is_masked_where_reflectance_is_fill(self):
        ds = self._open(
            [("500", ["0.30"] * 12 + ["1000.0"], ["0.01"] * 13)],
        )
        self.assertTrue(np.isnan(ds["reflectance_uncertainty"].sel(wavelength=500).values[12]))
        self.assertFalse(np.isnan(ds["reflectance_uncertainty"].sel(wavelength=500).values[0]))

    def test_reflectance_is_masked_where_only_its_uncertainty_is_fill(self):
        ds = self._open(
            [("500", ["0.30"] * 13, ["0.01"] * 12 + ["1000.0"])],
        )
        self.assertTrue(np.isnan(ds["reflectance"].sel(wavelength=500).values[12]))
        self.assertTrue(np.isnan(ds["reflectance_uncertainty"].sel(wavelength=500).values[12]))

    def test_all_valid_reflectance_is_left_untouched(self):
        ds = self._open([("500", ["0.42"] * 13, ["0.01"] * 13)])
        self.assertFalse(np.isnan(ds["reflectance"].sel(wavelength=500).values).any())


class testVariableNamingAndAttrs(unittest.TestCase):
    """Regression tests for RadCalNet's variable naming and CF-style attrs: the raw ascii
    file's own abbreviated column headers (P, T, WV, ...) are translated to descriptive
    names matching every other eoio reader, and every variable gets long_name/standard_name/
    units (previously attached nowhere in the reading pipeline)."""

    def setUp(self):
        self.toa_path = _write_synthetic_ascii_file(".output")
        self.boa_path = _write_synthetic_ascii_file(".input")

    def tearDown(self):
        os.unlink(self.toa_path)
        os.unlink(self.boa_path)

    def test_old_abbreviated_names_are_gone(self):
        ds = RadCalNetReader(self.toa_path).open_dataset()
        for old_name in ("P", "T", "esd", "Type", "P_unc", "T_unc"):
            self.assertNotIn(old_name, ds.variables)

    def test_new_descriptive_names_present(self):
        ds = RadCalNetReader(self.toa_path).open_dataset()
        for new_name in (
            "air_pressure",
            "air_temperature",
            "earth_sun_distance",
            "aerosol_type",
            "air_pressure_uncertainty",
        ):
            self.assertIn(new_name, ds.variables)

    def test_aux_variables_have_units_and_standard_name(self):
        ds = RadCalNetReader(self.toa_path).open_dataset()
        self.assertEqual(ds["air_pressure"].attrs.get("units"), "hPa")
        self.assertEqual(ds["air_pressure"].attrs.get("standard_name"), "air_pressure")
        self.assertEqual(ds["air_temperature"].attrs.get("units"), "K")
        self.assertEqual(ds["earth_sun_distance"].attrs.get("units"), "AU")

    def test_meteo_variables_have_standard_name(self):
        """Regression test: water_vapour/ozone/aerosol_optical_depth (and the
        no-CF-equivalent aerosol_type/earth_sun_distance/local_time) used to have no
        standard_name in VARIABLE_ATTRS at all, so BaseMetadataExtractor warned
        "missing expected metadata key: standard_name" for every one of them on every
        read -- see test_no_missing_standard_name_warnings below for the warning-free
        assertion across the whole dataset."""
        ds = RadCalNetReader(self.toa_path).open_dataset()
        self.assertEqual(ds["water_vapour"].attrs.get("standard_name"), "atmosphere_mass_content_of_water_vapor")
        self.assertEqual(
            ds["ozone"].attrs.get("standard_name"), "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
        )
        self.assertEqual(
            ds["aerosol_optical_depth"].attrs.get("standard_name"), "atmosphere_optical_thickness_due_to_aerosol"
        )
        self.assertEqual(ds["aerosol_type"].attrs.get("standard_name"), "aerosol_type")
        self.assertEqual(ds["earth_sun_distance"].attrs.get("standard_name"), "earth_sun_distance")

    def test_no_missing_standard_name_warnings(self):
        """No variable in a fully-read TOA or BOA dataset should trigger
        BaseMetadataExtractor.get_variable_basic_metadata's "missing expected metadata
        key" warning for standard_name -- see test_meteo_variables_have_standard_name
        and _uncertainty_attrs for where each variable's standard_name now comes from."""
        for path, reader_cls in ((self.toa_path, RadCalNetReader), (self.boa_path, RadCalNetInputReader)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reader_cls(path).open_dataset()

            standard_name_warnings = [
                str(w.message) for w in caught if "missing expected metadata key: standard_name" in str(w.message)
            ]
            self.assertEqual(standard_name_warnings, [])

    def test_uncertainty_variable_has_attrs(self):
        ds = RadCalNetReader(self.toa_path).open_dataset()
        self.assertEqual(ds["air_pressure_uncertainty"].attrs.get("units"), "hPa")
        self.assertIn("uncertainty", ds["air_pressure_uncertainty"].attrs.get("long_name", ""))
        self.assertEqual(ds["air_pressure_uncertainty"].attrs.get("standard_name"), "air_pressure_uncertainty")

    def test_wavelength_coordinate_has_units(self):
        ds = RadCalNetReader(self.toa_path).open_dataset()
        self.assertEqual(ds["wavelength"].attrs.get("units"), "nm")

    def test_reflectance_standard_name_is_toa_for_output_reader(self):
        ds = RadCalNetReader(self.toa_path).open_dataset()
        self.assertEqual(ds["reflectance"].attrs.get("standard_name"), "toa_reflectance")

    def test_reflectance_standard_name_is_boa_for_input_reader(self):
        ds = RadCalNetInputReader(self.boa_path).open_dataset()
        self.assertEqual(ds["reflectance"].attrs.get("standard_name"), "boa_reflectance")

    def test_collection_attr_survives_as_top_level_attr(self):
        """Regression test: reader.py sets ds.attrs["collection"] before the metadata
        extractor is constructed. clear_metadata() then wiped it, and since "collection"
        wasn't a key in get_basic_metadata()'s output, it never came back at the top level --
        only as a nested, easy-to-miss ds.attrs["product_metadata"]["collection"]."""
        ds = RadCalNetReader(self.toa_path).open_dataset()
        self.assertEqual(ds.attrs.get("collection"), "Top of Atmosphere")

        ds = RadCalNetInputReader(self.boa_path).open_dataset()
        self.assertEqual(ds.attrs.get("collection"), "Bottom of Atmosphere")


if __name__ == "__main__":
    unittest.main()
