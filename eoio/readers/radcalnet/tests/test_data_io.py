import os
import tempfile
import unittest
import numpy as np
import xarray as xr
from eoio.readers.radcalnet.data_io import VARIABLE_ATTRS, _uncertainty_attrs, read_dataset, read_file


class DummySubset:
    def __init__(self, series_indices, wavelength_indices):
        self.series_indices = series_indices
        self.wavelength_indices = wavelength_indices


# Minimal but complete RadCalNet ASCII fixture: one site line, a reflectance
# wavelength row plus its uncertainty row (same leading key "559" twice --
# read_file() suffixes the repeat with "_unc"), and the 13 time-column rows
# read_file() requires (Year, DOY(U), DOY(L), UTC, Local -- hardcoded to 13
# columns, see `range(13)`).
_YEAR_COLUMNS = "\t".join(["2021"] * 13)
_DOY_COLUMNS = "\t".join(str(d) for d in range(289, 302))
_UTC_COLUMNS = "\t".join(
    [
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
)
_RADCALNET_FIXTURE = "\n".join(
    [
        "Site:\tGONA01",
        "Lat:\t-23.59999",
        "Lon:\t15.119215",
        "Alt:\t510.0",
        f"Year:\t{_YEAR_COLUMNS}",
        f"DOY(U):\t{_DOY_COLUMNS}",
        f"DOY(L):\t{_DOY_COLUMNS}",
        f"UTC:\t{_UTC_COLUMNS}",
        f"Local:\t{_UTC_COLUMNS}",
        "559\t" + "\t".join(["0.1"] * 13),
        "559\t" + "\t".join(["0.01"] * 13),
    ]
)


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

    def test_read_dataset_with_no_subset_criteria_returns_all_rows(self):
        """Regression test: series_indices/wavelength_indices are None when
        no subset criteria applies to that dimension (e.g. build_subset()
        leaves series_indices None when time_of_day_utc/time_of_day_local/
        angle/datetime are all None -- the default since RadCalNetReader
        stopped hardcoding a universal time_of_day_utc window). Passing None
        straight through to isel() as an indexer used to raise ("invalid
        indexer array, does not have integer dtype") instead of being
        treated as "don't subset this dimension"."""
        ds = xr.Dataset(
            {
                "var1": (["time", "wavelength"], np.arange(6).reshape(2, 3)),
            }
        )

        include_vars = ["var1"]
        subset = DummySubset(series_indices=None, wavelength_indices=None)
        result = read_dataset(ds=ds, include_vars=include_vars, subset=subset)
        assert result["var1"].shape == (2, 3)
        assert np.all(result["var1"].values == ds["var1"].values)

    def test_read_dataset_with_only_wavelength_subset_leaves_time_unsubset(self):
        ds = xr.Dataset(
            {
                "var1": (["time", "wavelength"], np.arange(6).reshape(2, 3)),
            }
        )

        include_vars = ["var1"]
        subset = DummySubset(series_indices=None, wavelength_indices=[1, 2])
        result = read_dataset(ds=ds, include_vars=include_vars, subset=subset)
        assert result["var1"].shape == (2, 2)
        assert np.all(result["var1"].values == np.array([[1, 2], [4, 5]]))


class testReadFile(unittest.TestCase):
    """Regression tests for read_file()'s Site/Lat/Lon/Alt attrs.

    These header lines only ever carry one value each, but every DATA[k]
    parses to a list (see read_file()'s try/except around float(_x) for
    _x in x[1:]) -- previously left as-is, so e.g. attrs["Longitude"] was
    [15.119215] instead of 15.119215. That silently corrupted any string
    built from it, e.g. get_basic_metadata()'s WKT footprint became
    "POINT ([15.119215] [-23.59999])" -- invalid WKT that only surfaced as
    a downstream parse failure in eoio's own footprint_utils, far from the
    actual bug.
    """

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(_RADCALNET_FIXTURE)

    def tearDown(self):
        os.remove(self.path)

    def test_site_lat_lon_alt_are_scalars_not_lists(self):
        ds = read_file(self.path, aux_vars=[])

        self.assertIsInstance(ds.attrs["Site"], str)
        self.assertIsInstance(ds.attrs["Lattitude"], float)
        self.assertIsInstance(ds.attrs["Longitude"], float)
        self.assertIsInstance(ds.attrs["Altitude"], float)

    def test_site_lat_lon_alt_values(self):
        ds = read_file(self.path, aux_vars=[])

        self.assertEqual(ds.attrs["Site"], "GONA01")
        self.assertEqual(ds.attrs["Lattitude"], -23.59999)
        self.assertEqual(ds.attrs["Longitude"], 15.119215)
        self.assertEqual(ds.attrs["Altitude"], 510.0)


class TestReflectanceUncertaintyIsAnObsarrayComponent(unittest.TestCase):
    """reflectance_uncertainty must be registered as a proper obsarray uncertainty
    component of reflectance (ds.unc[...]), not just a plain, same-shaped sibling
    data_var -- eoalign's uncertainty-detection utilities rely on this registration
    (ds.unc.unc_vars) to recognise and skip it rather than processing it as if it were
    ordinary data (see eoalign.utils.uncertainty)."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(_RADCALNET_FIXTURE)

    def tearDown(self):
        os.remove(self.path)

    def test_reflectance_uncertainty_is_a_registered_unc_var(self):
        ds = read_file(self.path, aux_vars=[])
        self.assertIn("reflectance_uncertainty", ds.unc.unc_vars)
        self.assertIn("reflectance", ds.unc.obs_vars)
        self.assertEqual(ds["reflectance"].attrs["unc_comps"], "reflectance_uncertainty")

    def test_reflectance_uncertainty_values_are_unchanged(self):
        # _RADCALNET_FIXTURE's single "559" wavelength row has reflectance 0.1,
        # uncertainty 0.01, at every one of its 13 time samples.
        ds = read_file(self.path, aux_vars=[])
        self.assertTrue(np.allclose(ds["reflectance_uncertainty"].values, 0.01))

    def test_reflectance_uncertainty_has_err_corr_metadata(self):
        ds = read_file(self.path, aux_vars=[])
        attrs = ds["reflectance_uncertainty"].attrs
        # wavelength block: "systematic" (fully correlated) -- matches s2radval's own
        # documented assumption when combining these across a sensor SRF.
        self.assertEqual(attrs["err_corr_1_dim"], ["wavelength"])
        self.assertEqual(attrs["err_corr_1_form"], "systematic")
        # time block: "random" (independent atmospheric retrieval per time sample).
        self.assertEqual(attrs["err_corr_2_dim"], ["time"])
        self.assertEqual(attrs["err_corr_2_form"], "random")

    def test_aux_uncertainty_is_a_registered_unc_var(self):
        # _RADCALNET_FIXTURE has no aux (P/T/...) rows -- build one with a "P" (air
        # pressure) row plus its repeated-key uncertainty row, matching how reflectance's
        # own value/uncertainty pair is written.
        fixture = _RADCALNET_FIXTURE + "\nP\t" + "\t".join(["1013.0"] * 13) + "\nP\t" + "\t".join(["1.0"] * 13)
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(fixture)
        self.addCleanup(os.remove, path)

        ds = read_file(path, aux_vars=["air_pressure"])
        self.assertIn("air_pressure_uncertainty", ds.unc.unc_vars)
        self.assertEqual(ds["air_pressure"].attrs["unc_comps"], "air_pressure_uncertainty")
        self.assertEqual(ds["air_pressure_uncertainty"].attrs["err_corr_1_dim"], ["time"])
        self.assertEqual(ds["air_pressure_uncertainty"].attrs["err_corr_1_form"], "random")


class TestVariableAttrsHaveStandardName(unittest.TestCase):
    """Regression test: several VARIABLE_ATTRS entries (water_vapour, ozone,
    aerosol_optical_depth, aerosol_type, earth_sun_distance, local_time,
    reflectance_uncertainty) used to have no standard_name at all, which
    BaseMetadataExtractor.get_variable_basic_metadata warns about on every read since it
    treats standard_name as a required basic-metadata key -- see
    eoio.readers.radcalnet.tests.test_reader.testVariableNamingAndAttrs for the
    end-to-end, warning-free assertion."""

    def test_every_variable_attrs_entry_has_a_standard_name(self):
        # "reflectance" is the one deliberate exception: its standard_name is
        # TOA-/BOA-specific and set dynamically by the reader (see VARIABLE_ATTRS'
        # own comment), not present in this static dict.
        for name, attrs in VARIABLE_ATTRS.items():
            if name == "reflectance":
                continue
            self.assertIn("standard_name", attrs, f"{name!r} has no standard_name")
            self.assertTrue(attrs["standard_name"], f"{name!r} has an empty standard_name")

    def test_uncertainty_attrs_has_a_standard_name(self):
        for base_name in ("air_pressure", "water_vapour", "ozone"):
            attrs = _uncertainty_attrs(base_name)
            self.assertEqual(attrs["standard_name"], f"{base_name}_uncertainty")


if __name__ == "__main__":
    unittest.main()
