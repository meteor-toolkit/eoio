import os
import tempfile
import unittest
import numpy as np
import xarray as xr
from eoio.readers.radcalnet.data_io import read_dataset, read_file


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


if __name__ == "__main__":
    unittest.main()
