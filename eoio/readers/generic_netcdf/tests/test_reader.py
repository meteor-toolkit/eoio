"""eoio.readers.generic_netcdf.tests.test_reader - unit tests for eoio.readers.generic_netcdf.reader"""

import datetime as dt
import tempfile
import unittest
from pathlib import Path
import numpy as np
import xarray as xr
from shapely.geometry import box
from eoio.interface import write
from eoio.readers.generic_netcdf.reader import NetCDFReader
from eoio.serialisation import MARKER_ATTR


class TestNetCDFReader(unittest.TestCase):
    def setUp(self):
        # Create a simple NetCDF file for testing
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.test_path = str(self.tmp_path / "test.nc")
        self.ds = xr.Dataset(
            {
                "var1": ("series", [1, 2, 3]),
                "var2": ("wavelength", [4, 5, 6]),
            }
        )
        self.ds.to_netcdf(self.test_path)

    def tearDown(self):
        self.tmp.cleanup()

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
        path = str(self.tmp_path / "test_datetime.nc")
        xr.Dataset(
            {"var1": ("datetime", [1, 2, 3])},
            coords={"datetime": np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")},
        ).to_netcdf(path)
        ds = NetCDFReader(path).open_dataset()
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("var1", ds)

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


class TestNetCDFReaderAttrDecoding(unittest.TestCase):
    """Attributes encoded for serialisation are decoded on every read."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.sensing_time = dt.datetime(2023, 7, 2, 10, 30)
        self.ds = xr.Dataset(
            {
                "B02": (
                    ("y", "x"),
                    np.zeros((3, 3), dtype="float32"),
                    {
                        "long_name": "B02",
                        "standard_name": "toa_reflectance",
                        "units": "1",
                        "product_metadata": {"band_id": "B02", "detector": {"id": "D01"}},
                    },
                )
            },
            coords={"x": np.arange(3.0), "y": np.arange(3.0)},
            attrs={
                "product_name": "S2A_TEST",
                "product_version": "03.01",
                "product_datetime": self.sensing_time,
                "product_bounds": box(0, 0, 1, 1),
                "footprint": {"geometry": box(0, 0, 1, 1), "crs": "EPSG:32630", "bounds": (0, 0, 1, 1)},
                "product_metadata": {"sensing_time": self.sensing_time, "tile": "30UXC"},
            },
        )
        self.path = write(self.ds, self.tmp_path / "out.nc", writer="netcdf")

    def tearDown(self):
        self.tmp.cleanup()

    def test_attrs_written_by_eoio_are_restored(self):
        ds = NetCDFReader(self.path, read_params={"metadata_level": "original"}).open_dataset()
        self.assertEqual(ds.attrs["product_datetime"], self.sensing_time)
        self.assertEqual(ds.attrs["product_metadata"], {"sensing_time": self.sensing_time, "tile": "30UXC"})
        self.assertEqual(ds.attrs["product_bounds"], box(0, 0, 1, 1))
        self.assertEqual(ds["B02"].attrs["product_metadata"], {"band_id": "B02", "detector": {"id": "D01"}})

    def test_encoding_marker_is_removed(self):
        ds = NetCDFReader(self.path, read_params={"metadata_level": "original"}).open_dataset()
        self.assertNotIn(MARKER_ATTR, ds.attrs)
        self.assertNotIn(MARKER_ATTR, ds["B02"].attrs)

    def test_numeric_looking_strings_are_left_alone(self):
        ds = NetCDFReader(self.path, read_params={"metadata_level": "original"}).open_dataset()
        self.assertEqual(ds.attrs["product_version"], "03.01")

    def test_decoded_attrs_reach_the_attached_metadata(self):
        # At metadata_level "all" the file's own attributes are nested under
        # product_metadata, so the original product_metadata appears one level down.
        ds = NetCDFReader(self.path).open_dataset()
        self.assertEqual(ds.attrs["product_datetime"], self.sensing_time)
        self.assertEqual(ds.attrs["product_metadata"]["product_metadata"]["sensing_time"], self.sensing_time)
        self.assertEqual(ds["B02"].attrs["product_metadata"]["product_metadata"]["detector"], {"id": "D01"})

    def test_footprint_crs_survives_the_round_trip(self):
        ds = NetCDFReader(self.path).open_dataset()
        self.assertEqual(ds.attrs["footprint"]["crs"], "EPSG:32630")
        self.assertEqual(ds.attrs["footprint"]["geometry"], box(0, 0, 1, 1))

    def test_source_dataset_is_not_modified(self):
        reader = NetCDFReader(self.path, read_params={"metadata_level": "original"})
        reader.open_dataset()
        self.assertIn(MARKER_ATTR, reader.ds_src.attrs)

    def test_third_party_json_and_datetime_strings_are_decoded(self):
        """A file eoio did not write carries no marker, so decoding is best-effort."""
        path = str(self.tmp_path / "third_party.nc")
        xr.Dataset(
            {"var1": ("series", [1, 2, 3])},
            attrs={
                "time_coverage_start": "2023-07-02T10:30:00Z",
                "bounds": "[0, 0, 1, 1]",
                "date_issued": "2023-07-02",
                "title": "A third-party product",
            },
        ).to_netcdf(path)
        ds = NetCDFReader(path, read_params={"metadata_level": "original"}).open_dataset()
        self.assertEqual(ds.attrs["time_coverage_start"], dt.datetime(2023, 7, 2, 10, 30, tzinfo=dt.timezone.utc))
        self.assertEqual(ds.attrs["bounds"], [0, 0, 1, 1])
        self.assertEqual(ds.attrs["date_issued"], "2023-07-02")
        self.assertEqual(ds.attrs["title"], "A third-party product")


if __name__ == "__main__":
    unittest.main()
