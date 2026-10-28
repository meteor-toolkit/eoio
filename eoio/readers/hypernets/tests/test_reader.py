import unittest
from unittest import mock

import numpy as np
import xarray as xr
from eoio.readers.hypernets.reader import (
    HYPERNETSReader,
    HYPERNETSL1IrrReader,
    HYPERNETSL1RadReader,
    HYPERNETSL2RefReader,
)


class TestHYPERNETSReader(unittest.TestCase):
    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_all_options(self, mock_open_dataset):
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
        mock_open_dataset.return_value = ds
        reader = HYPERNETSReader(path=".", vars_sel=None, subset=None, read_params=None)
        options = reader.all_options
        print(options)
        assert "mask" in options["subset"]
        assert "wavelength" in options["subset"]
        assert "angle" in options["subset"]
        assert "datetime" in options["subset"]
        assert "time_of_day_utc" in options["subset"]
        assert "metadata_level" in options["read_params"]
        assert "include_uncertainties" in options["read_params"]
        assert "aux" in options["vars_sel"]
        assert "meas" in options["vars_sel"]
        assert "mask" in options["vars_sel"]

    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_open_dataset(self, mock_open_dataset):
        ds = xr.Dataset(
            {
                "quality_flag": ("series", np.array([0, 0, 1])),
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "viewing_zenith_angle": ("series", np.array([10, 20, 30])),
                "viewing_azimuth_angle": ("series", np.array([0, 90, 180])),
                "solar_zenith_angle": ("series", np.array([20, 30, 40])),
                "solar_azimuth_angle": ("series", np.array([180, 270, 360])),
                "acquisition_time": (
                    "series",
                    np.array(
                        ["2020-01-01T10:00", "2020-01-01T12:00", "2020-01-01T14:00"],
                        dtype="datetime64",
                    ),
                ),
            }
        )
        mock_open_dataset.return_value = ds
        reader = HYPERNETSReader(
            path=".",
            vars_sel=None,
            subset=None,
            read_params={"metadata_level": "original"},
        )
        result_ds = reader.open_dataset()
        assert isinstance(result_ds, xr.Dataset)
        assert "wavelength" in result_ds


class TestHYPERNETSL1IrrReader(unittest.TestCase):
    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_open_dataset(self, mock_open_dataset):
        ds = xr.Dataset(
            {
                "irradiance": (("wavelength", "series"), np.ones((3, 3))),
                "quality_flag": ("series", np.array([0, 0, 1])),
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "viewing_zenith_angle": ("series", np.array([10, 20, 30])),
                "viewing_azimuth_angle": ("series", np.array([0, 90, 180])),
                "solar_zenith_angle": ("series", np.array([20, 30, 40])),
                "solar_azimuth_angle": ("series", np.array([180, 270, 360])),
                "acquisition_time": (
                    "series",
                    np.array(
                        ["2020-01-01T10:00", "2020-01-01T12:00", "2020-01-01T14:00"],
                        dtype="datetime64",
                    ),
                ),
            }
        )
        mock_open_dataset.return_value = ds
        reader = HYPERNETSL1IrrReader(
            path=".",
            vars_sel={
                "aux": [
                    "viewing_zenith_angle",
                    "viewing_azimuth_angle",
                    "solar_zenith_angle",
                    "solar_azimuth_angle",
                ]
            },
            subset=None,
            read_params={"metadata_level": "original", "include_uncertainties": False},
        )
        result_ds = reader.open_dataset()
        assert isinstance(result_ds, xr.Dataset)
        assert "wavelength" in result_ds


class TestHYPERNETSL1RadReader(unittest.TestCase):
    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_open_dataset(self, mock_open_dataset):
        ds = xr.Dataset(
            {
                "radiance": (("wavelength", "series"), np.ones((3, 3))),
                "quality_flag": ("series", np.array([0, 0, 1])),
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "viewing_zenith_angle": ("series", np.array([10, 20, 30])),
                "viewing_azimuth_angle": ("series", np.array([0, 90, 180])),
                "solar_zenith_angle": ("series", np.array([20, 30, 40])),
                "solar_azimuth_angle": ("series", np.array([180, 270, 360])),
                "acquisition_time": (
                    "series",
                    np.array(
                        ["2020-01-01T10:00", "2020-01-01T12:00", "2020-01-01T14:00"],
                        dtype="datetime64",
                    ),
                ),
            }
        )
        mock_open_dataset.return_value = ds
        reader = HYPERNETSL1RadReader(
            path=".",
            vars_sel={
                "aux": [
                    "viewing_zenith_angle",
                    "viewing_azimuth_angle",
                    "solar_zenith_angle",
                    "solar_azimuth_angle",
                ]
            },
            subset=None,
            read_params={"metadata_level": "original", "include_uncertainties": False},
        )
        result_ds = reader.open_dataset()
        assert isinstance(result_ds, xr.Dataset)
        assert "wavelength" in result_ds


class TestHYPERNETSL2RefReader(unittest.TestCase):
    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_open_dataset(self, mock_open_dataset):
        ds = xr.Dataset(
            {
                "reflectance": (("wavelength", "series"), np.ones((3, 3))),
                "quality_flag": ("series", np.array([0, 0, 1])),
                "wavelength": ("wavelength", np.array([400, 500, 600])),
                "viewing_zenith_angle": ("series", np.array([10, 20, 30])),
                "viewing_azimuth_angle": ("series", np.array([0, 90, 180])),
                "solar_zenith_angle": ("series", np.array([20, 30, 40])),
                "solar_azimuth_angle": ("series", np.array([180, 270, 360])),
                "acquisition_time": (
                    "series",
                    np.array(
                        ["2020-01-01T10:00", "2020-01-01T12:00", "2020-01-01T14:00"],
                        dtype="datetime64",
                    ),
                ),
            }
        )
        mock_open_dataset.return_value = ds
        reader = HYPERNETSL2RefReader(
            path=".",
            vars_sel={
                "aux": [
                    "viewing_zenith_angle",
                    "viewing_azimuth_angle",
                    "solar_zenith_angle",
                    "solar_azimuth_angle",
                ]
            },
            subset=None,
            read_params={"metadata_level": "original", "include_uncertainties": False},
        )
        result_ds = reader.open_dataset()
        assert isinstance(result_ds, xr.Dataset)
        assert "wavelength" in result_ds


class TestHYPERNETSMeasDef(unittest.TestCase):
    """The subclasses' own meas_def is used, not every variable in the file.

    Regression test: NetCDFReader.__init__ used to assign meas_def from the file as an
    instance attribute, shadowing the class-level meas_def of every HYPERNETS reader.
    """

    def setUp(self):
        self.ds = xr.Dataset(
            {
                "radiance": (("wavelength", "series"), np.ones((3, 3))),
                "u_rel_random_radiance": (("wavelength", "series"), np.ones((3, 3))),
                "viewing_zenith_angle": ("series", np.array([10, 20, 30])),
                "viewing_azimuth_angle": ("series", np.array([0, 90, 180])),
                "solar_zenith_angle": ("series", np.array([20, 30, 40])),
                "solar_azimuth_angle": ("series", np.array([180, 270, 360])),
                "acceleration_x_mean": ("series", np.array([0.0, 0.0, 0.0])),
                "acquisition_time": ("series", np.array([1.0, 2.0, 3.0])),                
            },
            coords={"wavelength": np.array([400, 500, 600])},
        )

    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_meas_def_is_the_declared_one(self, mock_open_dataset):
        mock_open_dataset.return_value = self.ds
        reader = HYPERNETSL1RadReader(path=".", read_params={"metadata_level": "original"})
        assert reader.meas_def == {"all": ["radiance"]}
        assert reader.all_options["vars_sel"]["meas"] == ["all", "radiance"]

    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_aux_and_uncertainties_can_be_excluded(self, mock_open_dataset):
        mock_open_dataset.return_value = self.ds
        reader = HYPERNETSL1RadReader(
            path=".",
            vars_sel={"meas": "all", "aux": None, "mask": None},
            read_params={"metadata_level": "original", "include_uncertainties": False},
        )
        result_ds = reader.open_dataset()
        assert list(result_ds.data_vars) == ["radiance","acquisition_time"]

    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_an_aux_variable_is_not_a_measurement(self, mock_open_dataset):
        mock_open_dataset.return_value = self.ds
        with self.assertRaises(ValueError):
            HYPERNETSL1RadReader(path=".", vars_sel={"meas": "acceleration_x_mean"})

    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_base_reader_still_lists_the_file_variables(self, mock_open_dataset):
        """HYPERNETSReader itself declares no meas_def, so it falls back to the file."""
        mock_open_dataset.return_value = self.ds
        reader = HYPERNETSReader(path=".", read_params={"metadata_level": "original"})
        assert reader.meas_def == {"all": list(self.ds.variables)}


if __name__ == "__main__":
    unittest.main()
