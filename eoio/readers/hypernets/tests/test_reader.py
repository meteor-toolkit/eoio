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


if __name__ == "__main__":
    unittest.main()
