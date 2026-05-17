"""
eoio.readers.sentinel2.aux_vars.tests.test_aux_data - tests for aux_data helper functions
"""

from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import xarray as xr
import eoio.readers.sentinel2.aux_vars.aux_data as aux_data


class TestAuxData(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = MagicMock(name="S2Layout")
        self.mtd = MagicMock(name="S2MSIMetadataExtractor")
        self.ds = xr.Dataset()

    def test_get_available_aux_proc_version_ge_5(self):
        self.layout.proc_version = (5, 0, 0)

        with (
            patch.object(aux_data, "AUX_ECMWF_VARS_NEW", ["ecmwf_new_a", "ecmwf_new_b"]),
            patch.object(aux_data, "AUX_CAMS_VARS", ["cams_a"]),
            patch.object(aux_data, "ANGLE_VARS", ["ang_a", "ang_b"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_OLD", ["ecmwf_old_should_not_appear"]),
        ):
            out = aux_data.get_available_aux(self.layout)
            self.assertEqual(out, ["ecmwf_new_a", "ecmwf_new_b", "cams_a", "ang_a", "ang_b"])

    def test_get_available_aux_proc_version_lt_5(self):
        self.layout.proc_version = (4, 0, 0)

        with (
            patch.object(aux_data, "AUX_ECMWF_VARS_NEW", ["ecmwf_new_should_not_appear"]),
            patch.object(aux_data, "AUX_CAMS_VARS", ["cams_should_not_appear"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_OLD", ["ecmwf_old_a", "ecmwf_old_b"]),
            patch.object(aux_data, "ANGLE_VARS", ["ang_a"]),
        ):
            out = aux_data.get_available_aux(self.layout)
            self.assertEqual(out, ["ecmwf_old_a", "ecmwf_old_b", "ang_a"])

    @patch.object(aux_data, "add_meteo")
    @patch.object(aux_data, "add_angles")
    def test_add_aux_calls_add_angles_only_when_only_angle_vars_requested(self, mock_add_angles, mock_add_meteo):
        with (
            patch.object(aux_data, "ANGLE_VARS", ["sun_zenith", "sun_azimuth", "va_zenith"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_NEW", ["tcwv"]),
            patch.object(aux_data, "AUX_CAMS_VARS", ["tco3"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_OLD", ["msl"]),
        ):
            config = SimpleNamespace(
                vars_sel={"aux": ["va_zenith", "sun_azimuth"]},
                read_params={"ave_va_det": True},
            )

            ds_out = xr.Dataset(attrs={"marker": "from_add_angles"})
            mock_add_angles.return_value = ds_out

            out = aux_data.add_aux(ds=self.ds, layout=self.layout, config=config, mtd=self.mtd)

            # angle_names are built in ANGLE_VARS order, filtered by requested vars
            expected_angle_names = ["sun_azimuth", "va_zenith"]

            mock_add_angles.assert_called_once()
            _, kwargs = mock_add_angles.call_args
            self.assertIs(kwargs["ds"], self.ds)
            self.assertEqual(kwargs["angle_names"], expected_angle_names)
            self.assertIs(kwargs["mtd"], self.mtd)
            self.assertEqual(kwargs["ave_det"], True)

            mock_add_meteo.assert_not_called()
            self.assertIs(out, ds_out)

    @patch.object(aux_data, "add_meteo")
    @patch.object(aux_data, "add_angles")
    def test_add_aux_calls_add_meteo_only_when_only_meteo_vars_requested(self, mock_add_angles, mock_add_meteo):
        with (
            patch.object(aux_data, "ANGLE_VARS", ["sun_zenith"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_NEW", ["tcwv", "u10"]),
            patch.object(aux_data, "AUX_CAMS_VARS", ["tco3"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_OLD", ["msl"]),
        ):
            config = SimpleNamespace(
                vars_sel={"aux": ["msl", "tcwv", "not_aux"]},
                read_params={"ave_va_det": False},
            )

            ds_out = xr.Dataset(attrs={"marker": "from_add_meteo"})
            mock_add_meteo.return_value = ds_out

            out = aux_data.add_aux(ds=self.ds, layout=self.layout, config=config, mtd=self.mtd)

            mock_add_angles.assert_not_called()

            # meteo_names preserve the concatenation order:
            # AUX_ECMWF_VARS_NEW + AUX_CAMS_VARS + AUX_ECMWF_VARS_OLD filtered by requested vars
            expected_meteo_names = ["tcwv", "msl"]

            mock_add_meteo.assert_called_once()
            _, kwargs = mock_add_meteo.call_args
            self.assertIs(kwargs["ds"], self.ds)
            self.assertEqual(kwargs["var_names"], expected_meteo_names)
            self.assertIs(kwargs["layout"], self.layout)
            self.assertIs(kwargs["config"], config)

            self.assertIs(out, ds_out)

    @patch.object(aux_data, "add_meteo")
    @patch.object(aux_data, "add_angles")
    def test_add_aux_calls_angles_then_meteo_when_mixed_requested(self, mock_add_angles, mock_add_meteo):
        with (
            patch.object(aux_data, "ANGLE_VARS", ["sun_zenith", "sun_azimuth"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_NEW", ["tcwv", "u10"]),
            patch.object(aux_data, "AUX_CAMS_VARS", ["tco3"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_OLD", ["msl"]),
        ):
            config = SimpleNamespace(
                vars_sel={"aux": ["sun_zenith", "msl", "u10"]},
                read_params={"ave_va_det": True},
            )

            ds_after_angles = xr.Dataset(attrs={"stage": "after_angles"})
            ds_after_meteo = xr.Dataset(attrs={"stage": "after_meteo"})

            mock_add_angles.return_value = ds_after_angles
            mock_add_meteo.return_value = ds_after_meteo

            out = aux_data.add_aux(ds=self.ds, layout=self.layout, config=config, mtd=self.mtd)

            mock_add_angles.assert_called_once()
            _, ang_kwargs = mock_add_angles.call_args
            self.assertEqual(ang_kwargs["angle_names"], ["sun_zenith"])

            mock_add_meteo.assert_called_once()
            _, met_kwargs = mock_add_meteo.call_args

            # ensure meteo sees the dataset returned by add_angles
            self.assertIs(met_kwargs["ds"], ds_after_angles)
            self.assertEqual(met_kwargs["var_names"], ["u10", "msl"])
            self.assertIs(met_kwargs["layout"], self.layout)
            self.assertIs(met_kwargs["config"], config)

            self.assertIs(out, ds_after_meteo)

    @patch.object(aux_data, "add_meteo")
    @patch.object(aux_data, "add_angles")
    def test_add_aux_ignores_unknown_aux_names(self, mock_add_angles, mock_add_meteo):
        with (
            patch.object(aux_data, "ANGLE_VARS", ["sun_zenith"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_NEW", ["tcwv"]),
            patch.object(aux_data, "AUX_CAMS_VARS", ["tco3"]),
            patch.object(aux_data, "AUX_ECMWF_VARS_OLD", ["msl"]),
        ):
            config = SimpleNamespace(
                vars_sel={"aux": ["not_real_1", "not_real_2"]},
                read_params={"ave_va_det": True},
            )

            out = aux_data.add_aux(ds=self.ds, layout=self.layout, config=config, mtd=self.mtd)

            mock_add_angles.assert_not_called()
            mock_add_meteo.assert_not_called()
            self.assertIs(out, self.ds)

    @patch.object(aux_data, "add_meteo")
    @patch.object(aux_data, "add_angles")
    def test_add_aux_handles_empty_aux_list(self, mock_add_angles, mock_add_meteo):
        config = SimpleNamespace(
            vars_sel={"aux": []},
            read_params={"ave_va_det": True},
        )

        out = aux_data.add_aux(ds=self.ds, layout=self.layout, config=config, mtd=self.mtd)

        mock_add_angles.assert_not_called()
        mock_add_meteo.assert_not_called()
        self.assertIs(out, self.ds)


if __name__ == "__main__":
    unittest.main()
