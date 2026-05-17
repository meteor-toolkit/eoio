"""
eoio.readers.sentinel2.aux_vars.tests.test_aux_data - tests for aux_data helper functions
"""

import glob
import os
import socket
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import xarray as xr
import numpy as np
from eoio.readers.sentinel2.layout import S2Layout
from eoio.readers.base import ReaderConfig
from eoio.readers.sentinel2.aux_vars.meteo import add_meteo

# Path to a real Sentinel-2 SAFE product for integration testing.
# If you run the tests, ensure this path points to a valid SAFE product on your system.
# Otherwise, the integration test will be skipped.
# SAFE_PATH = "/Users/seh2/Library/CloudStorage/OneDrive-NationalPhysicalLaboratory/Data/archive/sentinel2/S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"
if "lyon" or "leipzig" in socket.gethostname().lower():
    DATA_DIRECTORY = os.path.abspath("/mnt/t/data/product_archive/data")
else:
    DATA_DIRECTORY = os.path.abspath("T:/ECO/EOServer/data/product_archive/data")

SAFE_PATH_PATTERN = os.path.join(DATA_DIRECTORY, "Sentinel-2", "S2A", "S2_MSI_L1C", "202*", "*", "*", "*.SAFE")
SAFE_PATH = glob.glob(SAFE_PATH_PATTERN)[0] if glob.glob(SAFE_PATH_PATTERN) else "invalid_path_to_trigger_skip"


class TestAddMeteo(unittest.TestCase):
    def setUp(self) -> None:
        # A simple base dataset that we will attach aux vars to
        self.base_ds = xr.Dataset(
            data_vars={"B02": (("y", "x"), np.zeros((2, 3), dtype=np.float32))},
            coords={"y": np.arange(2), "x": np.arange(3)},
            attrs={"test_attr": "keep_me"},
        )

        # Mock layout
        self.layout = MagicMock()
        self.layout.granule_dir.return_value = "/fake/granule"
        self.layout.aux_path.side_effect = lambda name, granule=None: f"/fake/{name}.grib"

        # Dummy config (unused)
        self.config = object()

    @patch("eoio.readers.sentinel2.aux_vars.meteo.lazy_cfgrib")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.open_dataset")
    def test_returns_input_ds_if_no_requested_aux_vars(self, open_dataset_mock, lazy_cfgrib_mock):
        # proc_version doesn't matter here because we request a var that is in neither ECMWF nor CAMS lists
        self.layout.proc_version = (5, 0)

        # 'not_an_aux_var' should not trigger any opens
        out = add_meteo(
            ds=self.base_ds,
            var_names=["not_an_aux_var"],
            layout=self.layout,
            config=self.config,
        )

        self.assertIs(
            out,
            self.base_ds,
            "Function should return the original dataset object when nothing is needed.",
        )
        lazy_cfgrib_mock.assert_called_once()
        open_dataset_mock.assert_not_called()
        self.layout.aux_path.assert_not_called()

    @patch("eoio.readers.sentinel2.aux_vars.meteo.lazy_cfgrib")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.merge")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.open_dataset")
    def test_reads_ecmwf_only_when_needed(self, open_dataset_mock, merge_mock, lazy_cfgrib_mock):
        from eoio.readers.sentinel2.metadata.var_names import AUX_ECMWF_VARS_NEW

        self.layout.proc_version = (5, 0)  # use NEW list

        requested_var = AUX_ECMWF_VARS_NEW[0]
        requested = [requested_var]

        # xr.merge(parts, ...) returns aux; then code calls .rename(...)
        aux = xr.Dataset(
            data_vars={
                requested_var: (
                    ("t", "latitude", "longitude"),
                    np.ones((1, 2, 3), dtype=np.float32),
                ),
            },
            coords={"t": [0], "latitude": [50.0, 51.0], "longitude": [0.0, 1.0, 2.0]},
        )
        merge_mock.return_value = aux

        # open_dataset is called but its return value isn't used (merge_mock controls downstream)
        open_dataset_mock.return_value = xr.Dataset()

        out = add_meteo(ds=self.base_ds, var_names=requested, layout=self.layout, config=self.config)

        lazy_cfgrib_mock.assert_called_once()
        self.layout.aux_path.assert_called_once_with("AUX_ECMWFT", granule="/fake/granule")
        open_dataset_mock.assert_called_once_with("/fake/AUX_ECMWFT.grib", engine="cfgrib")
        merge_mock.assert_called_once()

        self.assertIn(requested_var, out.data_vars)

        # Check renamed coords exist on the variable we attached
        self.assertIn("latitude_aux", out[requested_var].coords)
        self.assertIn("longitude_aux", out[requested_var].coords)
        self.assertNotIn("latitude", out[requested_var].coords)
        self.assertNotIn("longitude", out[requested_var].coords)

        # Ensure original data vars preserved
        self.assertIn("B02", out.data_vars)
        self.assertEqual(out.attrs.get("test_attr"), "keep_me")

    @patch("eoio.readers.sentinel2.aux_vars.meteo.lazy_cfgrib")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.merge")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.open_dataset")
    def test_reads_cams_only_when_needed(self, open_dataset_mock, merge_mock, lazy_cfgrib_mock):
        from eoio.readers.sentinel2.metadata.var_names import AUX_CAMS_VARS

        self.layout.proc_version = (5, 0)

        aux = xr.Dataset(
            data_vars={
                "aod550": (
                    ("t", "latitude", "longitude"),
                    np.ones((1, 2, 3), dtype=np.float32),
                )
            },
            coords={"t": [0], "latitude": [50.0, 51.0], "longitude": [0.0, 1.0, 2.0]},
        )
        merge_mock.return_value = aux
        open_dataset_mock.return_value = xr.Dataset()

        requested = [AUX_CAMS_VARS[0]]
        out = add_meteo(ds=self.base_ds, var_names=requested, layout=self.layout, config=self.config)

        lazy_cfgrib_mock.assert_called_once()
        self.layout.aux_path.assert_called_once_with("AUX_CAMSFO", granule="/fake/granule")
        open_dataset_mock.assert_called_once_with("/fake/AUX_CAMSFO.grib", engine="cfgrib")
        merge_mock.assert_called_once()
        self.assertIn(requested[0], out.data_vars)

    @patch("eoio.readers.sentinel2.aux_vars.meteo.lazy_cfgrib")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.merge")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.open_dataset")
    def test_reads_both_sources_when_needed(self, open_dataset_mock, merge_mock, lazy_cfgrib_mock):
        from eoio.readers.sentinel2.metadata.var_names import (
            AUX_ECMWF_VARS_NEW,
            AUX_CAMS_VARS,
        )

        self.layout.proc_version = (5, 0)

        ecmwf_var = AUX_ECMWF_VARS_NEW[0]
        cams_var = AUX_CAMS_VARS[0]
        requested = [ecmwf_var, cams_var]

        # Merge returns a dataset containing *both* requested vars + lat/lon
        aux = xr.Dataset(
            data_vars={
                ecmwf_var: (
                    ("t", "latitude", "longitude"),
                    np.ones((1, 2, 3), dtype=np.float32),
                ),
                cams_var: (
                    ("t", "latitude", "longitude"),
                    np.ones((1, 2, 3), dtype=np.float32) * 2,
                ),
            },
            coords={"t": [0], "latitude": [50.0, 51.0], "longitude": [0.0, 1.0, 2.0]},
        )
        merge_mock.return_value = aux
        open_dataset_mock.return_value = xr.Dataset()

        out = add_meteo(ds=self.base_ds, var_names=requested, layout=self.layout, config=self.config)

        lazy_cfgrib_mock.assert_called_once()

        # aux_path called for both sources (order: ECMWF then CAMS per function)
        self.assertEqual(self.layout.aux_path.call_count, 2)
        self.layout.aux_path.assert_any_call("AUX_ECMWFT", granule="/fake/granule")
        self.layout.aux_path.assert_any_call("AUX_CAMSFO", granule="/fake/granule")

        # open_dataset called twice with the correct paths
        self.assertEqual(open_dataset_mock.call_count, 2)
        open_dataset_mock.assert_any_call("/fake/AUX_ECMWFT.grib", engine="cfgrib")
        open_dataset_mock.assert_any_call("/fake/AUX_CAMSFO.grib", engine="cfgrib")

        merge_mock.assert_called_once()

        self.assertIn(ecmwf_var, out.data_vars)
        self.assertIn(cams_var, out.data_vars)

        # sanity check the rename happened (coords renamed on attached variables)
        self.assertIn("latitude_aux", out[ecmwf_var].coords)
        self.assertIn("longitude_aux", out[ecmwf_var].coords)
        self.assertNotIn("latitude", out[ecmwf_var].coords)
        self.assertNotIn("longitude", out[ecmwf_var].coords)

    @patch("eoio.readers.sentinel2.aux_vars.meteo.lazy_cfgrib")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.merge")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.open_dataset")
    def test_missing_requested_var_raises_keyerror(self, open_dataset_mock, merge_mock, lazy_cfgrib_mock):
        from eoio.readers.sentinel2.metadata.var_names import AUX_ECMWF_VARS_NEW

        self.layout.proc_version = (5, 0)

        # Merge returns a dataset without the requested var
        aux = xr.Dataset(
            data_vars={
                "some_other_var": (
                    ("t", "latitude", "longitude"),
                    np.ones((1, 2, 3), dtype=np.float32),
                )
            },
            coords={"t": [0], "latitude": [50.0, 51.0], "longitude": [0.0, 1.0, 2.0]},
        )
        merge_mock.return_value = aux
        open_dataset_mock.return_value = xr.Dataset()

        requested = [AUX_ECMWF_VARS_NEW[0]]  # not present in aux
        with self.assertRaises(KeyError) as ctx:
            add_meteo(
                ds=self.base_ds,
                var_names=requested,
                layout=self.layout,
                config=self.config,
            )

        self.assertIn("Requested aux vars not found", str(ctx.exception))

    @patch("eoio.readers.sentinel2.aux_vars.meteo.lazy_cfgrib")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.merge")
    @patch("eoio.readers.sentinel2.aux_vars.meteo.xr.open_dataset")
    def test_ecmwf_var_list_switches_on_proc_version(self, open_dataset_mock, merge_mock, lazy_cfgrib_mock):
        """
        If proc_version major < 5 we use AUX_ECMWF_VARS_OLD; otherwise AUX_ECMWF_VARS_NEW.
        This test checks the OLD branch triggers ECMWF reading when requesting an OLD-list var.
        """
        from eoio.readers.sentinel2.metadata.var_names import AUX_ECMWF_VARS_OLD

        self.layout.proc_version = (4, 9)  # OLD list

        requested_var = AUX_ECMWF_VARS_OLD[0]
        requested = [requested_var]

        # Merge returns a dataset that *does* contain the requested var + lat/lon
        aux = xr.Dataset(
            data_vars={
                requested_var: (
                    ("t", "latitude", "longitude"),
                    np.ones((1, 2, 3), dtype=np.float32),
                ),
            },
            coords={"t": [0], "latitude": [50.0, 51.0], "longitude": [0.0, 1.0, 2.0]},
        )
        merge_mock.return_value = aux
        open_dataset_mock.return_value = xr.Dataset()

        out = add_meteo(ds=self.base_ds, var_names=requested, layout=self.layout, config=self.config)

        lazy_cfgrib_mock.assert_called_once()
        self.layout.aux_path.assert_called_once_with("AUX_ECMWFT", granule="/fake/granule")
        open_dataset_mock.assert_called_once_with("/fake/AUX_ECMWFT.grib", engine="cfgrib")
        merge_mock.assert_called_once()

        self.assertIn(requested_var, out.data_vars)


class TestMeteoData(unittest.TestCase):
    def test_real_safe(self):
        safe_path = Path(SAFE_PATH)
        if not safe_path.exists():
            self.skipTest(f"Test SAFE product not found at {safe_path}")

        # Minimal config that should work with your new "roi=None => no resolver" behaviour
        _vars_sel = {"meas": "rgb", "aux": None}
        _subset = {"roi": None, "roi_crs": None}

        layout = S2Layout(safe_path)
        config = ReaderConfig(vars_sel={}, subset={}, read_params={})

        ds = xr.Dataset()
        ds = add_meteo(ds=ds, var_names=["tcwv", "aod550"], layout=layout, config=config)

        pass


if __name__ == "__main__":
    unittest.main()
