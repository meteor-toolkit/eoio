"""Tests for eoio.processors.fill.processor"""

import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from eoio.processors.populate_grid.processor import PopulateGrid


class TestPopulateGridProcessor(unittest.TestCase):
    def test_parse_params_defaults_to_expected_values(self):
        """Parser should keep the provided config values and default the rest."""
        processor = PopulateGrid(params={"reference_grid": "ref.nc", "fill_value": -999, "tolerance": 0.5})

        self.assertEqual(processor.populate_grid_config.reference_grid, "ref.nc")
        self.assertEqual(processor.populate_grid_config.fill_value, -999)
        self.assertEqual(processor.populate_grid_config.tolerance, 0.5)

    def test_parse_params_uses_defaults_when_values_are_missing(self):
        """Parser should apply the default tolerance and fill values when not supplied."""
        processor = PopulateGrid(params={})

        self.assertIsNone(processor.populate_grid_config.reference_grid)
        self.assertIsNone(processor.populate_grid_config.fill_value)
        self.assertEqual(processor.populate_grid_config.tolerance, 1)

    def test_format_dims_to_populate_raises_when_required_variables_are_missing(self):
        """Reference-grid formatting should reject datasets without the required viewing vars."""
        processor = PopulateGrid(params={})
        ds = xr.Dataset(
            data_vars={
                "reflectance": (("series",), np.array([1.0, 2.0])),
                "viewing_zenith_angle": (("series",), np.array([10.0, 20.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0, 40.0])),
            },
            coords={"series": [0, 1], "wavelength": [0.5]},
        )
        ref_ds = xr.Dataset(
            data_vars={
                "viewing_zenith_angle": ("series", [10.0]),
                "other_var": ("series", [1.0]),
            },
            coords={"series": [0]},
        )

        with self.assertRaises(ValueError):
            processor._format_dims_to_populate(ds, ref_ds, ["viewing_zenith_angle", "viewing_azimuth_angle"])

    @patch("eoio.processors.populate_grid.processor.xr.open_dataset")
    def test_format_reference_grid_opens_dataset_from_path(self, mock_open_dataset):
        """Reference-grid formatting should open a dataset from a filepath when given a string."""
        processor = PopulateGrid(params={})
        ref_ds = xr.Dataset(
            data_vars={
                "viewing_zenith_angle": ("series", [10.0]),
                "viewing_azimuth_angle": ("series", [20.0]),
            },
            coords={"series": [0]},
        )
        mock_open_dataset.return_value = ref_ds

        formatted = processor._format_reference_grid("ref.nc")

        self.assertEqual(set(formatted.data_vars), {"viewing_zenith_angle", "viewing_azimuth_angle"})
        mock_open_dataset.assert_called_once_with("ref.nc")

    def test_regrid_series_reuses_matching_series_positions(self):
        """Regridding should place existing series values into the matching reference positions."""
        processor = PopulateGrid(params={"fill_value": -1, "tolerance": 1.0})

        ds = xr.Dataset(
            data_vars={
                "reflectance": (("series",), np.array([1.0, 2.0])),
                "viewing_zenith_angle": (("series",), np.array([10.0, 20.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0, 40.0])),
            },
            coords={"series": [0, 1], "wavelength": [0.5]},
        )

        ref_ds = xr.Dataset(
            data_vars={
                "viewing_zenith_angle": (("series",), np.array([10.0, 20.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0, 40.0])),
            },
            coords={"series": [0, 1]},
        )

        out = processor.regrid_series(ds, ref_ds, ["viewing_zenith_angle", "viewing_azimuth_angle"])

        self.assertEqual(out.sizes["series"], 2)
        self.assertEqual(out["reflectance"].shape, (2,))
        self.assertEqual(out["reflectance"].values.tolist(), [1.0, 2.0])

    def test_regrid_series_fills_unmatched_series_with_fill_value(self):
        """Regridding should fill unmatched positions with the configured fill value."""
        processor = PopulateGrid(params={"fill_value": -1, "tolerance": 0.5})

        ds = xr.Dataset(
            data_vars={
                "reflectance": (("series",), np.array([1.0, 2.0])),
                "viewing_zenith_angle": (("series",), np.array([10.0, 15.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0, 35.0])),
            },
            coords={"series": [0, 1], "wavelength": [0.5]},
        )

        ref_ds = xr.Dataset(
            data_vars={
                "viewing_zenith_angle": (("series",), np.array([10.0, 20.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0, 40.0])),
            },
            coords={"series": [0, 1]},
        )

        out = processor.regrid_series(ds, ref_ds, ["viewing_zenith_angle", "viewing_azimuth_angle"])

        self.assertEqual(out["reflectance"].values.tolist(), [1.0, -1.0])

    def test_regrid_series_keeps_variables_without_series_dimension(self):
        """Regridding should leave non-series variables unchanged when copying the dataset."""
        processor = PopulateGrid(params={"fill_value": -1, "tolerance": 1.0})

        ds = xr.Dataset(
            data_vars={
                "quality": (("wavelength",), np.array([7.0])),
                "reflectance": (("series",), np.array([1.0])),
                "viewing_zenith_angle": (("series",), np.array([10.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0])),
            },
            coords={"series": [0], "wavelength": [0.5]},
        )

        ref_ds = xr.Dataset(
            data_vars={
                "viewing_zenith_angle": (("series",), np.array([10.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0])),
            },
            coords={"series": [0]},
        )

        out = processor.regrid_series(ds, ref_ds, ["viewing_zenith_angle", "viewing_azimuth_angle"])

        self.assertIn("quality", out.data_vars)
        self.assertTrue(np.array_equal(out["quality"].values, ds["quality"].values))

    def test_run_regrids_input_dataset_using_reference_grid(self):
        """Run should use the formatted reference grid to produce a regridded dataset."""
        processor = PopulateGrid(params={"fill_value": -999, "tolerance": 1.0})
        ds = xr.Dataset(
            data_vars={
                "reflectance": (("series",), np.array([1.0, 2.0])),
                "viewing_zenith_angle": (("series",), np.array([10.0, 15.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0, 35.0])),
            },
            coords={"series": [0, 1], "wavelength": [0.5]},
        )
        reference_grid = xr.Dataset(
            data_vars={
                "viewing_zenith_angle": (("series",), np.array([10.0, 20.0])),
                "viewing_azimuth_angle": (("series",), np.array([30.0, 40.0])),
            },
            coords={"series": [0, 1]},
        )

        with patch.object(processor, "_format_reference_grid", return_value=reference_grid):
            out = processor.run(ds)

        self.assertEqual(out["reflectance"].values.tolist(), [1.0, -999.0])

    def test_run_raises_on_non_dataset_input(self):
        """Run should reject non-xarray dataset inputs with a clear error."""
        processor = PopulateGrid(params={})

        with self.assertRaises(TypeError):
            processor.run("not-a-dataset")


if __name__ == "__main__":
    unittest.main()
