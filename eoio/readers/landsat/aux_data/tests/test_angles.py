"""eoio.readers.landsat.tests.test_angles - unit tests for eoio.readers.landsat.aux_data.angles"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from eoio.readers.landsat.aux_data.angles import read_angles_into_dataset


class TestReadAnglesIntoDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = MagicMock()
        self.mtd = MagicMock()
        self.subset = MagicMock()
        self.subset.clip_box = None
        self.subset.geometries = None

        self.angle_paths = {"viewing_zenith_angle": "/tmp/fake_VZA.TIF"}
        self.layout.get_angle_files.return_value = self.angle_paths

        # Default metadata
        self.mtd.get_angle_metadata.return_value = {"viewing_zenith_angle": {"geometry_id": None}}

    def _make_da(self, values: np.ndarray) -> xr.DataArray:
        da = xr.DataArray(
            values,
            dims=("band", "y", "x"),
            coords={
                "band": [1],
                "y": np.arange(values.shape[1]),
                "x": np.arange(values.shape[2]),
            },
        )
        return da

    @patch("eoio.readers.landsat.aux_data.angles.lazy_rioxarray")
    def test_read_angles_into_dataset_BasicUse(self, mock_lazy_rioxarray):
        # Input 1000. Scaling / 100.0 -> 10.0
        da = self._make_da(np.array([[[1000]]], dtype=np.uint16))

        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        read_angles_into_dataset(
            ds=ds,
            layout=self.layout,
            angle_vars=["viewing_zenith_angle"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("viewing_zenith_angle", ds)
        self.assertAlmostEqual(ds["viewing_zenith_angle"].values[0, 0], 10.0, places=4)

    @patch("eoio.readers.landsat.aux_data.angles.lazy_rioxarray")
    def test_read_angles_into_dataset_IgnoresUnrequestedVars(self, mock_lazy_rioxarray):
        mock_lazy_rioxarray.return_value.open_rasterio.side_effect = Exception("Should not read")

        ds = xr.Dataset()
        read_angles_into_dataset(
            ds=ds,
            layout=self.layout,
            angle_vars=["other_angle"],  # Not in layout mock
            subset=None,
            mtd=self.mtd,
        )
        # Should not raise, just return empty/same ds (unless other_angle IS in layout)
        # In setup, only 'viewing_zenith_angle' is in layout.
        self.assertNotIn("other_angle", ds)

    @patch("eoio.readers.landsat.aux_data.angles.lazy_rioxarray")
    def test_read_angles_into_dataset_RenamesDimsIfGeometryPresent(self, mock_lazy_rioxarray):
        self.mtd.get_angle_metadata.return_value = {"viewing_zenith_angle": {"geometry_id": "viewing"}}

        da = self._make_da(np.array([[[1000]]], dtype=np.uint16))
        rxr = MagicMock()
        rxr.open_rasterio.return_value = da
        mock_lazy_rioxarray.return_value = rxr

        ds = xr.Dataset()
        read_angles_into_dataset(
            ds=ds,
            layout=self.layout,
            angle_vars=["viewing_zenith_angle"],
            subset=None,
            mtd=self.mtd,
        )

        self.assertIn("viewing_zenith_angle", ds)
        self.assertIn("x_viewing", ds["viewing_zenith_angle"].dims)
        self.assertIn("y_viewing", ds["viewing_zenith_angle"].dims)


if __name__ == "__main__":
    unittest.main()
