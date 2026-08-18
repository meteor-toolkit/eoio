import unittest
from unittest.mock import Mock, patch
import xarray as xr

from eoio.readers.copernicus_dem.reader import CopernicusDEMReader


class TestCopernicusDEMReader(unittest.TestCase):

    @patch("eoio.readers.copernicus_dem.reader.apply_conventions")
    @patch("eoio.readers.copernicus_dem.reader.add_masks")
    @patch("eoio.readers.copernicus_dem.reader.append_data_vars")
    def test_open_dataset_with_meas_masks_and_metadata(
        self,
        mock_append_data_vars,
        mock_add_masks,
        mock_apply_conventions,
    ):
        reader = CopernicusDEMReader.__new__(CopernicusDEMReader)

        reader.layout = Mock()
        reader.layout.dem = "/tmp/dem.tif"

        reader.resolved_config = Mock()
        reader.resolved_config.vars_sel = {
            "meas": ["elevation"],
            "mask": ["water_body_mask"],
        }
        reader.resolved_config.subset = None
        reader.resolved_config.read_params = {
            "metadata_level": "all",
            "use_chunks": True,
            "chunks": {"x": 256, "y": 256},
        }

        reader.mtd = Mock()

        ds_after_append = xr.Dataset({"elevation": xr.DataArray([1])})
        ds_after_metadata = xr.Dataset({"elevation": xr.DataArray([1])})
        ds_after_masks = xr.Dataset(
            {
                "elevation": xr.DataArray([1]),
                "water_body_mask": xr.DataArray([0]),
            }
        )
        ds_final = xr.Dataset({"final": xr.DataArray([1])})

        mock_append_data_vars.return_value = ds_after_append
        reader.mtd.attach_metadata.return_value = ds_after_metadata
        mock_add_masks.return_value = ds_after_masks
        mock_apply_conventions.return_value = ds_final

        result = reader.open_dataset()

        mock_append_data_vars.assert_called_once_with(
            ds=unittest.mock.ANY,
            layout=reader.layout.dem,
            subset=None,
            use_chunks=True,
            chunks={"x": 256, "y": 256},
        )

        reader.mtd.attach_metadata.assert_called_once_with(
            ds_after_append,
            level="all",
        )

        mock_add_masks.assert_called_once_with(
            ds=ds_after_metadata,
            masks=["water_body_mask"],
            layout=reader.layout,
            subset=None,
            config=reader.resolved_config,
        )

        mock_apply_conventions.assert_called_once_with(
            ds_after_masks,
            layout=reader.layout,
            config=reader.resolved_config,
        )

        self.assertIs(result, ds_final)

    @patch("eoio.readers.copernicus_dem.reader.apply_conventions")
    @patch("eoio.readers.copernicus_dem.reader.append_data_vars")
    def test_open_dataset_passes_chunking_params(
        self,
        mock_append_data_vars,
        mock_apply_conventions,
    ):
        reader = CopernicusDEMReader.__new__(CopernicusDEMReader)

        reader.layout = Mock()
        reader.layout.dem = "/tmp/dem.tif"

        reader.resolved_config = Mock()
        reader.resolved_config.vars_sel = {
            "meas": ["elevation"],
            "mask": [],
        }
        reader.resolved_config.subset = None
        reader.resolved_config.read_params = {
            "metadata_level": None,
            "use_chunks": True,
            "chunks": {"x": 512, "y": 512},
        }

        reader.mtd = Mock()

        ds = xr.Dataset()
        mock_append_data_vars.return_value = ds
        mock_apply_conventions.return_value = ds

        reader.open_dataset()

        mock_append_data_vars.assert_called_once_with(
            ds=unittest.mock.ANY,
            layout=reader.layout.dem,
            subset=None,
            use_chunks=True,
            chunks={"x": 512, "y": 512},
        )

    @patch("eoio.readers.copernicus_dem.reader.apply_conventions")
    @patch("eoio.readers.copernicus_dem.reader.append_data_vars")
    def test_open_dataset_without_measurements(
        self,
        mock_append_data_vars,
        mock_apply_conventions,
    ):
        reader = CopernicusDEMReader.__new__(CopernicusDEMReader)

        reader.layout = Mock()

        reader.resolved_config = Mock()
        reader.resolved_config.vars_sel = {
            "meas": [],
            "mask": [],
        }
        reader.resolved_config.subset = None
        reader.resolved_config.read_params = {
            "metadata_level": None,
        }

        reader.mtd = Mock()

        ds = xr.Dataset()
        mock_apply_conventions.return_value = ds

        reader.open_dataset()

        mock_append_data_vars.assert_not_called()
        
if __name__ == "__main__":
    unittest.main()