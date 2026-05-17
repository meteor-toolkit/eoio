"""eoio.readers.landsat.tests.test_aux - unit tests for eoio.readers.landsat.aux_data.aux_data"""

import unittest
from unittest.mock import MagicMock, patch
import xarray as xr
from eoio.readers.landsat.aux_data.aux_data import LSAuxData


class TestLSAuxData(unittest.TestCase):
    def setUp(self):
        self.reader = MagicMock()
        self.reader.layout.get_angle_files.return_value = {"viewing_zenith_angle": "/path/to/vza.tif"}
        self.reader.resolved_config.subset = "fake_subset"
        self.reader.resolved_config.read_params = {"use_chunks": False, "chunks": None}
        self.reader.mtd = "fake_mtd"

        self.aux = LSAuxData(self.reader)

    def test_attach_aux_NoAuxVars_ReturnsSameDs(self):
        ds = xr.Dataset()
        out = self.aux.attach_aux(ds, [])
        self.assertIs(out, ds)

    @patch("eoio.readers.landsat.aux_data.aux_data.read_angles_into_dataset")
    def test_attach_aux_WithAngleVars_CallsReadAngles(self, mock_read_angles):
        ds = xr.Dataset()
        ds_out = xr.Dataset({"vza": (("y", "x"), [[1]])})
        mock_read_angles.return_value = ds_out

        out = self.aux.attach_aux(ds, ["viewing_zenith_angle"])

        self.assertIs(out, ds_out)
        mock_read_angles.assert_called_once_with(
            ds=ds,
            layout=self.reader.layout,
            angle_vars=["viewing_zenith_angle"],
            subset="fake_subset",
            mtd="fake_mtd",
            use_chunks=False,
            chunks=None,
        )

    @patch("eoio.readers.landsat.aux_data.aux_data.read_angles_into_dataset")
    def test_attach_aux_WithNonAngleVars_DoesNotCallReadAngles(self, mock_read_angles):
        ds = xr.Dataset()
        # "unknown_var" is not in layout.get_angle_files()
        out = self.aux.attach_aux(ds, ["unknown_var"])

        self.assertIs(out, ds)
        mock_read_angles.assert_not_called()


if __name__ == "__main__":
    unittest.main()
