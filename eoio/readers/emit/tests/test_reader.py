"""Tests for eoio.readers.emit.reader"""

import unittest
from unittest.mock import patch, MagicMock
import xarray as xr
from eoio.readers.emit import reader


class TestEMITL1BReader(unittest.TestCase):
    def _make_reader(self):
        r = MagicMock(spec=reader.EMITL1BReader)
        r.layout = MagicMock()
        r.layout.path = "dummy_RAD"
        r.ds_src = xr.Dataset({"radiance": (["y", "x"], [[1]])})
        r.config = MagicMock()
        r.config.subset = {}
        r.config.vars_sel = {"aux": None}
        r.config.read_params = {}
        r.list_selected_meas = MagicMock(return_value=["radiance"])
        return r

    @patch(
        "eoio.readers.emit.reader.read_dataset",
        lambda ds, src, meas_vars, subset: xr.Dataset({"radiance": (["y", "x"], [[1]])}),
    )
    @patch("eoio.readers.emit.reader.build_subset", lambda ds, subset: {})
    def test_open_dataset_basic(self):
        r = self._make_reader()
        with patch.object(reader.EMITL1BReader, "open_dataset", reader.EMITL1BReader.open_dataset):
            ds = reader.EMITL1BReader.open_dataset(r)
            self.assertIsInstance(ds, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
