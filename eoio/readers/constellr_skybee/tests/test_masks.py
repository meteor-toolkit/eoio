"""eoio.readers.constellr_skybee.tests.test_masks - unit tests for eoio.readers.constellr_skybee.masks"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from eoio.readers.constellr_skybee.masks import MASK_OPTIONS, add_masks


def _write_qa(path: Path):
    flags = np.zeros((3, 2, 2), dtype="uint8")
    flags[0] = [[1, 0], [0, 0]]  # blackfill
    flags[1, 1, 1] = 1  # saturated
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32733",
        transform=from_origin(0, 20, 10, 10),
        nodata=255,
    ) as dst:
        dst.write(flags)


class TestSkybeeMasks(unittest.TestCase):
    def test_mask_options(self):
        self.assertEqual(MASK_OPTIONS, ["blackfill_mask", "saturated_mask", "untested_mask"])

    def test_add_masks_reads_flags_on_10m_grid(self):
        with tempfile.TemporaryDirectory() as td:
            qa = Path(td) / "vnir04_qa.tiff"
            _write_qa(qa)
            layout = MagicMock()
            layout.qa_file.return_value = str(qa)
            config = SimpleNamespace(read_params={})

            out = add_masks(
                ds=xr.Dataset(),
                masks=["blackfill_mask", "saturated_mask"],
                layout=layout,
                subset=None,
                config=config,
            )

        layout.qa_file.assert_called_once_with("B4")
        self.assertEqual(out["blackfill_mask"].dims, ("y_10m", "x_10m"))
        self.assertEqual(out["blackfill_mask"].values.tolist(), [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(out["saturated_mask"].values[1, 1], 1.0)
        self.assertEqual(out["blackfill_mask"].attrs["flag_meanings"], "valid_data no_data")
        self.assertNotIn("untested_mask", out)

    def test_add_masks_raises_for_unknown_mask(self):
        with self.assertRaises(ValueError):
            add_masks(ds=xr.Dataset(), masks=["nope"], layout=MagicMock(), subset=None, config=SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
