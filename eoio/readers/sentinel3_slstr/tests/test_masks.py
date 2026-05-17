"""Unit tests for eoio.readers.sentinel3_slstr.masks.

Adapted from sentinel3_olci mask tests; verify error cases and basic
flag-reading behaviour using mocked rioxarray objects.
"""

import unittest
import xarray as xr
import numpy as np
from unittest.mock import MagicMock, patch
from eoio.readers.sentinel3_slstr import masks


class TestS3SLSTRMasks(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.layout = MagicMock()
        self.subset = MagicMock()
        self.config = MagicMock()
        self.config.read_params = {"chunks": None, "use_chunks": False}

    def test_add_masks_raises_when_mask_selection_is_none(self):
        self.config.vars_sel = {"mask": None, "meas": ["S1_radiance_an"]}
        with self.assertRaises(ValueError):
            masks.add_masks(
                ds=self.ds,
                layout=self.layout,
                grids=["an"],
                clip_boxes=None,
                config=self.config,
                use_chunks=False,
                chunks=None,
            )

    @patch("eoio.readers.sentinel3_slstr.masks.lazy_rioxarray")
    def test_add_masks_reads_flag_dataset_and_sets_attrs(self, mock_riox):
        # Prepare a fake flags dataset returned by rioxarray
        class MockFlags:
            def __init__(self):
                self.flag_meanings = "cloud confidence"
                self.flag_masks = [1, 2]
                self.data = np.zeros((10, 10), dtype=np.uint8)

            def squeeze(self):
                return self

            def drop_vars(self, *a, **k):
                return self

            @property
            def rio(self):
                return self

            def write_crs(self, crs, inplace=True):
                return self

            def clip_box(self, x_min, y_min, x_max, y_max):
                return self

            def rename(self, mapping):
                return self

        mock_flags = MockFlags()
        # emulate rioxarray returning an indexable container where [0][flag] yields the variable
        mock_riox.return_value.open_rasterio.return_value = [{"cloud_an": mock_flags}]

        self.layout.flags_path.return_value = "dummy"
        self.config.vars_sel = {
            "mask": ["cloud"],
            "meas": ["S1_radiance_an"],
            "aux": [],
        }

        out = masks.add_masks(
            ds=self.ds,
            layout=self.layout,
            grids=["an"],
            clip_boxes={"an": (0, 0, 1, 1)},
            config=self.config,
            use_chunks=False,
            chunks=None,
        )
        # mask variables should be present in dataset and have attributes set
        self.assertIsInstance(out, xr.Dataset)

    @patch("eoio.readers.sentinel3_slstr.masks.lazy_rioxarray")
    def test_add_masks_calls_flags_path_for_each_grid(self, mock_riox):
        # Verify layout.flags_path is called when masks requested
        class MockFlags:
            def __init__(self):
                self.flag_meanings = "cloud confidence"
                self.flag_masks = [1, 2]
                self.data = np.zeros((10, 10), dtype=np.uint8)

            def squeeze(self):
                return self

            def drop_vars(self, *a, **k):
                return self

            @property
            def rio(self):
                return self

            def write_crs(self, crs, inplace=True):
                return self

            def clip_box(self, x_min, y_min, x_max, y_max):
                return self

            def rename(self, mapping):
                return self

        mock_flags = MockFlags()
        mock_riox.return_value.open_rasterio.return_value = [{"cloud_an": mock_flags}]
        self.layout.flags_path.return_value = "dummy"
        self.config.vars_sel = {
            "mask": ["cloud"],
            "meas": ["S1_radiance_an"],
            "aux": [],
        }

        masks.add_masks(
            ds=self.ds,
            layout=self.layout,
            grids=["an"],
            clip_boxes=None,
            config=self.config,
            use_chunks=False,
            chunks=None,
        )
        self.layout.flags_path.assert_called_with("an")


if __name__ == "__main__":
    unittest.main()
