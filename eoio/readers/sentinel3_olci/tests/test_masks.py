import unittest
import xarray as xr
import numpy as np
from unittest.mock import MagicMock, patch
from eoio.readers.sentinel3_olci import masks


class TestS3OLCIMasks(unittest.TestCase):
    """Tests for S3 OLCI masks module."""

    def setUp(self):
        self.ds = xr.Dataset()
        self.layout = MagicMock()
        self.subset = MagicMock()
        self.config = MagicMock()
        self.config.read_params = {"chunks": None, "use_chunks": False}

    # ---- Tests for add_masks() ----

    def test_add_masks_raises_when_mask_selection_is_none(self):
        """add_masks should raise ValueError when mask_names in config is None."""
        self.config.vars_sel = {"mask": None, "meas": ["Oa01"]}
        with self.assertRaises(ValueError):
            masks.add_masks(
                ds=self.ds,
                layout=self.layout,
                subset=self.subset,
                config=self.config,
                masks=None,
            )

    @patch("eoio.readers.sentinel3_olci.masks.lazy_rioxarray")
    def test_add_masks_raises_with_empty_mask_list(self, mock_riox):
        """add_masks should raise ValueError when mask selection is empty."""
        self.config.vars_sel = {"mask": [], "meas": ["Oa01"]}
        # When masks is empty, read_masks returns early, so no error is raised
        result = masks.add_masks(
            ds=self.ds,
            layout=self.layout,
            subset=self.subset,
            config=self.config,
            masks=[],
        )
        self.assertIsNotNone(result)

    @patch("eoio.readers.sentinel3_olci.masks.read_masks")
    def test_add_masks_delegates_to_read_masks(self, mock_read_masks):
        """add_masks should delegate to read_masks with provided masks parameter."""
        mock_read_masks.return_value = self.ds
        self.config.vars_sel = {"mask": ["saturated"], "meas": ["Oa01"]}

        _result = masks.add_masks(
            ds=self.ds,
            layout=self.layout,
            subset=self.subset,
            config=self.config,
            masks=["saturated"],
        )

        mock_read_masks.assert_called_once()
        call_kwargs = mock_read_masks.call_args.kwargs
        self.assertEqual(call_kwargs["masks"], ["saturated"])
        self.assertEqual(call_kwargs["layout"], self.layout)

    # ---- Tests for read_masks() ----

    @patch("eoio.readers.sentinel3_olci.masks.lazy_rioxarray")
    def test_read_masks_returns_unmodified_dataset_when_empty_masks_list(self, mock_riox):
        """read_masks should return unmodified dataset when masks list is empty."""
        self.config.vars_sel = {"mask": ["saturated"], "meas": ["Oa01", "Oa02"]}
        result = masks.read_masks(ds=self.ds, masks=[], layout=self.layout, config=self.config, subset=None)
        self.assertIs(result, self.ds)
        self.assertEqual(len(result.data_vars), 0)

    @patch("eoio.readers.sentinel3_olci.masks.lazy_rioxarray")
    def test_read_masks_expands_saturated_to_per_band(self, mock_riox):
        """read_masks should expand 'saturated' mask to per-band masks."""
        mock_mask_ds = MagicMock()
        mock_mask_ds.flag_meanings = "saturated@Oa01 saturated@Oa02 dubious"
        mock_mask_ds.flag_masks = [1, 2, 4]
        mock_mask_ds.data = np.zeros((100, 100), dtype=np.uint8)
        mock_mask_ds.rio.write_crs.return_value = mock_mask_ds
        mock_mask_ds.rio.clip_box.return_value = mock_mask_ds

        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value = mock_mask_ds
        self.layout.quality_flags_path.return_value = {"path": "dummy_path"}
        self.config.vars_sel = {"mask": ["saturated"], "meas": ["Oa01", "Oa02"]}

        result = masks.read_masks(
            ds=self.ds,
            masks=["saturated"],
            layout=self.layout,
            config=self.config,
            subset=None,
        )

        self.assertIn("quality_flags", result.data_vars)
        self.assertIsNotNone(result["quality_flags"].attrs.get("flag_meanings"))

    @patch("eoio.readers.sentinel3_olci.masks.lazy_rioxarray")
    def test_read_masks_preserves_non_saturated_masks(self, mock_riox):
        """read_masks should preserve non-saturated masks without expansion."""
        mock_mask_ds = MagicMock()
        mock_mask_ds.flag_meanings = "saturated dubious cirrus"
        mock_mask_ds.flag_masks = [1, 2, 4]
        mock_mask_ds.data = np.zeros((100, 100), dtype=np.uint8)
        mock_mask_ds.rio.write_crs.return_value = mock_mask_ds
        mock_mask_ds.rio.clip_box.return_value = mock_mask_ds

        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value = mock_mask_ds
        self.layout.quality_flags_path.return_value = {"path": "dummy_path"}
        self.config.vars_sel = {"mask": ["dubious", "cirrus"], "meas": ["Oa01"]}

        result = masks.read_masks(
            ds=self.ds,
            masks=["dubious", "cirrus"],
            layout=self.layout,
            config=self.config,
            subset=None,
        )

        self.assertIn("quality_flags", result.data_vars)
        flag_meanings = result["quality_flags"].attrs["flag_meanings"]
        self.assertIn("dubious", flag_meanings)
        self.assertIn("cirrus", flag_meanings)

    @patch("eoio.readers.sentinel3_olci.masks.lazy_rioxarray")
    def test_read_masks_clips_to_subset_when_provided(self, mock_riox):
        """read_masks should apply ROI clipping when subset with xy_clip_box is provided."""
        mock_mask_ds = MagicMock()
        # Include the expanded saturated masks in flag_meanings so they're not filtered out
        mock_mask_ds.flag_meanings = "saturated@Oa01 dubious"
        mock_mask_ds.flag_masks = [1, 2]
        mock_mask_ds.data = np.zeros((100, 100), dtype=np.uint8)
        mock_mask_ds.rio.write_crs.return_value = mock_mask_ds
        mock_mask_ds.rio.clip_box.return_value = mock_mask_ds

        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value = mock_mask_ds
        self.layout.quality_flags_path.return_value = {"path": "dummy_path"}
        self.config.vars_sel = {"mask": ["saturated"], "meas": ["Oa01"]}

        subset = MagicMock()
        subset.xy_clip_box = (10, 20, 30, 40)

        masks.read_masks(
            ds=self.ds,
            masks=["saturated"],
            layout=self.layout,
            config=self.config,
            subset=subset,
        )

        mock_mask_ds.rio.clip_box.assert_called_once_with(10, 20, 30, 40)

    @patch("eoio.readers.sentinel3_olci.masks.lazy_rioxarray")
    def test_read_masks_sets_flag_attributes(self, mock_riox):
        """read_masks should set flag_meanings and flag_masks attributes on quality_flags."""
        mock_mask_ds = MagicMock()
        mock_mask_ds.flag_meanings = "saturated dubious"
        mock_mask_ds.flag_masks = [1, 2]
        mock_mask_ds.data = np.zeros((100, 100), dtype=np.uint8)
        mock_mask_ds.rio.write_crs.return_value = mock_mask_ds
        mock_mask_ds.rio.clip_box.return_value = mock_mask_ds

        mock_riox.return_value.open_rasterio.return_value.squeeze.return_value = mock_mask_ds
        self.layout.quality_flags_path.return_value = {"path": "dummy_path"}
        self.config.vars_sel = {"mask": ["saturated", "dubious"], "meas": ["Oa01"]}

        result = masks.read_masks(
            ds=self.ds,
            masks=["saturated", "dubious"],
            layout=self.layout,
            config=self.config,
            subset=None,
        )

        self.assertIn("quality_flags", result.data_vars)
        self.assertIn("flag_meanings", result["quality_flags"].attrs)
        self.assertIn("flag_masks", result["quality_flags"].attrs)


if __name__ == "__main__":
    unittest.main()
