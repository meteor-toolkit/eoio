"""eoio.readers.airbus_pleiades.tests.test_conventions - unit tests for eoio.readers.airbus_pleiades.conventions"""

import unittest
import xarray as xr
from unittest.mock import Mock, patch
from eoio.readers.airbus_pleiades.conventions import apply_conventions


class TestPleiadesConventions(unittest.TestCase):
    """Unit tests for Pleiades conventions method."""

    @patch("eoio.readers.airbus_pleiades.conventions.PleiadesLayout")
    @patch("eoio.readers.airbus_pleiades.conventions.ResolvedROISubset")
    def test_apply_conventions_returns_dataset(self, mock_layout, mock_subset):
        """Test that apply_conventions returns dataset."""

        mock_config = Mock()
        _clip_box = Mock()
        mock_subset.clip_box = "test_subset"

        result = apply_conventions(
            ds=xr.Dataset(),
            layout=mock_layout,
            roi_subset=mock_subset,
            config=mock_config,
        )

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("eoio:reader", result.attrs)
        self.assertEqual(result.attrs["eoio:subset"], "test_subset")


if __name__ == "__main__":
    unittest.main()
