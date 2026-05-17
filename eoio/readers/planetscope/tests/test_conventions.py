"""
eoio.readers.planetscope.tests.test_conventions - tests for eoio.readers.planetscope.conventions
"""

import unittest
import xarray as xr
from unittest.mock import Mock, patch
from eoio.readers.planetscope.conventions import apply_conventions


class TestPlanetScopeConventions(unittest.TestCase):
    """Unit tests for PlanetScope conventions method."""

    @patch("eoio.readers.planetscope.conventions.PlanetScopeLayout")
    @patch("eoio.readers.planetscope.conventions.ResolvedROISubset")
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
