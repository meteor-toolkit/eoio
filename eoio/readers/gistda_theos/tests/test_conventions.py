"""eoio.readers.gistda_theos.tests.test_conventions - unit tests for eoio.readers.gistda_theos.conventions"""

import unittest
from unittest.mock import Mock

import xarray as xr

from eoio.readers.gistda_theos.conventions import apply_conventions


class TestGistdaTheosConventions(unittest.TestCase):
    """Unit tests for THEOS conventions method."""

    def test_apply_conventions_returns_dataset(self):
        mock_config = Mock()
        subset = Mock()
        subset.clip_box = "test_subset"

        result = apply_conventions(
            ds=xr.Dataset(),
            layout=Mock(),
            roi_subset=subset,
            config=mock_config,
        )

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("eoio:reader", result.attrs)
        self.assertEqual(result.attrs["eoio:reader"], "gistda_theos")
        self.assertEqual(result.attrs["eoio:subset"], "test_subset")


if __name__ == "__main__":
    unittest.main()
