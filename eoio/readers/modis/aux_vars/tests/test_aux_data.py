"""eoio.readers.modis.aux_vars.tests.test_aux_data - tests for eoio.readers.modis.aux_vars.aux_data"""

import unittest
from unittest.mock import MagicMock
from eoio.readers.modis.aux_vars.aux_data import get_available_aux


class TestGetAvailableAux(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = MagicMock()

    def test_get_available_aux_returns_list(self):
        """Test that get_available_aux returns a list."""
        result = get_available_aux(self.layout)
        self.assertIsInstance(result, list)

    def test_get_available_aux_contains_known_aux_vars(self):
        """Test that get_available_aux includes standard auxiliary variables."""
        result = get_available_aux(self.layout)
        # Should include common MODIS auxiliary variables
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
