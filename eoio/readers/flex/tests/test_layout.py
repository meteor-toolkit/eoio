"""Tests for eoio.readers.flex.layout"""

import datetime as dt
import unittest

from eoio.readers.flex.layout import FlexLayout

NAME = "flx_l2__flxsyn_20190930T102242_20190930T102254_20260312T190507_31TDG_100_036_01.nc"


class TestFlexLayout(unittest.TestCase):
    def test_parses_name(self):
        layout = FlexLayout(f"/some/dir/{NAME}")
        self.assertEqual(layout.level, "L2")
        self.assertEqual(layout.product_type, "FLXSYN")
        self.assertEqual(layout.processing_level, 2)
        self.assertEqual(layout.start_time, dt.datetime(2019, 9, 30, 10, 22, 42, tzinfo=dt.timezone.utc))
        self.assertEqual(layout.stop_time, dt.datetime(2019, 9, 30, 10, 22, 54, tzinfo=dt.timezone.utc))
        self.assertEqual(layout.tile, "31TDG")
        self.assertEqual(layout.numbers, (100, 36, 1))
        self.assertEqual(layout.name, NAME)

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            FlexLayout("not_a_flex_product.nc")

    def test_unsupported_product_raises(self):
        with self.assertRaises(ValueError):
            FlexLayout(NAME.replace("l2__flxsyn", "l1b_flxrad"))


if __name__ == "__main__":
    unittest.main()
