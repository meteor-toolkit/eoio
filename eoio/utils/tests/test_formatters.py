"""eoio.utils.tests.test_formatters - test for eoio.utils.formatters"""

import datetime as dt
import unittest

from eoio.utils.formatters import is_datetime, is_number, list_to_dict, str2datetime, val_format

__author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

__all__ = []


class TestFormatters(unittest.TestCase):
    def test_is_number_int(self):
        self.assertTrue(is_number("9"))

    def test_is_number_float(self):
        self.assertTrue(is_number("4.3"))

    def test_is_number_negative_float(self):
        self.assertTrue(is_number("-0.5"))

    def test_is_number_letter(self):
        self.assertFalse(is_number("l"))

    def test_is_datetime_microseconds(self):
        self.assertTrue(is_datetime("2022-09-10T20:23:47.111356Z"))
        self.assertTrue(is_datetime("2022-09-10T20:23:47.111356"))

    def test_is_datetime_seconds(self):
        self.assertTrue(is_datetime("2022-09-10T20:23:47Z"))
        self.assertTrue(is_datetime("2022-09-10T20:23:47"))

    def test_is_datetime_int(self):
        self.assertFalse(is_datetime("23451232"))

    def test_str2datetime_microseconds(self):
        self.assertEqual(
            str2datetime("2022-09-10T20:23:47.111356Z"),
            dt.datetime(2022, 9, 10, 20, 23, 47, 111356),
        )
        self.assertEqual(
            str2datetime("2022-09-10T20:23:47.111356"),
            dt.datetime(2022, 9, 10, 20, 23, 47, 111356),
        )

    def test_str2datetime_seconds(self):
        self.assertEqual(str2datetime("2022-09-10T20:23:47Z"), dt.datetime(2022, 9, 10, 20, 23, 47))
        self.assertEqual(str2datetime("2022-09-10T20:23:47"), dt.datetime(2022, 9, 10, 20, 23, 47))

    def test_str2datetime_date(self):
        self.assertEqual(str2datetime("2022-09-10"), dt.datetime(2022, 9, 10, 0, 0))

    def test_str2datetime_time(self):
        self.assertEqual(str2datetime("20:23:47"), dt.time(20, 23, 47))
        self.assertEqual(str2datetime("20:23:47.111356Z"), dt.time(20, 23, 47, 111356))
        self.assertEqual(str2datetime("20:23:47.11135654Z"), dt.time(20, 23, 47, 111356))

    def test_str2datetime_str(self):
        self.assertEqual(str2datetime("time"), "time")

    def test_val_format_str(self):
        self.assertEqual(val_format("str"), "str")
        self.assertEqual(val_format("multiple separated strings"), "multiple separated strings")
        self.assertEqual(
            val_format("this=text;that=text;other=text"),
            {"this": "text", "that": "text", "other": "text"},
        )
        self.assertEqual(
            val_format("this=text;that=text;and the other=text"),
            {"this": "text", "that": "text", "and the other": "text"},
        )

    def test_val_format_int(self):
        self.assertEqual(val_format("2"), 2)
        self.assertEqual(val_format("1 2 3 4 5"), [1, 2, 3, 4, 5])

    def test_val_format_float(self):
        self.assertEqual(val_format("2.3"), 2.3)
        self.assertEqual(val_format("1.2 1.3 1.4 1.5"), [1.2, 1.3, 1.4, 1.5])
        self.assertEqual(val_format("0.8 1 1.2"), [0.8, 1.0, 1.2])

    def test_val_format_other(self):
        self.assertEqual(val_format({"A": "a"}), {"A": "a"})

    # todo - look into nested dictionaries
    def test_list_to_dict(self):
        input_list = [
            ("GROUP", "Satellite"),
            ("Name", "Sentinel 2A"),
            ("Agency", "European Space Agency"),
            ("Launch_Date", "23 June 2015"),
            ("END_GROUP", "Satellite"),
            ("GROUP", "Metadata"),
            ("Aquisition_time", "01:51:58"),
            ("Processing_time", "03:21:37"),
            ("END_GROUP", "Metadata"),
            ("GROUP", "BAND_1"),
            ("Name", "Spectral band 1"),
            ("END_GROUP", "BAND_1"),
        ]
        output_dict = {
            "Satellite": {
                "Name": "Sentinel 2A",
                "Agency": "European Space Agency",
                "Launch_Date": "23 June 2015",
            },
            "Metadata": {
                "Aquisition_time": "01:51:58",
                "Processing_time": "03:21:37",
            },
            "BAND_1": {
                "Name": "Spectral band 1",
            },
        }

        self.maxDiff = None
        self.assertDictEqual(list_to_dict(input_list), output_dict)

    def test_txt_to_dict(self):
        pass


if __name__ == "__main__":
    unittest.main()
