# """eoio.utils.tests.test_jp2_tools - test for eoio.utils.jp2_tools"""

# import unittest
# import unittest.mock as mock

# import numpy as np
# import pyproj

# from eoio.utils.jp2_tools import file_header, get_header_crs, get_header_xy

# __author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

# __all__ = []


# class MockJp2k:
#     def __init__(self, path):
#         self.box = [MockBox()]


# class MockBox:
#     def __init__(self):
#         self.box = [MockBox2()]
#         self.box_id = "asoc"


# class MockBox2:
#     def __init__(self):
#         self.box = [MockBox3()]
#         self.box_id = "asoc"


# class MockBox3:
#     def __init__(self):
#         self.box_id = "xml "
#         self.xml = MockXML()


# class MockXML:
#     def __init__(self):
#         pass

#     def getroot(self):
#         return None


# class MyJP2Test(unittest.TestCase):
#     @mock.patch("eoio.utils.jp2_tools.get_value", return_value=True)
#     @mock.patch("eoio.utils.jp2_tools.xmltodict.parse", return_value={})
#     @mock.patch("eoio.utils.jp2_tools.ET.tostring", return_value="xmlstring")
#     @mock.patch("eoio.utils.jp2_tools.Jp2k", return_value=MockJp2k("path"))
#     def test_file_header(self, Jp2k_mock, tostring_mock, parse_mock, get_value_mock):
#         self.assertTrue(file_header("path"))

#         Jp2k_mock.assert_called_once_with("path")
#         tostring_mock.assert_called_once_with(None)
#         parse_mock.assert_called_once_with(
#             "xmlstring", dict_constructor=dict, namespaces={"ns0": None}
#         )
#         get_value_mock.assert_called_once_with({}, "RectifiedGrid")

#     @mock.patch(
#         "eoio.utils.jp2_tools.get_value",
#         side_effect=[[("#text", "10 0"), ("#text", "0 -10")], "0 9", "10 20"],
#     )
#     def test_get_header_xy(self, get_value_mock):
#         self.assertEqual(get_header_xy({}), (10, -10, 0, 9, 10, 20))
#         self.assertEqual(get_value_mock.call_count, 3)

#     # @mock.patch("eoio.utils.jp2_tools.pyproj.CRS", return_value="crs")
#     # @mock.patch("eoio.utils.jp2_tools.get_value", return_value="crs_value")
#     def test_get_header_crs(self):  # , get_value_mock, from_string_mock):
#         input_header = {
#             "@dimension": "2",
#             "limits": {"GridEnvelope": {"low": "1 1", "high": "10980 10980"}},
#             "axisName": ["x", "y"],
#             "origin": {
#                 "Point": {
#                     "@id": "P0001",
#                     "@srsName": "urn:ogc:def:crs:EPSG::32633",
#                     "pos": "600005 5500015",
#                 }
#             },
#             "offsetVector": [
#                 {"@srsName": "urn:ogc:def:crs:EPSG::32633", "#text": "10 0"},
#                 {"@srsName": "urn:ogc:def:crs:EPSG::32633", "#text": "0 -10"},
#             ],
#         }
#         self.assertDictEqual(
#             get_header_crs(input_header),
#             {"epsg": "EPSG:32633", "crs": pyproj.CRS("EPSG:32633")},
#         )
#         # get_value_mock.assert_called_once_with({}, "@srsName")
#         # from_string_mock.assert_called_once_with("crs_value")


# if __name__ == "__main__":
#     unittest.main()
