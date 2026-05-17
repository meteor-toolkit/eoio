# """eoio.utils.test.test_tif_tools - test for eoio.utils.tif_tools"""

# import datetime as dt
# import unittest
# from unittest import mock

# import numpy as np
# import pyproj

# from eoio.utils.tif_tools import file_header, get_header_crs, get_header_xy

# __author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

# __all__ = []


# class MockObject:
#     name = "obj_name"
#     value = "obj_value"


# class MockTiff:
#     geotiff_metadata = {
#         "name": "filename",
#         "age": 9,
#         "bands": ["B1", "B2", "B3", "B4"],
#         "object": MockObject,
#     }
#     pages = np.array(
#         [
#             [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
#             [[2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]],
#         ]
#     )

#     def __init__(self, path):
#         self.path = path

#     def __enter__(self):
#         return self

#     def __exit__(self, a, b, c):
#         pass


# class MyTestCase(unittest.TestCase):
#     @mock.patch("eoio.utils.tif_tools.TiffFile", return_value=MockTiff("path"))
#     def test_file_header(self, mockTiffFile):
#         output_dict = {
#             "name": "filename",
#             "age": 9,
#             "bands": ["B1", "B2", "B3", "B4"],
#             "object": ("obj_name", "obj_value"),
#             "TIFShape": (3, 4),
#         }
#         self.assertEqual(file_header("path"), output_dict)

#     def test_get_header_xy(self):
#         header_dict = {
#             "KeyDirectoryVersion": 1,
#             "KeyRevision": 1,
#             "KeyRevisionMinor": 0,
#             "GTModelTypeGeoKey": ("Projected", 1),
#             "GTRasterTypeGeoKey": ("IsPoint", 2),
#             "GTCitationGeoKey": "UTM Zone 34 N with WGS84",
#             "GeogLinearUnitsGeoKey": ("Meter", 9001),
#             "GeogAngularUnitsGeoKey": ("Degree", 9102),
#             "ProjectedCSTypeGeoKey": ("WGS84_UTM_zone_34N", 32634),
#             "ModelPixelScale": [30.0, 30.0, 0.0],
#             "ModelTiepoint": [0.0, 0.0, 0.0, 664200.0, 3311400.0, 0.0],
#             "TIFShape": (7681, 7531),
#         }
#         self.assertEqual(
#             get_header_xy(header_dict), (30, -30, 664200, 3311400, 7531, 7681)
#         )

#     def test_get_header_crs(self):
#         header_dict = {
#             "KeyDirectoryVersion": 1,
#             "KeyRevision": 1,
#             "KeyRevisionMinor": 0,
#             "GTModelTypeGeoKey": ("Projected", 1),
#             "GTRasterTypeGeoKey": ("IsPoint", 2),
#             "GTCitationGeoKey": "UTM Zone 34 N with WGS84",
#             "GeogLinearUnitsGeoKey": ("Meter", 9001),
#             "GeogAngularUnitsGeoKey": ("Degree", 9102),
#             "ProjectedCSTypeGeoKey": ("WGS84_UTM_zone_34N", 32634),
#             "ModelPixelScale": [30.0, 30.0, 0.0],
#             "ModelTiepoint": [0.0, 0.0, 0.0, 664200.0, 3311400.0, 0.0],
#             "TIFShape": (7681, 7531),
#         }
#         self.assertDictEqual(
#             get_header_crs(header_dict),
#             {"epsg": "EPSG:32634", "crs": pyproj.CRS.from_epsg(32634)},
#         )

#     # edited to get an undefined crs from reading in Band 1
#     def test_get_header_crs_user_defined(self):
#         header_dict = {
#             "KeyDirectoryVersion": 1,
#             "KeyRevision": 1,
#             "KeyRevisionMinor": 0,
#             "GTModelTypeGeoKey": ("Projected", 1),
#             "GTRasterTypeGeoKey": ("IsPoint", 2),
#             "GTCitationGeoKey": "PS         WGS84",
#             "GeographicTypeGeoKey": ("WGS_84", 4326),
#             "GeogCitationGeoKey": "WGS 84",
#             "GeogAngularUnitsGeoKey": ("Degree", 9102),
#             "GeogSemiMajorAxisGeoKey": 6378137.0,
#             "GeogInvFlatteningGeoKey": 298.257223563,
#             "ProjectedCSTypeGeoKey": ("User_Defined", 32767),
#             "ProjectionGeoKey": ("User_Defined", 32767),
#             "ProjCoordTransGeoKey": ("PolarStereographic", 15),
#             "ProjLinearUnitsGeoKey": ("Meter", 9001),
#             "ProjNatOriginLatGeoKey": -71.0,
#             "ProjFalseEastingGeoKey": 0.0,
#             "ProjFalseNorthingGeoKey": 0.0,
#             "ProjScaleAtNatOriginGeoKey": 1.0,
#             "ProjStraightVertPoleLongGeoKey": 0.0,
#             "ModelPixelScale": [30.0, 30.0, 0.0],
#             "ModelTiepoint": [0.0, 0.0, 0.0, 642000.0, 2079300.0, 0.0],
#             "TIFShape": (9211, 9201),
#         }

#         output_wkt = (
#             'PROJCS["PS         WGS84",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
#             '298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
#             'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],'
#             'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-71],PARAMETER['
#             '"central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",'
#             '1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]] '
#         )

#         self.assertEqual(
#             get_header_crs(header_dict), None
#         )  # {"epsg": "EPSG:32767", "crs": pyproj.CRS(output_wkt)})


# if __name__ == "__main__":
#     unittest.main()
