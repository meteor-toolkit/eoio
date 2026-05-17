# """eoio.utils.tif_tools - TIFF utility functions"""

# import pyproj
# from tifffile import TiffFile

# __author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

# __all__ = ["file_header", "get_header_xy", "get_header_crs"]


# def file_header(path):
#     """
#     Get geotiff file header info from tiff file

#     :param path:
#     :return: geotiff metadata dictionary
#     """
#     with TiffFile(path) as tif:
#         geotiff_dict = {
#             k: (
#                 v
#                 if type(v) == int
#                 or type(v) == float
#                 or type(v) == str
#                 or type(v) == list
#                 else (v.name, v.value)
#             )
#             for (k, v) in tif.geotiff_metadata.items()
#         }
#         geotiff_dict.update({"TIFShape": tif.pages[0].shape})

#     return geotiff_dict


# def get_header_xy(header_dict):
#     """
#     Get image coordinate features from tif file

#     :param header_dict: geotiff metadata dictionary
#     :return: x_step, y_step, x_origin, y_origin, x_len, y_len
#     """
#     x_res, y_res, z_res = [int(v) for v in header_dict["ModelPixelScale"]]
#     x_origin, y_origin = [
#         int(v) for v in header_dict["ModelTiepoint"] if int(v) != 0
#     ]  # origin refers to the top left
#     x_step, y_step = x_res, -y_res  # y-values decrease from top to bottom
#     y_len, x_len = [int(v) for v in header_dict["TIFShape"]]

#     return x_step, y_step, x_origin, y_origin, x_len, y_len


# def get_header_crs(header_dict):
#     """
#     Get header coordinates reference system from tif file

#     :param header_dict: geotiff metadata dictionary
#     :return: file coordinate reference system
#     """
#     try:
#         # return pyproj.CRS.from_epsg(header_dict["ProjectedCSTypeGeoKey"][-1])
#         return {
#             "epsg": "EPSG:" + str(header_dict["ProjectedCSTypeGeoKey"][-1]),
#             "crs": pyproj.CRS.from_epsg(header_dict["ProjectedCSTypeGeoKey"][-1]),
#         }
#     except pyproj.exceptions.CRSError:
#         return None
#         # d = header_dict
#         # units_dict = {"degree": 0.0174532925199433, "meter": 1.0, "metre": 1.0}
#         # test_wkt = 'PROJCS["{}",GEOGCS["{}",DATUM["WGS_1984",SPHEROID["{}",{},{}]],PRIMEM["Greenwich",0],UNIT["{}",' \
#         #            '{}]], PROJECTION["{}"],PARAMETER["latitude_of_origin",{}],PARAMETER["central_meridian",{}],' \
#         #            'PARAMETER["false_easting",{}],PARAMETER["false_northing",{}],UNIT["{}",{}]]'.format(
#         #     d['GTCitationGeoKey'], d['GeogCitationGeoKey'], d['GeogCitationGeoKey'], d['GeogSemiMajorAxisGeoKey'],
#         #     d['GeogInvFlatteningGeoKey'], d['GeogAngularUnitsGeoKey'][0].lower(),
#         #     units_dict[d['GeogAngularUnitsGeoKey'][0].lower()], d['ProjCoordTransGeoKey'][0],
#         #     d['ProjNatOriginLatGeoKey'], d['ProjStraightVertPoleLongGeoKey'], d['ProjFalseEastingGeoKey'],
#         #     d['ProjFalseNorthingGeoKey'], d['ProjLinearUnitsGeoKey'][0].lower(),
#         #     units_dict[d['ProjLinearUnitsGeoKey'][0].lower()])
#         # return {"epsg": "EPSG:32767", "crs": pyproj.CRS(test_wkt)}
