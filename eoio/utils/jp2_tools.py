# """eoio.utils.jp2_tools - JPEG 2000 utility functions"""

# import os
# import re
# from copy import deepcopy
# from typing import Tuple

# import numpy as np
# import pyproj
# import xmltodict
# from glymur import Jp2k

# try:
#     import xml.etree.cElementTree as ET

# except ImportError:
#     import xml.etree.ElementTree as ET  # type: ignore

# from eoio.utils.dict_tools import get_value

# __author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

# __all__ = ["file_header", "get_header_xy", "get_header_crs"]


# def file_header(path) -> dict:
#     """
#     Get file header grid dictionary from jp2 file

#     :param path: filepath to jp2 image file
#     :return: header dictionary
#     """
#     try:
#         jp2 = Jp2k(path)
#         xml_box: list = []
#         j = [jp2]
#         while xml_box == []:
#             xml_box = [box for box in j[0].box if box.box_id == "xml "]
#             if xml_box == []:
#                 j = [box for box in j[0].box if box.box_id == "asoc"]
#         xmlstring = ET.tostring(xml_box[0].xml.getroot())
#         xml_dict = xmltodict.parse(
#             xmlstring, dict_constructor=dict, namespaces={"ns0": None}
#         )
#         return get_value(xml_dict, "RectifiedGrid")

#     except (
#         IndexError
#     ):  # empty header dictionary - make own mock one from sentinel 2 metadata tile file (MTD_TL.xml)
#         raise IOError(
#             """Corrupt file detected: '{}'
#         Check product download.""".format(
#                 path
#             )
#         )


# def get_header_xy(header_dict) -> Tuple[int, int, int, int, int, int]:
#     """
#     Get header coordinate features from jp2 file

#     :param header_dict: header dictionary for jp2 image file
#     :return: x_step, y_step, x_origin, y_origin, x_len, y_len
#     """
#     x_step, y_step = [
#         [int(k) for k in j if int(k) != 0][0]
#         for j in [i[1].split() for i in get_value(header_dict, "#text", True)]
#     ]
#     x_origin, y_origin = [int(i) for i in get_value(header_dict, "pos").split()]
#     x_len, y_len = [int(i) for i in get_value(header_dict, "high").split()]

#     return x_step, y_step, x_origin, y_origin, x_len, y_len


# def get_header_crs(header_dict) -> dict:
#     """
#     Get header coordinate reference system from jp2 file

#     :param header_dict: header dictionary for jp2 image file
#     :return: file coordinate reference system
#     """

#     return {
#         "epsg": "EPSG:" + str(pyproj.CRS(get_value(header_dict, "@srsName")).to_epsg()),
#         "crs": pyproj.CRS(get_value(header_dict, "@srsName")),
#     }
