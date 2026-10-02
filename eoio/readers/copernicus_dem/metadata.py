from __future__ import annotations
import xml.etree.ElementTree as ET

from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.footprint_utils import normalize_footprint

NS = {
    "gmd": "http://www.isotc211.org/2005/gmd",
    "gco": "http://www.isotc211.org/2005/gco",
    "gml": "http://www.opengis.net/gml",
    "tsxx": "http://www.terrasar.com/tsxx/",
}


class CopernicusDEMMetadataExtractor(BaseMetadataExtractor):
    def __init__(self, reader):
        super().__init__(reader)
        self.metadata_filepath = reader.layout.metadata_file()

    def _root(self):
        tree = ET.parse(self.metadata_filepath)
        return tree.getroot()

    def _text(self, root, xpath):
        element = root.find(xpath, NS)
        if element is not None and element.text:
            return element.text.strip()
        return None

    def _float_or_none(self, value):
        try:
            return float(value)
        except ValueError:
            return None

    def _resolution_m(self):

        variant = self._text(
            self._root(),
            ".//tsxx:productVariantInfo/tsxx:resolutionVariant",
        )

        return 30 if variant == "10" else 90

    def get_basic_metadata(self):

        root = self._root()

        west = self._float_or_none(
            self._text(
                root,
                ".//gmd:westBoundLongitude/gco:Decimal",
            )
        )
        east = self._float_or_none(
            self._text(
                root,
                ".//gmd:eastBoundLongitude/gco:Decimal",
            )
        )
        south = self._float_or_none(
            self._text(
                root,
                ".//gmd:southBoundLatitude/gco:Decimal",
            )
        )
        north = self._float_or_none(
            self._text(
                root,
                ".//gmd:northBoundLatitude/gco:Decimal",
            )
        )

        bbox = [west, south, east, north]
        resolution = self._resolution_m()

        return {
            "collection_name": "Copernicus DEM",
            "product_name": self._text(
                root,
                "./gmd:dataSetURI/gco:CharacterString",
            ),
            "processing_level": self._text(
                root,
                ".//tsxx:productVariantInfo/tsxx:productVariant",
            ),
            "platform": "TanDEM-X",
            "instrument": "SAR",
            "constellation": "TanDEM-X",
            "spatial_resolution": [resolution],
            "geometry_ids": [f"{resolution}m"],
            "product_geospatial_bounds": bbox,
            "geospatial_bounds_crs": "EPSG:4326",
            # Only date granularity is published for this product (a static elevation
            # composite, no acquisition time-of-day concept) -- source tag is gco:Date, not
            # gco:DateTime.
            "product_datetime": self._text(
                root,
                ".//gmd:CI_Citation/gmd:date//gco:Date",
            ),
            "description": self._text(
                root,
                ".//gmd:abstract/gco:CharacterString",
            ),
            "institution": "Airbus Defence and Space GmbH",
            "keywords": [
                kw.text
                for kw in root.findall(
                    ".//gmd:keyword/gco:CharacterString",
                    NS,
                )
            ],
            "source": "Copernicus Digital Elevation Model",
            "platform_type": "satellite",
            "footprint": normalize_footprint(
                geometry_input=bbox,
                crs_input=4326,
            ),
        }

    def get_product_metadata(self):

        root = self._root()

        md = self.get_basic_metadata()

        md.update(
            {
                "epsg": 4326,
                "horizontal_crs": self._text(
                    root,
                    "./gmd:referenceSystemInfo[1]"
                    "/gmd:MD_ReferenceSystem"
                    "/gmd:referenceSystemIdentifier"
                    "/gmd:RS_Identifier"
                    "/gmd:code"
                    "/gco:CharacterString",
                ),
                "vertical_crs": self._text(
                    root,
                    "./gmd:referenceSystemInfo[2]"
                    "/gmd:MD_ReferenceSystem"
                    "/gmd:referenceSystemIdentifier"
                    "/gmd:RS_Identifier"
                    "/gmd:code"
                    "/gco:CharacterString",
                ),
                "rows": int(
                    self._text(
                        root,
                        ".//tsxx:imageRaster/tsxx:numberOfRows",
                    )
                ),
                "columns": int(
                    self._text(
                        root,
                        ".//tsxx:imageRaster/tsxx:numberOfColumns",
                    )
                ),
                "format": self._text(
                    root,
                    ".//tsxx:imageDataFormat",
                ),
                "data_type": self._text(
                    root,
                    ".//tsxx:imageDataType",
                ),
                "mean_elevation_m": self._float_or_none(
                    self._text(
                        root,
                        ".//tsxx:productStatistics/tsxx:meanValue",
                    )
                ),
                "min_elevation_m": self._float_or_none(
                    self._text(
                        root,
                        ".//tsxx:productStatistics/tsxx:minValue",
                    )
                ),
                "max_elevation_m": self._float_or_none(
                    self._text(
                        root,
                        ".//tsxx:productStatistics/tsxx:maxValue",
                    )
                ),
                "stddev_elevation_m": self._float_or_none(
                    self._text(
                        root,
                        ".//tsxx:productStatistics/tsxx:stdDev",
                    )
                ),
                "valid_pixels": int(
                    self._text(
                        root,
                        ".//tsxx:nrValidPixels",
                    )
                ),
                "le68_m": self._float_or_none(
                    self._text(
                        root,
                        ".//tsxx:absolutePositionalAccuracyLE68",
                    )
                ),
                "le90_m": self._float_or_none(
                    self._text(
                        root,
                        ".//tsxx:absolutePositionalAccuracyLE90",
                    )
                ),
                "acquisition_start": self._text(
                    root,
                    ".//tsxx:tsxx_startTime",
                ),
                "acquisition_end": self._text(
                    root,
                    ".//tsxx:tsxx_stopTime",
                ),
                "mission": self._text(
                    root,
                    ".//tsxx:mission",
                ),
            }
        )

        return md

    def get_variable_basic_metadata(self, var):

        if var.lower() != "elevation":
            return {}

        return {
            "standard_name": "surface_altitude",
            "long_name": "Digital Surface Model elevation",
            "units": "m",
            "measurand": "elevation",
        }

    def get_variable_product_metadata(self, var):
        if var.lower() != "elevation":
            return {}
        root = self._root()

        resolution = self._resolution_m()

        return {
            "standard_name": "surface_altitude",
            "long_name": "Digital Surface Model elevation",
            "units": "m",
            "measurand": "elevation",
            "spatial_resolution": resolution,
            "spatial_resolution_units": "m",
            "geometry_id": "30m",
            "data_type": self._text(
                root,
                ".//tsxx:imageDataType",
            ),
            "data_format": self._text(
                root,
                ".//tsxx:imageDataFormat",
            ),
            "rows": int(
                self._text(
                    root,
                    ".//tsxx:numberOfRows",
                )
            ),
            "columns": int(
                self._text(
                    root,
                    ".//tsxx:numberOfColumns",
                )
            ),
            "ancillary_variables": [
                "edm",
                "flm",
                "wbm",
                "hem",
            ],
        }

    def get_aux_metadata(self):

        return {
            "edm": {
                "long_name": "Editing Mask",
                "description": "Indicates whether a pixel was edited",
                "units": "1",
            },
            "flm": {
                "long_name": "Filling Mask",
                "description": "Indicates whether a pixel was filled from ancillary data",
                "units": "1",
            },
            "wbm": {
                "long_name": "Water Body Mask",
                "description": "Water mask",
                "units": "1",
            },
            "hem": {
                "long_name": "Height Error Mask",
                "description": "Per-pixel elevation standard deviation",
                "units": "m",
            },
        }
