import unittest
from unittest.mock import Mock, patch

from eoio.readers.copernicus_dem.metadata import (
    CopernicusDEMMetadataExtractor,
)


class TestCopernicusDEMMetadataExtractor(unittest.TestCase):
    def setUp(self):
        reader = Mock()
        reader.layout.metadata_file.return_value = "dummy.xml"

        self.extractor = CopernicusDEMMetadataExtractor(reader)

    @patch.object(CopernicusDEMMetadataExtractor, "_root")
    @patch.object(CopernicusDEMMetadataExtractor, "_text")
    def test_resolution_m_30m(self, mock_text, mock_root):
        mock_root.return_value = Mock()
        mock_text.return_value = "10"

        self.assertEqual(
            self.extractor._resolution_m(),
            30,
        )

    @patch.object(CopernicusDEMMetadataExtractor, "_root")
    @patch.object(CopernicusDEMMetadataExtractor, "_text")
    def test_resolution_m_90m(self, mock_text, mock_root):
        mock_root.return_value = Mock()
        mock_text.return_value = "30"

        self.assertEqual(
            self.extractor._resolution_m(),
            90,
        )

    @patch.object(CopernicusDEMMetadataExtractor, "_root")
    def test_get_variable_product_metadata_unknown(self, mock_root):
        mock_root.return_value = Mock()

        self.assertEqual(
            self.extractor.get_variable_product_metadata("unknown"),
            {},
        )

    @patch.object(CopernicusDEMMetadataExtractor, "_root")
    @patch.object(CopernicusDEMMetadataExtractor, "_resolution_m")
    @patch.object(CopernicusDEMMetadataExtractor, "_text")
    def test_get_variable_product_metadata(
        self,
        mock_text,
        mock_resolution,
        mock_root,
    ):
        mock_root.return_value = Mock()
        mock_resolution.return_value = 30

        values = {
            ".//tsxx:imageDataType": "INT16",
            ".//tsxx:imageDataFormat": "GeoTIFF",
            ".//tsxx:numberOfRows": "3600",
            ".//tsxx:numberOfColumns": "3600",
        }

        mock_text.side_effect = lambda root, xpath: values[xpath]

        md = self.extractor.get_variable_product_metadata("elevation")

        self.assertEqual(md["spatial_resolution"], 30)
        self.assertEqual(md["data_type"], "INT16")
        self.assertEqual(md["data_format"], "GeoTIFF")
        self.assertEqual(md["rows"], 3600)
        self.assertEqual(md["columns"], 3600)


    @patch("eoio.readers.copernicus_dem.metadata.normalize_footprint")
    @patch.object(CopernicusDEMMetadataExtractor, "_resolution_m")
    @patch.object(CopernicusDEMMetadataExtractor, "_root")
    @patch.object(CopernicusDEMMetadataExtractor, "_text")
    def test_get_basic_metadata(
        self,
        mock_text,
        mock_root,
        mock_resolution,
        mock_normalize_footprint,
    ):
        root = Mock()
        root.findall.return_value = [
            Mock(text="DEM"),
            Mock(text="Elevation"),
        ]

        mock_root.return_value = root
        mock_resolution.return_value = 30
        mock_normalize_footprint.return_value = "footprint"

        values = {
            ".//gmd:westBoundLongitude/gco:Decimal": "-10",
            ".//gmd:eastBoundLongitude/gco:Decimal": "10",
            ".//gmd:southBoundLatitude/gco:Decimal": "45",
            ".//gmd:northBoundLatitude/gco:Decimal": "55",
            "./gmd:dataSetURI/gco:CharacterString": "product",
            ".//tsxx:productVariantInfo/tsxx:productVariant": "DSM",
            ".//gmd:CI_Citation/gmd:date//gco:Date": "2024-01-01",
            ".//gmd:abstract/gco:CharacterString": "description",
        }

        mock_text.side_effect = lambda root, xpath: values[xpath]

        md = self.extractor.get_basic_metadata()

        self.assertEqual(
            md["collection_name"],
            "Copernicus DEM",
        )
        self.assertEqual(md["product_name"], "product")
        self.assertEqual(
            md["product_geospatial_bounds"],
            [-10.0, 45.0, 10.0, 55.0],
        )
        self.assertEqual(
            md["spatial_resolution"],
            [30],
        )
        self.assertEqual(
            md["keywords"],
            ["DEM", "Elevation"],
        )
        self.assertEqual(
            md["footprint"],
            "footprint",
        )

    @patch.object(
        CopernicusDEMMetadataExtractor,
        "get_basic_metadata",
    )
    @patch.object(CopernicusDEMMetadataExtractor, "_root")
    @patch.object(CopernicusDEMMetadataExtractor, "_text")
    def test_get_product_metadata(
        self,
        mock_text,
        mock_root,
        mock_basic_metadata,
    ):
        mock_root.return_value = Mock()

        mock_basic_metadata.return_value = {
            "collection_name": "Copernicus DEM",
        }

        values = {
            "./gmd:referenceSystemInfo[1]/gmd:MD_ReferenceSystem/gmd:referenceSystemIdentifier/gmd:RS_Identifier/gmd:code/gco:CharacterString": "EPSG:4326",
            "./gmd:referenceSystemInfo[2]/gmd:MD_ReferenceSystem/gmd:referenceSystemIdentifier/gmd:RS_Identifier/gmd:code/gco:CharacterString": "EGM2008",
            ".//tsxx:imageRaster/tsxx:numberOfRows": "3600",
            ".//tsxx:imageRaster/tsxx:numberOfColumns": "3600",
            ".//tsxx:imageDataFormat": "GeoTIFF",
            ".//tsxx:imageDataType": "INT16",
            ".//tsxx:productStatistics/tsxx:meanValue": "100.0",
            ".//tsxx:productStatistics/tsxx:minValue": "0.0",
            ".//tsxx:productStatistics/tsxx:maxValue": "1000.0",
            ".//tsxx:productStatistics/tsxx:stdDev": "25.0",
            ".//tsxx:nrValidPixels": "100000",
            ".//tsxx:absolutePositionalAccuracyLE68": "1.0",
            ".//tsxx:absolutePositionalAccuracyLE90": "2.0",
            ".//tsxx:tsxx_startTime": "2024-01-01T00:00:00",
            ".//tsxx:tsxx_stopTime": "2024-01-01T00:10:00",
            ".//tsxx:mission": "TanDEM-X",
        }

        mock_text.side_effect = lambda root, xpath: values[xpath]

        md = self.extractor.get_product_metadata()

        self.assertEqual(md["epsg"], 4326)
        self.assertEqual(md["rows"], 3600)
        self.assertEqual(md["columns"], 3600)
        self.assertEqual(md["format"], "GeoTIFF")
        self.assertEqual(md["data_type"], "INT16")
        self.assertEqual(md["mean_elevation_m"], 100.0)
        self.assertEqual(md["valid_pixels"], 100000)
        self.assertEqual(md["mission"], "TanDEM-X")

    def test_get_aux_metadata(self):
        md = self.extractor.get_aux_metadata()

        self.assertIn("edm", md)
        self.assertIn("flm", md)
        self.assertIn("wbm", md)
        self.assertIn("hem", md)

        self.assertEqual(
            md["hem"]["units"],
            "m",
        )


if __name__ == "__main__":
    unittest.main()
