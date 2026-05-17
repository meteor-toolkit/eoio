"""Tests for eoio.readers.sentinel3_slstr.metadata.extractor.S3SLSTRMetadataExtractor."""

from __future__ import annotations
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
from pathlib import Path
import tempfile


from eoio.readers.sentinel3_slstr.metadata.extractor import S3SLSTRMetadataExtractor


class TestS3SLSTRMetadataExtractor(unittest.TestCase):
    def _make_reader_with_layout(self, manifest_xml_path: str):
        layout = SimpleNamespace(manifest_path=MagicMock(return_value=manifest_xml_path))
        resolved_config = SimpleNamespace(subset=None)
        # tests create a lightweight fake reader; BaseMetadataExtractor expects
        # a `path` attribute on the reader, so provide it here.
        reader = SimpleNamespace(layout=layout, resolved_config=resolved_config, path=manifest_xml_path)
        return reader

    def setUp(self):
        # create a minimal manifest used by the XML reader tests
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu" xmlns:safe="http://www.esa.int/safe/sentinel/1.1" xmlns:sentinel3="http://www.esa.int/safe/sentinel/sentinel-3/1.0" xmlns:slstr="http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0" xmlns:gml="http://www.opengis.net/gml">\n'
            "  <safe:acquisitionPeriod>\n    <safe:startTime>2024-01-02T03:04:05.000000Z</safe:startTime>\n  </safe:acquisitionPeriod>\n"
            '  <safe:frameSet>\n    <safe:footPrint srsName="http://www.opengis.net/def/crs/EPSG/0/4326">\n      <gml:posList>60.0 10.0 60.0 20.0 50.0 20.0 50.0 10.0 60.0 10.0</gml:posList>\n    </safe:footPrint>\n  </safe:frameSet>\n'
            "  <sentinel3:generalProductInformation>\n    <sentinel3:productName>TEST_PRODUCT</sentinel3:productName>\n    <sentinel3:productType>SL_1_RBT____</sentinel3:productType>\n    <sentinel3:processingBaseline>03.10</sentinel3:processingBaseline>\n  </sentinel3:generalProductInformation>\n"
            '  <slstr:slstrProductInformation>\n    <slstr:bandDescriptions>\n      <sentinel3:band name="S1">\n        <sentinel3:centralWavelength>400.5</sentinel3:centralWavelength>\n        <sentinel3:bandwidth>10.0</sentinel3:bandwidth>\n      </sentinel3:band>\n    </slstr:bandDescriptions>\n  </slstr:slstrProductInformation>\n'
            "</xfdu:XFDU>"
        )

        self.xml_file = self.tmp_path / "xfdumanifest.xml"
        self.xml_file.write_text(xml, encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_and_caching(self):
        reader = self._make_reader_with_layout(str(self.xml_file))
        extractor = S3SLSTRMetadataExtractor(reader)
        # xml_reader should exist and caches populated
        self.assertIsNotNone(extractor.xml_reader)
        self.assertIsInstance(extractor._central_wavelengths, dict)

    def test_get_basic_metadata_and_product_metadata(self):
        reader = self._make_reader_with_layout(str(self.xml_file))
        # make layout.get_grid/get_grid_res used later
        reader.layout.get_grid = MagicMock(return_value="an")
        reader.layout.get_grid_res = MagicMock(return_value=500)
        extractor = S3SLSTRMetadataExtractor(reader)

        basic = extractor.get_basic_metadata()
        self.assertIsInstance(basic, dict)
        self.assertIn("product_name", basic)
        self.assertIn("instrument", basic)
        self.assertEqual(basic.get("instrument"), "SLSTR")
        self.assertIn("footprint", basic)
        self.assertIsInstance(basic["footprint"], dict)
        self.assertEqual(basic["footprint"]["crs"], "EPSG:4326")

        pm = extractor.get_product_metadata()
        self.assertIsInstance(pm, dict)
        self.assertIn("acquisition_start_time", pm)
        self.assertIn("crs_code", pm)

    def test_variable_metadata_helpers(self):
        reader = self._make_reader_with_layout(str(self.xml_file))
        reader.layout.get_grid = MagicMock(return_value="an")
        reader.layout.get_grid_res = MagicMock(return_value=500)
        extractor = S3SLSTRMetadataExtractor(reader)

        vb = extractor.get_variable_basic_metadata("S1_radiance_an")
        self.assertIn("units", vb)
        self.assertEqual(vb.get("units"), "W/m2/sr/nm")

        vp = extractor.get_variable_product_metadata("S1_radiance_an")
        self.assertIn("band_name", vp)
        self.assertIn("spatial_resolution", vp)
        self.assertEqual(vp.get("spatial_resolution"), 500)


if __name__ == "__main__":
    unittest.main()
