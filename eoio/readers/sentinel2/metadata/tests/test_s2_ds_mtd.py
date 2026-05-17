"""eoio.readers.sentinel2.metadata.tests.test_s2_ds_mtd- tests for eoio.readers.sentinel2.metadata.test_s2_ds_mtd"""

from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
from eoio.readers.sentinel2.metadata.s2_ds_mtd import S2DSXMLReader


class TestS2DSXMLReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Matches your real structure:
        # - n1 prefix on Quality_Indicators_Info only
        # - everything inside is unprefixed
        self.xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Root xmlns:n1="https://example.com/s2ds">
  <n1:Quality_Indicators_Info>
    <Radiometric_Info>
      <Radiometric_Quality_List>

        <Radiometric_Quality bandId="0">
          <Noise_Model>
            <ALPHA>0.241</ALPHA>
            <BETA>0.000164</BETA>
          </Noise_Model>
          <ABSOLUTE_CALIBRATION_ACCURACY>5</ABSOLUTE_CALIBRATION_ACCURACY>
          <CROSS_BAND_CALIBRATION_ACCURACY>3</CROSS_BAND_CALIBRATION_ACCURACY>
          <MULTI_TEMPORAL_CALIBRATION_ACCURACY>1</MULTI_TEMPORAL_CALIBRATION_ACCURACY>
        </Radiometric_Quality>

        <Radiometric_Quality bandId="1">
          <Noise_Model>
            <ALPHA>0.529</ALPHA>
            <BETA>0.0104</BETA>
          </Noise_Model>
          <ABSOLUTE_CALIBRATION_ACCURACY>5</ABSOLUTE_CALIBRATION_ACCURACY>
          <CROSS_BAND_CALIBRATION_ACCURACY>3</CROSS_BAND_CALIBRATION_ACCURACY>
          <MULTI_TEMPORAL_CALIBRATION_ACCURACY>1</MULTI_TEMPORAL_CALIBRATION_ACCURACY>
        </Radiometric_Quality>

        <Radiometric_Quality bandId="12">
          <Noise_Model>
            <ALPHA>0.729</ALPHA>
            <BETA>0.00237</BETA>
          </Noise_Model>
          <ABSOLUTE_CALIBRATION_ACCURACY>5</ABSOLUTE_CALIBRATION_ACCURACY>
          <CROSS_BAND_CALIBRATION_ACCURACY>3</CROSS_BAND_CALIBRATION_ACCURACY>
          <MULTI_TEMPORAL_CALIBRATION_ACCURACY>1</MULTI_TEMPORAL_CALIBRATION_ACCURACY>
        </Radiometric_Quality>

      </Radiometric_Quality_List>
    </Radiometric_Info>

    <DEGRADED_ANC_DATA_PERCENTAGE>0</DEGRADED_ANC_DATA_PERCENTAGE>
  </n1:Quality_Indicators_Info>
</n1:Root>
"""
        self.xml_file = self.tmp_path / "MTD_DS.xml"
        self.xml_file.write_text(self.xml_text, encoding="utf-8")
        self.reader = S2DSXMLReader(self.xml_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_find_noise_model_alpha(self):
        out = self.reader.find_noise_model_alpha()
        self.assertEqual(out, {0: 0.241, 1: 0.529, 12: 0.729})
        self.assertIsInstance(out[0], float)

    def test_find_noise_model_beta(self):
        out = self.reader.find_noise_model_beta()
        self.assertEqual(out, {0: 0.000164, 1: 0.0104, 12: 0.00237})

    def test_find_absolute_calibration_accuracy(self):
        out = self.reader.find_absolute_calibration_accuracy()
        self.assertEqual(out, {0: 5.0, 1: 5.0, 12: 5.0})

    def test_find_cross_band_calibration_accuracy(self):
        out = self.reader.find_cross_band_calibration_accuracy()
        self.assertEqual(out, {0: 3.0, 1: 3.0, 12: 3.0})

    def test_find_multi_temporal_calibration_accuracy(self):
        out = self.reader.find_multi_temporal_calibration_accuracy()
        self.assertEqual(out, {0: 1.0, 1: 1.0, 12: 1.0})

    def test_returns_None_when_section_missing(self):
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Root xmlns:n1="https://example.com/s2ds">
  <n1:Quality_Indicators_Info>
    <Radiometric_Info/>
  </n1:Quality_Indicators_Info>
</n1:Root>
"""
        xml_file = self.tmp_path / "empty.xml"
        xml_file.write_text(xml_text, encoding="utf-8")
        r = S2DSXMLReader(xml_file)

        self.assertIsNone(r.find_noise_model_alpha())
        self.assertIsNone(r.find_noise_model_beta())
        self.assertIsNone(r.find_absolute_calibration_accuracy())
        self.assertIsNone(r.find_cross_band_calibration_accuracy())
        self.assertIsNone(r.find_multi_temporal_calibration_accuracy())

    def test_raises_if_child_missing_for_value_xpath(self):
        # Missing <ALPHA> inside Noise_Model for band 0
        bad = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Root xmlns:n1="https://example.com/s2ds">
  <n1:Quality_Indicators_Info>
    <Radiometric_Info>
      <Radiometric_Quality_List>
        <Radiometric_Quality bandId="0">
          <Noise_Model>
            <BETA>0.1</BETA>
          </Noise_Model>
        </Radiometric_Quality>
      </Radiometric_Quality_List>
    </Radiometric_Info>
  </n1:Quality_Indicators_Info>
</n1:Root>
"""
        xml_file = self.tmp_path / "bad.xml"
        xml_file.write_text(bad, encoding="utf-8")
        r = S2DSXMLReader(xml_file)

        with self.assertRaises(ValueError):
            r.find_noise_model_alpha()


if __name__ == "__main__":
    unittest.main()
