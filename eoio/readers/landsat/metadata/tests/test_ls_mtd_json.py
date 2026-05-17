"""eoio.readers.landsat.tests.test_ls_mtd_json - tests for eoio.readers.landsat.metadata.ls_mtd_json"""

import unittest
import tempfile
import json
from pathlib import Path
from eoio.readers.landsat.metadata.ls_mtd_json import LSL1ProdJSONReader


class TestLSL1ProdJSONReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Sample STAC JSON
        self.stac_content = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "assets": {
                "B1": {"eo:bands": [{"name": "B1", "center_wavelength": 0.44, "gsd": 30}]},
                "B2": {"eo:bands": [{"name": "B2", "center_wavelength": 0.48, "gsd": 30}]},
            },
            "properties": {
                "proj:epsg": 32630,
                "proj:shape": [1000, 1000],
                "proj:transform": [30, 0, 0, 0, -30, 0],
            },
        }

        # Sample MTL JSON (wrapped)
        self.mtl_content = {"LANDSAT_METADATA_FILE": {"some": "data"}}

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_DetectsStac(self):
        p = self.tmp_path / "stac.json"
        p.write_text(json.dumps(self.stac_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        self.assertTrue(reader._is_stac)

    def test_init_DetectsMtlAndUnwraps(self):
        p = self.tmp_path / "mtl.json"
        p.write_text(json.dumps(self.mtl_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        self.assertFalse(reader._is_stac)
        self.assertEqual(reader.data.get("some"), "data")

    def test_find_epsg_ReturnsEpsg_FromStac(self):
        p = self.tmp_path / "stac.json"
        p.write_text(json.dumps(self.stac_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        self.assertEqual(reader.find_epsg(), 32630)

    def test_find_epsg_RaisesIfMissing_AndNoDefault(self):
        p = self.tmp_path / "stac_no_epsg.json"
        content = self.stac_content.copy()
        del content["properties"]["proj:epsg"]
        p.write_text(json.dumps(content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        with self.assertRaises(ValueError):
            reader.find_epsg()

    def test_find_epsg_ReturnsDefaultIfMissing(self):
        p = self.tmp_path / "stac_no_epsg.json"
        content = self.stac_content.copy()
        del content["properties"]["proj:epsg"]
        p.write_text(json.dumps(content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        self.assertEqual(reader.find_epsg(default=4326), 4326)

    def test_find_all_band_central_wavelengths_ReturnsDict(self):
        p = self.tmp_path / "stac.json"
        p.write_text(json.dumps(self.stac_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        wls = reader.find_all_band_central_wavelengths()
        # 0.44 um -> 440 nm
        self.assertAlmostEqual(wls["B1"], 440.0)
        self.assertAlmostEqual(wls["B2"], 480.0)

    def test_find_all_band_gsds_ReturnsDict(self):
        p = self.tmp_path / "stac.json"
        p.write_text(json.dumps(self.stac_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        gsds = reader.find_all_band_gsds()
        self.assertEqual(gsds["B1"], 30.0)

    def test_find_band_central_wavelength_ReturnsValue(self):
        p = self.tmp_path / "stac.json"
        p.write_text(json.dumps(self.stac_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        wl = reader.find_band_central_wavelength("B1")
        self.assertAlmostEqual(wl, 440.0)

    def test_find_band_central_wavelength_RaisesIfMissing(self):
        p = self.tmp_path / "stac.json"
        p.write_text(json.dumps(self.stac_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        with self.assertRaises(KeyError):
            reader.find_band_central_wavelength("B99")

    def test_find_band_central_wavelength_ReturnsDefault(self):
        p = self.tmp_path / "stac.json"
        p.write_text(json.dumps(self.stac_content), encoding="utf-8")
        reader = LSL1ProdJSONReader(p)
        self.assertEqual(reader.find_band_central_wavelength("B99", default=0), 0)


if __name__ == "__main__":
    unittest.main()
