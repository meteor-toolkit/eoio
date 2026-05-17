"""eoio.readers.landsat.tests.test_layout - unit tests for eoio.readers.landsat.layout"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from eoio.readers.landsat.layout import LandsatLayout, LandsatLayoutError


class TestLandsatLayout(unittest.TestCase):
    def _make_landsat(
        self,
        root: Path,
        *,
        product_name: str = "LC08_L1TP_TEST",
        with_mtl_xml: bool = True,
        with_stac_json: bool = True,
        band_tokens: list[str] = None,
        with_angles: bool = False,
    ) -> Path:
        if band_tokens is None:
            band_tokens = ["1", "2"]  # B1, B2

        product = root / product_name
        product.mkdir(parents=True, exist_ok=False)

        if with_mtl_xml:
            (product / f"{product_name}_MTL.xml").write_text("<xml/>", encoding="utf-8")

        if with_stac_json:
            (product / f"{product_name}_STAC.json").write_text("{}", encoding="utf-8")

        for token in band_tokens:
            (product / f"{product_name}_B{token}.TIF").write_text("tif", encoding="utf-8")

        if with_angles:
            for suffix in ["VZA", "SZA", "VAA", "SAA"]:
                (product / f"{product_name}_{suffix}.tif").write_text("tif", encoding="utf-8")

        return product

    def test_init_RaisesIfPathMissing(self):
        with TemporaryDirectory() as td:
            missing = Path(td) / "missing_dir"
            with self.assertRaises(LandsatLayoutError):
                LandsatLayout(str(missing))

    def test_init_RaisesIfNotDirectory(self):
        with TemporaryDirectory() as td:
            f = Path(td) / "file.txt"
            f.write_text("x", encoding="utf-8")
            with self.assertRaises(LandsatLayoutError):
                LandsatLayout(str(f))

    def test_product_dir_ReturnsPath(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), product_name="L8_TEST")
            layout = LandsatLayout(str(p))
            self.assertEqual(layout.product_dir.name, "L8_TEST")

    def test_product_metadata_xml_ReturnsXMLPath(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), with_mtl_xml=True)
            layout = LandsatLayout(str(p))
            xml = layout.product_metadata_xml()
            self.assertTrue(xml.name.endswith("MTL.xml"))

    def test_product_metadata_xml_RaisesIfMaxMissing(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), with_mtl_xml=False)
            layout = LandsatLayout(str(p))
            with self.assertRaises(LandsatLayoutError):
                layout.product_metadata_xml()

    def test_product_metadata_json_ReturnsJSONPath(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), with_stac_json=True)
            layout = LandsatLayout(str(p))
            json = layout.product_metadata_json()
            self.assertTrue(json.name.endswith("STAC.json"))

    def test_product_metadata_json_RaisesIfJsonMissing(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), with_stac_json=False)
            layout = LandsatLayout(str(p))
            with self.assertRaises(LandsatLayoutError):
                layout.product_metadata_json()

    def test_available_band_tokens_ReturnsTokens(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), band_tokens=["1", "2"])
            layout = LandsatLayout(str(p))
            tokens = layout.available_band_tokens()
            self.assertEqual(tokens, {"1", "2"})

    def test_tif_band_files_ReturnsFiles(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), band_tokens=["1", "2"])
            layout = LandsatLayout(str(p))
            files = layout.tif_band_files(meas_vars=["B1"])
            self.assertIn("B1", files)
            self.assertTrue(files["B1"].endswith(".TIF"))

    def test_tif_band_files_RaisesIfNoneFound(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), band_tokens=["2"])
            layout = LandsatLayout(str(p))
            with self.assertRaises(LandsatLayoutError):
                layout.tif_band_files(meas_vars=["B1"])

    def test_get_angle_files_ReturnsAngleFiles(self):
        with TemporaryDirectory() as td:
            p = self._make_landsat(Path(td), with_angles=True)
            layout = LandsatLayout(str(p))
            angles = layout.get_angle_files()
            self.assertEqual(len(angles), 4)
            self.assertIn("viewing_zenith_angle", angles)
            self.assertTrue(angles["viewing_zenith_angle"].endswith("VZA.tif"))


if __name__ == "__main__":
    unittest.main()
