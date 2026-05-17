"""Tests for Sentinel-2 L2A IMG_DATA token discovery in S2Layout."""

import unittest
import tempfile
from pathlib import Path

from eoio.readers.sentinel2.layout import S2Layout


class TestS2LayoutL2ATokens(unittest.TestCase):
    def test_available_img_tokens_includes_l2a_layers(self):
        with tempfile.TemporaryDirectory() as td:
            safe = Path(td) / "S2A_MSIL2A_20200101T000000_N0209_R000_T00XXX_20200101T000000.SAFE"
            img = safe / "GRANULE" / "L2A_T00XXX_A000000_20200101T000000" / "IMG_DATA" / "R10m"
            img.mkdir(parents=True)

            # Create fake JP2 files (empty files are fine for layout tests)
            (img / "T00XXX_20200101T000000_B02_10m.jp2").touch()
            (img / "T00XXX_20200101T000000_AOT_10m.jp2").touch()
            (img / "T00XXX_20200101T000000_WVP_10m.jp2").touch()
            (img / "T00XXX_20200101T000000_SCL_20m.jp2").touch()

            # Minimal metadata file so product_metadata_xml() succeeds
            (safe / "MTD_MSIL2A.xml").write_text("<xml></xml>")

            layout = S2Layout(str(safe))
            toks = layout.available_img_tokens()

            self.assertIn("B02", toks)
            self.assertIn("AOT", toks)
            self.assertIn("WVP", toks)
            self.assertIn("SCL", toks)

    def test_img_jp2_paths_accepts_l2a_layers(self):
        with tempfile.TemporaryDirectory() as td:
            safe = Path(td) / "S2B_MSIL2A_20200101T000000_N0209_R000_T00XXX_20200101T000000.SAFE"
            img = safe / "GRANULE" / "L2A_T00XXX_A000000_20200101T000000" / "IMG_DATA_R10m"
            img.mkdir(parents=True)

            (img / "T00XXX_20200101T000000_B03_10m.jp2").touch()
            (img / "T00XXX_20200101T000000_AOT_10m.jp2").touch()

            (safe / "MTD_MSIL2A.xml").write_text("<xml></xml>")

            layout = S2Layout(str(safe))
            paths = layout.img_jp2_paths(["B03", "AOT"])
            self.assertTrue(paths["B03"].endswith(".jp2"))
            self.assertTrue(paths["AOT"].endswith(".jp2"))


if __name__ == "__main__":
    unittest.main()
