"""eoio.readers.constellr_skybee.tests.test_layout - unit tests for eoio.readers.constellr_skybee.layout"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eoio.readers.constellr_skybee.layout import ConstellrSkybeeLayout, ConstellrSkybeeLayoutError


class TestConstellrSkybeeLayout(unittest.TestCase):
    def _make_product(
        self,
        root: Path,
        *,
        product_name: str = "HIVE_L1C_TEST",
        band_tokens: list[str] | None = None,
        with_qa: bool = True,
    ) -> Path:
        if band_tokens is None:
            band_tokens = ["01", "02", "03", "04"]

        product = root / product_name
        product.mkdir(parents=True, exist_ok=False)

        for token in band_tokens:
            (product / f"{product_name}vnir{int(token):02d}.tiff").write_text("tif", encoding="utf-8")

        if with_qa:
            (product / f"{product_name}vnir{int(token)}_qa.tiff").write_text("qa", encoding="utf-8")

        (product / f"{product_name}.json").write_text("{}", encoding="utf-8")
        return product

    def test_metadata_file_prefers_metadata_json(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td))
            (product / "metadata.json").write_text("{}", encoding="utf-8")
            (product / "aaa_other.json").write_text("{}", encoding="utf-8")
            self.assertEqual(ConstellrSkybeeLayout(product).metadata_file(), product / "metadata.json")

    def test_metadata_file_warns_when_there_is_none(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td))
            (product / "HIVE_L1C_TEST.json").unlink()
            with self.assertWarnsRegex(UserWarning, "No metadata json"):
                self.assertIsNone(ConstellrSkybeeLayout(product).metadata_file())

    def test_init_raises_if_path_missing(self):
        with TemporaryDirectory() as td:
            missing = Path(td) / "missing_dir"
            with self.assertRaises(ConstellrSkybeeLayoutError):
                ConstellrSkybeeLayout(str(missing))

    def test_init_raises_if_not_directory(self):
        with TemporaryDirectory() as td:
            file_path = Path(td) / "file.txt"
            file_path.write_text("content", encoding="utf-8")
            with self.assertRaises(ConstellrSkybeeLayoutError):
                ConstellrSkybeeLayout(str(file_path))

    def test_image_files_returns_tiffs(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td))
            layout = ConstellrSkybeeLayout(str(product))
            files = layout.image_files()

            self.assertTrue(files)
            self.assertTrue(all(Path(f).suffix.lower() == ".tiff" for f in files))

    def test_image_qa_files_returns_qa_tiffs(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td))
            layout = ConstellrSkybeeLayout(str(product))
            files = layout.image_qa_files()

            self.assertTrue(files)
            self.assertIn("QA", Path(files[0]).name.upper())

    def test_image_file_prefers_vnir_file(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td))
            layout = ConstellrSkybeeLayout(str(product))
            img = layout.image_file()

            self.assertTrue(Path(img).name.lower().startswith("hive_l1c_testvnir"))

    def test_metadata_file_returns_json(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td))
            layout = ConstellrSkybeeLayout(str(product))
            metadata_file = layout.metadata_file()

            self.assertIsNotNone(metadata_file)
            self.assertTrue(metadata_file.name.endswith(".json"))

    def test_tif_band_files_returns_mapping(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td), band_tokens=["01", "02"])
            layout = ConstellrSkybeeLayout(str(product))
            files = layout.tif_band_files(meas_vars=["B1", "B2"])

            self.assertEqual(set(files.keys()), {"B1", "B2"})
            self.assertTrue(all(Path(f).suffix.lower() == ".tiff" for f in files.values()))

    def test_tif_band_files_raises_if_missing(self):
        with TemporaryDirectory() as td:
            product = self._make_product(Path(td), band_tokens=["02"])
            layout = ConstellrSkybeeLayout(str(product))

            with self.assertRaises(ConstellrSkybeeLayoutError):
                layout.tif_band_files(meas_vars=["B1"])


if __name__ == "__main__":
    unittest.main()
