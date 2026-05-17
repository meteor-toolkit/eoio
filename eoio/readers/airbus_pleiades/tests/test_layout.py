"""eoio.readers.airbus_pleiades.tests.test_layout - unit tests for eoio.readers.airbus_pleiades.layout"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from eoio.readers.airbus_pleiades.layout import PleiadesLayout, PleiadesLayoutError


class TestPleiadesLayout(unittest.TestCase):
    """Unit tests for PleiadesLayout class."""

    def _make_product(
        self,
        root: Path,
        *,
        img_name: str = "IMG_test.TIF",
        xml_name: str = "DIM_test.XML",
        with_mask: bool = False,
        mask_name: str = "AUX_TEST_MSK.GML",
    ):
        """Create a minimal Pleiades product directory with IMG and DIM files."""
        prod = root / "PLEIADES_PRODUCT"
        prod.mkdir(parents=True, exist_ok=True)

        # create an image file and metadata XML
        img = prod / img_name
        img.write_text("tif", encoding="utf-8")

        xml = prod / xml_name
        xml.write_text("<xml/>", encoding="utf-8")

        if with_mask:
            masks_dir = prod / "MASKS"
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask = masks_dir / mask_name
            mask.write_text("gml", encoding="utf-8")

        return prod

    def test_init_raises_if_path_missing(self):
        """Test that __init__ raises PleiadesLayoutError for non-existent path."""
        with TemporaryDirectory() as td:
            missing = Path(td) / "missing_product"
            with self.assertRaises(PleiadesLayoutError) as ctx:
                PleiadesLayout(str(missing))
            self.assertIn("not found", str(ctx.exception))

    def test_init_raises_if_not_directory(self):
        """Test that __init__ raises PleiadesLayoutError if path is not a directory."""
        with TemporaryDirectory() as td:
            file_path = Path(td) / "file.txt"
            file_path.write_text("content", encoding="utf-8")
            with self.assertRaises(PleiadesLayoutError) as ctx:
                PleiadesLayout(str(file_path))
            self.assertIn("not a dir", str(ctx.exception))

    def test_init_with_valid_directory(self):
        """Test that __init__ succeeds with a valid directory path."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root)

            layout = PleiadesLayout(str(prod))
            self.assertEqual(Path(layout.product_dir), Path(prod))

    def test_tif_img_pardir_path(self):
        """Test that tif_img_pardir_path property returns correct path."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root)

            layout = PleiadesLayout(str(prod))
            self.assertEqual(layout.tif_img_pardir_path, prod)

    def test_image_file(self):
        """Test that image_file() returns a valid image file path."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root)

            layout = PleiadesLayout(prod)
            img = layout.image_file()

            self.assertIsInstance(img, str)
            self.assertTrue(Path(img).name.startswith("IMG"))
            self.assertTrue(str(img).upper().endswith(".TIF"))

    def test_image_file_raises_when_missing(self):
        """Test that image_file() raises PleiadesLayoutError when no TIF files found."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = root / "PLEIADES_PRODUCT"
            prod.mkdir(parents=True, exist_ok=True)

            # create XML but no image
            xml = prod / "DIM_test.XML"
            xml.write_text("<xml/>", encoding="utf-8")

            layout = PleiadesLayout(prod)
            with self.assertRaises(PleiadesLayoutError) as ctx:
                layout.image_file()
            self.assertIn("No image TIFs found", str(ctx.exception))

    def test_metadata_file(self):
        """Test that metadata_file() returns a valid XML file path."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root)

            layout = PleiadesLayout(str(prod))
            xml_file = layout.metadata_file()

            self.assertIsNotNone(xml_file)
            self.assertTrue("DIM" in Path(xml_file).name.upper() or str(xml_file).upper().endswith(".XML"))

    def test_metadata_file_raises_when_missing(self):
        """Test that metadata_file() raises PleiadesLayoutError when no XML found."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = root / "PLEIADES_PRODUCT"
            prod.mkdir(parents=True, exist_ok=True)
            # create image but no metadata XML
            img = prod / "IMG_test.TIF"
            img.write_text("tif", encoding="utf-8")

            layout = PleiadesLayout(prod)
            with self.assertRaises(PleiadesLayoutError) as ctx:
                layout.metadata_file()
            self.assertIn("No metadata XML file found", str(ctx.exception))

    def test_aux_dir(self):
        """Test that aux_dir() returns MASKS directory path."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root)

            layout = PleiadesLayout(prod)
            aux_dir = layout.aux_dir()

            self.assertEqual(aux_dir, prod / "MASKS")

    def test_aux_data_dir_returns_path_when_exists(self):
        """Test that aux_data_dir() and aux_path() return path when auxiliary file exists."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root, with_mask=True, mask_name="AUX_EXISTS_MSK.GML")

            layout = PleiadesLayout(prod)
            aux = layout.aux_data_dir("AUX_EXISTS_MSK.GML")
            aux_path = layout.aux_path("AUX_EXISTS_MSK")

            self.assertIsNotNone(aux)
            self.assertEqual(aux, prod / "MASKS" / "AUX_EXISTS_MSK.GML")

            self.assertIsNotNone(aux_path)
            self.assertEqual(aux_path, prod / "MASKS" / "AUX_EXISTS_MSK.GML")
            self.assertTrue(aux_path.exists())

    def test_aux_data_dir_returns_none_when_missing(self):
        """Test that aux_data_dir() returns None when auxiliary file doesn't exist."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root)

            layout = PleiadesLayout(prod)
            aux = layout.aux_data_dir("AUX_NONEXISTENT.GML")

            self.assertIsNone(aux)

    def test_aux_path_returns_none_when_missing(self):
        """Test that aux_path() returns None when auxiliary file not found."""
        with TemporaryDirectory() as td:
            root = Path(td)
            prod = self._make_product(root)

            layout = PleiadesLayout(prod)
            aux = layout.aux_path("AUX_NONEXISTENT")

            self.assertIsNone(aux)


if __name__ == "__main__":
    unittest.main()
