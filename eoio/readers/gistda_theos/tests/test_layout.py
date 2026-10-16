"""eoio.readers.gistda_theos.tests.test_layout - unit tests for eoio.readers.gistda_theos.layout"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eoio.readers.gistda_theos.layout import GISTDATHEOSLayout, GISTDATHEOSLayoutError


class TestGistdaTheosLayout(unittest.TestCase):
    """Unit tests for GistdaTheosLayout class."""

    def _make_product(self, root: Path, *, img_name: str = "IMAGERY.TIF", dim_name: str = "METADATA.DIM"):
        """Create a minimal THEOS product directory with image and metadata files."""
        prod = root / "TH_CAT_22021001412034_1"
        prod.mkdir(parents=True, exist_ok=True)

        (prod / img_name).write_text("tif", encoding="utf-8")
        (prod / dim_name).write_text("<Dimap_Document />", encoding="utf-8")
        return prod

    def test_init_raises_if_path_missing(self):
        with TemporaryDirectory() as td:
            missing = Path(td) / "missing_product"
            with self.assertRaises(GISTDATHEOSLayoutError):
                GISTDATHEOSLayout(str(missing))

    def test_init_raises_if_not_directory(self):
        with TemporaryDirectory() as td:
            file_path = Path(td) / "file.txt"
            file_path.write_text("content", encoding="utf-8")
            with self.assertRaises(GISTDATHEOSLayoutError):
                GISTDATHEOSLayout(str(file_path))

    def test_image_file(self):
        with TemporaryDirectory() as td:
            prod = self._make_product(Path(td))
            layout = GISTDATHEOSLayout(str(prod))
            img = layout.image_file()

            self.assertIsInstance(img, str)
            self.assertTrue(Path(img).name.upper().endswith(".TIF"))
            self.assertIn("IMAGERY", Path(img).name.upper())

    def test_metadata_file(self):
        with TemporaryDirectory() as td:
            prod = self._make_product(Path(td))
            layout = GISTDATHEOSLayout(str(prod))
            xml_file = layout.metadata_file()

            self.assertIsNotNone(xml_file)
            self.assertTrue(Path(xml_file).name.upper().endswith(".DIM"))

    def test_metadata_file_raises_when_missing(self):
        with TemporaryDirectory() as td:
            prod = Path(td) / "TH_CAT_22021001412034_1"
            prod.mkdir(parents=True, exist_ok=True)
            (prod / "IMAGERY.TIF").write_text("tif", encoding="utf-8")

            layout = GISTDATHEOSLayout(prod)
            with self.assertRaises(GISTDATHEOSLayoutError):
                layout.metadata_file()

    def test_finds_metadata_one_level_down_when_product_dir_is_the_outer_wrapper(self):
        """A real THEOS delivery zip extracts to an outer container that also holds
        loose README/PDF/LOGO wrapper files alongside the actual DIMAP folder (see
        eoio.utils.read_utils.extract_zipped_file) -- product_dir here is that
        outer wrapper, not the DIMAP folder itself."""
        with TemporaryDirectory() as td:
            outer = Path(td) / "TH_CAT_1_1M_SITE_20220101"
            outer.mkdir(parents=True)
            (outer / "README.HTM").write_text("readme", encoding="utf-8")
            (outer / "LOGO.JPG").write_text("logo", encoding="utf-8")
            inner = outer / "TH_CAT_1"
            inner.mkdir()
            (inner / "IMAGERY.TIF").write_text("tif", encoding="utf-8")
            (inner / "METADATA.DIM").write_text("<Dimap_Document />", encoding="utf-8")

            layout = GISTDATHEOSLayout(outer)

            self.assertEqual(layout.dimap_dir, inner)
            self.assertEqual(Path(layout.image_file()), inner / "IMAGERY.TIF")
            self.assertEqual(layout.metadata_file(), inner / "METADATA.DIM")

    def test_raises_when_no_unique_dimap_folder_found(self):
        """product_dir with no METADATA.DIM at its own top level, and none (or more
        than one) in an immediate subdirectory, is an unresolvable layout -- not a
        silent pick of the wrong one."""
        with TemporaryDirectory() as td:
            outer = Path(td) / "TH_CAT_1_1M_SITE_20220101"
            outer.mkdir(parents=True)
            (outer / "README.HTM").write_text("readme", encoding="utf-8")

            layout = GISTDATHEOSLayout(outer)
            with self.assertRaises(GISTDATHEOSLayoutError):
                _ = layout.dimap_dir

    def test_aux_data_dir_and_aux_path(self):
        with TemporaryDirectory() as td:
            prod = self._make_product(Path(td))
            aux_dir = prod / "AUX_DATA"
            aux_dir.mkdir()
            (aux_dir / "ANGLE.txt").write_text("angles", encoding="utf-8")

            layout = GISTDATHEOSLayout(prod)
            self.assertEqual(layout.aux_dir(), aux_dir)
            self.assertIsNotNone(layout.aux_data_dir("ANGLE.txt"))
            self.assertEqual(layout.aux_path("ANGLE.txt"), layout.aux_data_dir("ANGLE.txt"))


if __name__ == "__main__":
    unittest.main()
