import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eoio.readers.copernicus_dem.layout import CopernicusDEMLayout, get_layout


class TestCopernicusDEMLayout(unittest.TestCase):
    def test_metadata_file_returns_xml(self):
        layout = CopernicusDEMLayout(
            dem=Path("dem.tif"),
            xml=Path("metadata.xml"),
            wbm=None,
            flm=None,
            edm=None,
            hem=None,
        )

        self.assertEqual(layout.metadata_file(), Path("metadata.xml"))

    def test_metadata_file_returns_none_when_xml_missing(self):
        layout = CopernicusDEMLayout(
            dem=Path("dem.tif"),
            xml=None,
            wbm=None,
            flm=None,
            edm=None,
            hem=None,
        )

        self.assertIsNone(layout.metadata_file())


class TestGetLayout(unittest.TestCase):
    def test_get_layout_with_all_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aux = root / "AUXFILES"
            aux.mkdir()

            dem = root / "TEST_DEM.tif"
            xml = root / "metadata.xml"
            wbm = aux / "TEST_WBM.tif"
            flm = aux / "TEST_FLM.tif"
            edm = aux / "TEST_EDM.tif"
            hem = aux / "TEST_HEM.tif"

            for file in [dem, xml, wbm, flm, edm, hem]:
                file.touch()

            layout = get_layout(root)

            self.assertEqual(layout.dem, dem)
            self.assertEqual(layout.xml, xml)
            self.assertEqual(layout.wbm, wbm)
            self.assertEqual(layout.flm, flm)
            self.assertEqual(layout.edm, edm)
            self.assertEqual(layout.hem, hem)

    def test_get_layout_with_only_required_dem(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            dem = root / "TEST_DEM.tif"
            dem.touch()

            layout = get_layout(root)

            self.assertEqual(layout.dem, dem)
            self.assertIsNone(layout.xml)
            self.assertIsNone(layout.wbm)
            self.assertIsNone(layout.flm)
            self.assertIsNone(layout.edm)
            self.assertIsNone(layout.hem)

    def test_get_layout_raises_when_dem_missing(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaises(StopIteration):
                get_layout(root)


if __name__ == "__main__":
    unittest.main()
