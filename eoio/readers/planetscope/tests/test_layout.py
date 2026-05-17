"""eoio.readers.planetscope.tests.test_layout - unit tests for eoio.readers.planetscope.layout"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from eoio.readers.planetscope.layout import PlanetScopeLayout, PlanetScopeLayoutError


class TestPlanetScopeLayout(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def _make_planetscope_files(
        self,
        base_name: str = "20220401_084707_00_2477",
        with_metadata: bool = True,
        with_mask: bool = True,
        del_mask: bool = False,
    ) -> Path:
        """
        Create minimal PlanetScope-like file structure.

        :param base_name: Base filename for the product
        :param with_metadata: Whether to create metadata files
        :param with_mask: Whether to create mask files
        :return: Path to the created image file
        """
        # Create image file
        img_file = self.temp_dir / f"{base_name}_3B_AnalyticMS_8b_clip.tif"
        img_file.write_text("dummy_tif", encoding="utf-8")

        if with_mask and not del_mask:
            # Create mask file
            mask_file = self.temp_dir / f"{base_name}_3B_udm2_clip.tif"
            mask_file.write_text("dummy_mask", encoding="utf-8")

        if with_metadata:
            # Create XML metadata file
            xml_file = self.temp_dir / f"{base_name}_3B_AnalyticMS_8b_metadata_clip.xml"
            xml_file.write_text("<?xml version='1.0'?><metadata/>", encoding="utf-8")

            # Create JSON metadata file
            json_file = self.temp_dir / f"{base_name}.json"
            json_data = {
                "type": "Feature",
                "geometry_id": {},
                "properties": {
                    "acquired": "2022-04-01T08:47:07.003186Z",
                    "instrument": "PSB.SD",
                },
                "assets": {"analytic_ms": {"href": str(img_file)}},
                "other": {
                    "proj:epsg": 32733,
                    "proj:bbox": [510480.0, 7389249.0, 513738.0, 7390662.0],
                },
            }
            json_file.write_text(json.dumps(json_data), encoding="utf-8")

        return img_file

    def test_init_with_valid_file(self):
        img_path = self._make_planetscope_files()
        layout = PlanetScopeLayout(image_file=str(img_path))
        self.assertEqual(layout.image_file, str(img_path))

    def test_init_raises_if_path_missing(self):
        nonexistent = str(self.temp_dir / "nonexistent.tif")
        with self.assertRaises(PlanetScopeLayoutError):
            PlanetScopeLayout(image_file=nonexistent)

    def test_init_raises_if_not_file(self):
        with self.assertRaises(PlanetScopeLayoutError):
            PlanetScopeLayout(image_file=str(self.temp_dir))

    def test_mask_file_returns_path_when_found(self):
        img_path = self._make_planetscope_files(with_mask=True)
        layout = PlanetScopeLayout(image_file=str(img_path))
        mask_file = layout.mask_file()

        self.assertIsNotNone(mask_file)
        self.assertIsInstance(mask_file, str)

    def test_mask_file_raises_error_when_too_many_found(self):
        img_path = self._make_planetscope_files(with_mask=True)
        extra_mask_file = self.temp_dir / "20220401_084707_00_2477_udm2_extra.tif"
        extra_mask_file.write_text("extra_mask", encoding="utf-8")

        layout = PlanetScopeLayout(image_file=str(img_path))
        with self.assertRaises(PlanetScopeLayoutError) as ctx:
            layout.mask_file()
            self.assertIn("More than one", str(ctx.exception))

    def test_mask_file_raises_error_when_no_mask_file_found(self):
        img_path = self._make_planetscope_files(with_mask=True, del_mask=True)
        # del created mask file

        layout = PlanetScopeLayout(image_file=str(img_path))
        with self.assertRaises(PlanetScopeLayoutError) as ctx:
            layout.mask_file()
            self.assertIn("No corresponding mask file", str(ctx.exception))

    def test_mask_file_identifies_correct_pattern(self):
        img_path = self._make_planetscope_files(base_name="20230115_103045_ssc1_u0003")
        layout = PlanetScopeLayout(image_file=str(img_path))
        mask_file = layout.mask_file()

        self.assertIsNotNone(mask_file)
        self.assertTrue("udm2" in mask_file)
        self.assertTrue("20230115_103045_ssc1" in mask_file)

    def test_metadata_files_returns_list_when_found(self):
        img_path = self._make_planetscope_files(with_metadata=True)
        layout = PlanetScopeLayout(image_file=str(img_path))
        metadata_files = layout.metadata_files()

        self.assertIsNotNone(metadata_files)
        self.assertIsInstance(metadata_files, list)
        self.assertGreater(len(metadata_files), 0)

    def test_metadata_files_returns_none_when_not_found(self):
        img_path = self._make_planetscope_files(with_metadata=False)
        layout = PlanetScopeLayout(image_file=str(img_path))
        metadata_files = layout.metadata_files()

        self.assertIsNone(metadata_files)

    def test_metadata_files_includes_xml_and_json(self):
        img_path = self._make_planetscope_files(with_metadata=True)
        layout = PlanetScopeLayout(image_file=str(img_path))
        metadata_files = layout.metadata_files()

        self.assertTrue(any(f.endswith(".xml") for f in metadata_files))
        self.assertTrue(any(f.endswith(".json") for f in metadata_files))


if __name__ == "__main__":
    unittest.main()
