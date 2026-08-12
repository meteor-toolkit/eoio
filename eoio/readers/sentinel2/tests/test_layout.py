"""eoio.readers.sentinel2.tests.test_layout - unit tests for eoio.readers.sentinel2.layout"""

import unittest
from typing import List, Optional
from pathlib import Path
from tempfile import TemporaryDirectory
from eoio.readers.sentinel2.layout import S2Layout, S2LayoutError

# Path to a real Sentinel-2 SAFE product for integration testing.
# If you run the tests, ensure this path points to a valid SAFE product on your system.
# Otherwise, the integration test will be skipped.
# SAFE_PATH = "C:\Users\seh2\OneDrive - National Physical Laboratory\Data\Archive\S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"
SAFE_PATH = "/Users/seh2/Library/CloudStorage/OneDrive-NationalPhysicalLaboratory/Data/archive/sentinel2/S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"


class TestS2Layout(unittest.TestCase):
    def _make_safe(
        self,
        root: Path,
        *,
        safe_name: str = "S2A_TEST.SAFE",
        granule_name: str = "G1",
        with_tl_xml: bool = True,
        with_img_data: bool = True,
        jp2_band_tokens: Optional[List[str]] = None,
        with_aux: bool = False,
        datastrip_names: Optional[List[str]] = None,
        with_ds_xml: bool = False,
    ) -> Path:
        """
        Create a minimal SAFE-like directory tree under `root`.

        :param jp2_band_tokens: e.g. ["02", "03", "8A"] -> filenames contain B02, B03, B8A
        :param datastrip_names: e.g. ["DS1"] or ["DS1", "DS2"] creates DATASTRIP dirs
        :param with_ds_xml: if True, write MTD_DS.xml in each datastrip dir
        """
        if jp2_band_tokens is None:
            jp2_band_tokens = ["02", "03"]

        safe = root / safe_name
        safe.mkdir(parents=True, exist_ok=False)

        # Optional product metadata XML
        (safe / "MTD_MSIL1C.xml").write_text("<xml/>", encoding="utf-8")

        granule = safe / "GRANULE" / granule_name
        granule.mkdir(parents=True, exist_ok=False)

        if with_tl_xml:
            (granule / "MTD_TL.xml").write_text("<xml/>", encoding="utf-8")

        if with_img_data:
            img_data = granule / "IMG_DATA"
            img_data.mkdir(parents=True, exist_ok=False)
            for tok in jp2_band_tokens:
                # Must match S2Layout band regex: B(?P<band>\d{1,2}A?)
                (img_data / f"T0000_B{tok}_10m.jp2").write_text("jp2", encoding="utf-8")

        if with_aux:
            aux_dir = granule / "AUX_DATA"
            aux_dir.mkdir(parents=True, exist_ok=False)
            (aux_dir / "AUX_ECMWFT").write_text("aux", encoding="utf-8")

        # datastrip structure
        if datastrip_names is not None:
            ds_base = safe / "DATASTRIP"
            ds_base.mkdir(parents=True, exist_ok=False)
            for name in datastrip_names:
                ds_dir = ds_base / name
                ds_dir.mkdir(parents=True, exist_ok=False)
                if with_ds_xml:
                    (ds_dir / "MTD_DS.xml").write_text("<xml/>", encoding="utf-8")

        return safe

    def test_init_raises_if_path_missing(self):
        with TemporaryDirectory() as td:
            missing = Path(td) / "missing.SAFE"
            with self.assertRaises(S2LayoutError):
                S2Layout(str(missing))

    def test_init_raises_if_not_directory(self):
        with TemporaryDirectory() as td:
            f = Path(td) / "file.SAFE"
            f.write_text("x", encoding="utf-8")
            with self.assertRaises(S2LayoutError):
                S2Layout(str(f))

    def test_granule_dirs_and_granule_dir(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), granule_name="G1")
            layout = S2Layout(str(safe))

            granules = layout.granule_dirs()
            self.assertEqual(len(granules), 1)
            self.assertEqual(granules[0].name, "G1")
            self.assertEqual(layout.granule_dir().name, "G1")

    def test_product_metadata_xml(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td))
            layout = S2Layout(str(safe))

            p = layout.product_metadata_xml()
            self.assertIsNotNone(p)
            self.assertTrue(p.name.startswith("MTD_MSI"))

    def test_tl_metadata_xml_found(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), with_tl_xml=True)
            layout = S2Layout(str(safe))

            tl = layout.tl_metadata_xml()
            self.assertTrue(tl.exists())
            self.assertTrue(tl.name.startswith("MTD_TL"))

    def test_tl_metadata_xml_missing_raises(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), with_tl_xml=False)
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.tl_metadata_xml()

    def test_img_data_dirs_found(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), with_img_data=True)
            layout = S2Layout(str(safe))

            dirs = layout.img_data_dirs()
            self.assertGreaterEqual(len(dirs), 1)
            # dirs is list of (res, path)
            self.assertTrue(any(p.name.startswith("IMG_DATA") for _, p in dirs))

    def test_jp2_files_and_available_band_tokens(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), jp2_band_tokens=["02", "8A", "11"])
            layout = S2Layout(str(safe))

            jp2s = layout.jp2_files()
            self.assertEqual(len(jp2s), 3)

            bands = layout.available_band_tokens()
            self.assertIn("B02", bands)
            self.assertIn("B8A", bands)
            self.assertIn("B11", bands)

    def test_default_meas_vars(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), jp2_band_tokens=["02", "03"])
            layout = S2Layout(str(safe))

            self.assertEqual(set(layout.default_meas_vars()), {"B02", "B03"})

    def test_img_jp2_paths_success(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), jp2_band_tokens=["02", "03"])
            layout = S2Layout(str(safe))

            paths = layout.img_jp2_paths(["B02", "B03"])
            self.assertEqual(set(paths.keys()), {"B02", "B03"})
            for p in paths.values():
                self.assertTrue(Path(p).exists())

    def test_img_jp2_paths_preferred_resolution(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), jp2_band_tokens=["02"])
            granule = Path(safe) / "GRANULE" / "G1"
            img_data = granule / "IMG_DATA"
            (img_data / "T0000_B02_20m.jp2").write_text("jp2", encoding="utf-8")

            layout = S2Layout(str(safe))
            path = layout.img_jp2_paths(["B02"], prefer_res_m=20)["B02"]

            self.assertTrue(path.endswith("_20m.jp2"))

    def test_img_jp2_paths_missing_band_raises(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), jp2_band_tokens=["02"])
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.img_jp2_paths(["B03"])

    def test_aux_data_dir_and_aux_path(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), with_aux=True)
            layout = S2Layout(str(safe))

            aux_dir = layout.aux_data_dir()
            self.assertIsNotNone(aux_dir)
            self.assertEqual(aux_dir.name, "AUX_DATA")

            aux_file = layout.aux_path("AUX_ECMWFT")
            self.assertTrue(aux_file.exists())
            self.assertTrue(aux_file.name.startswith("AUX_ECMWFT"))

    def test_aux_path_missing_raises(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), with_aux=False)
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.aux_path("AUX_ECMWFT")

    def test_datastrip_dir_missing_raises(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=None)  # no DATASTRIP
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.datastrip_dir()

    def test_datastrip_dir_single(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=["DS1"], with_ds_xml=True)
            layout = S2Layout(str(safe))

            ds_dir = layout.datastrip_dir()
            self.assertTrue(ds_dir.exists())
            self.assertEqual(ds_dir.name, "DS1")
            self.assertEqual(ds_dir.parent.name, "DATASTRIP")

    def test_datastrip_dir_multiple_requires_id(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=["DS1", "DS2"], with_ds_xml=True)
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.datastrip_dir()

    def test_datastrip_dir_multiple_with_id(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=["DS1", "DS2"], with_ds_xml=True)
            layout = S2Layout(str(safe))

            ds_dir = layout.datastrip_dir(datastrip_id="DS2")
            self.assertTrue(ds_dir.exists())
            self.assertEqual(ds_dir.name, "DS2")

    def test_datastrip_dir_with_invalid_id_raises(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=["DS1"], with_ds_xml=True)
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.datastrip_dir(datastrip_id="DOES_NOT_EXIST")

    def test_ds_metadata_xml_single(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=["DS1"], with_ds_xml=True)
            layout = S2Layout(str(safe))

            p = layout.ds_metadata_xml()
            self.assertTrue(p.exists())
            self.assertEqual(p.name, "MTD_DS.xml")

    def test_ds_metadata_xml_multiple_requires_id(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=["DS1", "DS2"], with_ds_xml=True)
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.ds_metadata_xml()

    def test_ds_metadata_xml_multiple_with_id(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe(Path(td), datastrip_names=["DS1", "DS2"], with_ds_xml=True)
            layout = S2Layout(str(safe))

            p = layout.ds_metadata_xml(datastrip_id="DS1")
            self.assertTrue(p.exists())
            self.assertEqual(p.name, "MTD_DS.xml")
            self.assertIn("DATASTRIP", p.parts)
            self.assertIn("DS1", p.parts)

    def test_ds_metadata_xml_missing_file_raises(self):
        with TemporaryDirectory() as td:
            # DATASTRIP present but no MTD_DS.xml written
            safe = self._make_safe(Path(td), datastrip_names=["DS1"], with_ds_xml=False)
            layout = S2Layout(str(safe))

            with self.assertRaises(S2LayoutError):
                layout.ds_metadata_xml()


class TestS2LayoutData(unittest.TestCase):
    def test_real_safe(self):
        # Note: This test requires a real Sentinel-2 SAFE product at the specified path.
        # Update the SAFE_PATH to point to a valid SAFE product on your system for testing.
        safe_path = Path(SAFE_PATH)
        if not safe_path.exists():
            self.skipTest(f"Test SAFE product not found at {safe_path}")

        layout = S2Layout(str(safe_path))

        granules = layout.granule_dirs()
        self.assertGreaterEqual(len(granules), 1)

        p_xml = layout.product_metadata_xml()
        self.assertTrue(p_xml.exists())

        tl_xml = layout.tl_metadata_xml()
        self.assertTrue(tl_xml.exists())

        jp2s = layout.jp2_files()
        self.assertGreaterEqual(len(jp2s), 1)

        img_jp2_paths = layout.img_jp2_paths(["B02", "B03"])
        self.assertEqual(len(img_jp2_paths), 2)

        bands = layout.available_band_tokens()
        self.assertIn("B02", bands)  # Assuming B02 is present in the test data


class TestMskClassiPath(unittest.TestCase):
    """Tests for S2Layout.msk_classi_path (L1C cloud/snow classification mask)."""

    def _make_safe_with_qi(self, root: Path, qi_files: Optional[List[str]] = None) -> Path:
        safe = root / "S2A_TEST.SAFE"
        granule = safe / "GRANULE" / "G1"
        granule.mkdir(parents=True, exist_ok=False)
        (safe / "MTD_MSIL1C.xml").write_text("<xml/>", encoding="utf-8")
        (granule / "MTD_TL.xml").write_text("<xml/>", encoding="utf-8")
        if qi_files is not None:
            qi = granule / "QI_DATA"
            qi.mkdir(parents=True, exist_ok=False)
            for name in qi_files:
                (qi / name).write_text("jp2", encoding="utf-8")
        return safe

    def test_finds_msk_classi(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe_with_qi(Path(td), ["MSK_CLASSI_B00.jp2", "MSK_DETFOO_B02.jp2"])
            path = S2Layout(str(safe)).msk_classi_path()
            self.assertIsNotNone(path)
            self.assertTrue(path.endswith("MSK_CLASSI_B00.jp2"))

    def test_returns_none_when_only_legacy_masks_present(self):
        """Pre-04.00 baselines ship QA60.jp2 instead of MSK_CLASSI; those are not
        supported, and must yield None rather than a wrong file."""
        with TemporaryDirectory() as td:
            safe = self._make_safe_with_qi(Path(td), ["QA60.jp2", "MSK_DETFOO_B02.jp2"])
            self.assertIsNone(S2Layout(str(safe)).msk_classi_path())

    def test_returns_none_when_no_qi_data_dir(self):
        with TemporaryDirectory() as td:
            safe = self._make_safe_with_qi(Path(td), qi_files=None)
            self.assertIsNone(S2Layout(str(safe)).msk_classi_path())


if __name__ == "__main__":
    unittest.main()
