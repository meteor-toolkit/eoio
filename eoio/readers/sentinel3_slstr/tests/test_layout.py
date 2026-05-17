"""Unit tests for eoio.readers.sentinel3_slstr.layout.

These mirror the style used by the sentinel3_olci layout tests: create a
minimal SEN3 directory structure in a temporary directory and validate the
behaviour of `S3SLSTRLayout` helpers.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eoio.readers.sentinel3_slstr.layout import S3SLSTRLayout, S3SLSTRLayoutError


class TestS3SLSTRLayout(unittest.TestCase):
    def _make_sen3_structure(
        self,
        root: Path,
        *,
        sen3_name: str = "S3_SLSTR_XXX.350",
        with_meas: bool = True,
        meas_names=None,
        with_geodetic: bool = False,
        with_cartesian: bool = False,
        with_flags: bool = False,
        with_indices: bool = False,
        with_met: bool = False,
        with_time: bool = False,
        with_viscal: bool = False,
        with_geometry_tn: bool = False,
        with_geometry_to: bool = False,
        with_manifest: bool = True,
    ) -> Path:
        if meas_names is None:
            meas_names = ["S1_radiance_an", "S7_BT_an"]

        sen3_path = root / sen3_name
        sen3_path.mkdir(parents=True, exist_ok=False)

        # optional manifest
        if with_manifest:
            (sen3_path / "xfdumanifest.xml").write_text("<?xml?>", encoding="utf-8")

        # measurement files
        if with_meas:
            for m in meas_names:
                (sen3_path / f"{m}.nc").write_text("netcdf", encoding="utf-8")

        if with_geodetic:
            (sen3_path / "geodetic_an.nc").write_text("netcdf", encoding="utf-8")
        if with_cartesian:
            (sen3_path / "cartesian_an.nc").write_text("netcdf", encoding="utf-8")
        if with_flags:
            (sen3_path / "flags_an.nc").write_text("netcdf", encoding="utf-8")
        if with_indices:
            (sen3_path / "indices_an.nc").write_text("netcdf", encoding="utf-8")
        if with_met:
            (sen3_path / "met_tx.nc").write_text("netcdf", encoding="utf-8")
        if with_time:
            (sen3_path / "time_an.nc").write_text("netcdf", encoding="utf-8")
        if with_viscal:
            (sen3_path / "viscal.nc").write_text("netcdf", encoding="utf-8")
        if with_geometry_tn:
            (sen3_path / "geometry_tn.nc").write_text("netcdf", encoding="utf-8")
        if with_geometry_to:
            (sen3_path / "geometry_to.nc").write_text("netcdf", encoding="utf-8")

        return sen3_path

    def test_init_raises_if_path_missing(self):
        with self.assertRaises(S3SLSTRLayoutError):
            S3SLSTRLayout(path="/tmp/nonexistent_sen3_path")

    def test_init_raises_if_not_directory(self):
        with TemporaryDirectory() as td:
            file_path = Path(td) / "not_a_dir.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaises(S3SLSTRLayoutError):
                S3SLSTRLayout(path=str(file_path))

    def test_init_and_proc_version(self):
        with TemporaryDirectory() as td:
            # choose a name containing a numeric token to exercise proc_version
            sen3_path = self._make_sen3_structure(Path(td), sen3_name="product.042")
            layout = S3SLSTRLayout(path=str(sen3_path))
            self.assertEqual(layout.path, str(sen3_path))
            self.assertEqual(layout.proc_version, 42)

    def test_meas_paths_discovery_and_specific(self):
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), meas_names=["S1_radiance_an", "S2_radiance_an"])
            layout = S3SLSTRLayout(path=str(sen3_path))
            all_paths = layout.meas_paths()
            self.assertIsInstance(all_paths, dict)
            self.assertIn("S1_radiance_an", all_paths)
            # request specific
            specific = layout.meas_paths(["S2_radiance_an"])
            self.assertEqual(len(specific), 1)
            self.assertIn("S2_radiance_an", specific)

    def test_requested_uncertainty_paths_empty(self):
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_meas=False)
            layout = S3SLSTRLayout(path=str(sen3_path))
            self.assertEqual(layout.requested_uncertainty_paths(), {})

    def test_default_meas_returns_list(self):
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td))
            layout = S3SLSTRLayout(path=str(sen3_path))
            dm = layout.default_meas()
            self.assertIsInstance(dm, list)
            self.assertGreaterEqual(len(dm), 1)

    def test_grid_and_mask_mapping(self):
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td))
            layout = S3SLSTRLayout(path=str(sen3_path))
            grid = layout.get_grid("S1_radiance_an")
            self.assertEqual(grid, "an")
            # grid resolution from utils
            res = layout.get_grid_res(grid)
            self.assertIsNotNone(res)
            # mask to var mapping for radiance and BT
            self.assertEqual(layout.mask_to_var("S1_exception_an"), "S1_radiance_an")
            self.assertEqual(layout.mask_to_var("S7_exception_an"), "S7_BT_an")
            self.assertIsNone(layout.mask_to_var("UNKNOWN_exception_an"))

    def test_optional_paths_exist_and_missing(self):
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(
                Path(td),
                with_geodetic=True,
                with_cartesian=True,
                with_flags=True,
                with_indices=True,
                with_met=True,
                with_time=True,
                with_viscal=True,
                with_geometry_tn=True,
                with_geometry_to=True,
            )
            layout = S3SLSTRLayout(path=str(sen3_path))
            self.assertIsNotNone(layout.geodetic_path("an"))
            self.assertIsNotNone(layout.cartesian_path("an"))
            self.assertIsNotNone(layout.flags_path("an"))
            self.assertIsNotNone(layout.indices_path("an"))
            self.assertIsNotNone(layout.met_path())
            self.assertIsNotNone(layout.time_path("an"))
            self.assertIsNotNone(layout.viscal_path())
            self.assertIsNotNone(layout.geometry_tn_path())
            self.assertIsNotNone(layout.geometry_to_path())
            # manifest exists by default in helper
            self.assertIsNotNone(layout.manifest_path())

    def test_optional_paths_missing(self):
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_meas=False, with_manifest=False)
            layout = S3SLSTRLayout(path=str(sen3_path))
            self.assertIsNone(layout.geodetic_path("an"))
            self.assertIsNone(layout.cartesian_path("an"))
            self.assertIsNone(layout.flags_path("an"))
            self.assertIsNone(layout.indices_path("an"))
            self.assertIsNone(layout.met_path())
            self.assertIsNone(layout.time_path("an"))
            self.assertIsNone(layout.viscal_path())
            self.assertIsNone(layout.geometry_tn_path())
            self.assertIsNone(layout.geometry_to_path())
            self.assertIsNone(layout.manifest_path())


if __name__ == "__main__":
    unittest.main()
