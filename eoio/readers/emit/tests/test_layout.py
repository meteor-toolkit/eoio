"""Tests for eoio.readers.emit.layout.EmitLayout"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eoio.readers.emit import layout


class TestEmitLayout(unittest.TestCase):
    def _make_pair(self, root: Path, base: str = "example"):
        rad = root / f"{base}_RAD_v01"
        obs = root / f"{base}_OBS_v01"
        rad.mkdir()
        obs.mkdir()
        return rad, obs

    def test_format_path_with_rad(self):
        p = "/some/path/TO_RAD_product"
        self.assertEqual(layout.EmitLayout.format_path(p), p)

    def test_format_path_with_obs(self):
        p_obs = "/some/path/TO_OBS_product"
        p_rad = "/some/path/TO_RAD_product"
        self.assertEqual(layout.EmitLayout.format_path(p_obs), p_rad)

    def test_format_path_invalid(self):
        with self.assertRaises(ValueError):
            layout.EmitLayout.format_path("/no/token/here")

    def test_check_path_both_exist(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            rad, obs = self._make_pair(root, base="example")
            self.assertIsNone(layout.EmitLayout.check_path(str(rad)))

    def test_check_path_missing_obs(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            rad = root / "only_RAD"
            rad.mkdir()
            with self.assertRaises(ValueError):
                layout.EmitLayout.check_path(str(rad))

    def test_check_path_missing_rad(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            obs = root / "missing_OBS"
            obs.mkdir()
            rad_path = root / "missing_RAD"
            with self.assertRaises(ValueError):
                layout.EmitLayout.check_path(str(rad_path))

    def test_emitlayout_init_normalizes_and_checks(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            rad, obs = self._make_pair(root, base="myprod")
            inst = layout.EmitLayout(str(obs))
            self.assertIn("RAD", inst.path)
            self.assertEqual(inst.path, str(rad))


if __name__ == "__main__":
    unittest.main()
