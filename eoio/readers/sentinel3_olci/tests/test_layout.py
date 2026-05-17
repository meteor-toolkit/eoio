"""Unit tests for eoio.readers.sentinel3_olci.layout.

These tests follow the sentinel2 style: small, focused, and using temporary
directories to create minimal SEN3 structures for validation.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from eoio.readers.sentinel3_olci.layout import S3OLCILayout, S3LayoutError


class TestS3OLCILayout(unittest.TestCase):
    """Test cases for S3OLCILayout."""

    def _make_sen3_structure(
        self,
        root: Path,
        *,
        sen3_name: str = "S3A_OL_1_EFR____20230615T103045_20230615T104015_0350_078_OLCI_OBC_____LN3_O_ST_002.SEN3",
        with_radiance_bands: bool = True,
        band_names=None,
        with_geo_coords: bool = False,
        with_quality_flags: bool = False,
        with_tie_geo: bool = False,
        with_tie_geom: bool = False,
        with_removed_pixels: bool = False,
    ) -> Path:
        """Create a minimal SEN3 directory structure."""
        if band_names is None:
            band_names = ["Oa01", "Oa02", "Oa03"]

        sen3_path = root / sen3_name
        sen3_path.mkdir(parents=True, exist_ok=False)

        # Create xfdumanifest.xml
        (sen3_path / "xfdumanifest.xml").write_text("<?xml version='1.0'?><XFDU/>", encoding="utf-8")

        # Create radiance bands
        if with_radiance_bands:
            for band in band_names:
                (sen3_path / f"{band}_radiance.nc").write_text("netcdf", encoding="utf-8")

        # Create optional coordinate files
        if with_geo_coords:
            (sen3_path / "geo_coordinates.nc").write_text("netcdf", encoding="utf-8")
        if with_quality_flags:
            (sen3_path / "qualityFlags.nc").write_text("netcdf", encoding="utf-8")
        if with_tie_geo:
            (sen3_path / "tie_geo_coordinates.nc").write_text("netcdf", encoding="utf-8")
        if with_tie_geom:
            (sen3_path / "tie_geometries.nc").write_text("netcdf", encoding="utf-8")
        if with_removed_pixels:
            (sen3_path / "removed_pixels.nc").write_text("netcdf", encoding="utf-8")

        return sen3_path

    def test_init_raises_if_path_missing(self):
        """Test that initialization fails for missing path."""
        with self.assertRaises(S3LayoutError):
            S3OLCILayout(path="/tmp/nonexistent_sen3_path")

    def test_init_raises_if_path_not_directory(self):
        """Test that initialization fails if path is not a directory."""
        with TemporaryDirectory() as td:
            file_path = Path(td) / "test.txt"
            file_path.write_text("test", encoding="utf-8")
            with self.assertRaises(S3LayoutError):
                S3OLCILayout(path=str(file_path))

    def test_init_succeeds_with_valid_path(self):
        """Test successful initialization with valid SEN3 directory."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td))
            layout = S3OLCILayout(path=str(sen3_path))
            self.assertEqual(layout.path, str(sen3_path))

    def test_radiance_paths_returns_dict(self):
        """Test that radiance_paths returns a dictionary."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), band_names=["Oa01", "Oa02"])
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.radiance_paths()
            self.assertIsInstance(result, dict)
            self.assertIn("Oa01", result)
            self.assertIn("Oa02", result)

    def test_radiance_paths_with_specific_bands(self):
        """Test radiance_paths with specific band selection."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), band_names=["Oa01", "Oa02", "Oa03"])
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.radiance_paths(["Oa01", "Oa02"])
            self.assertEqual(len(result), 2)
            self.assertIn("Oa01", result)
            self.assertIn("Oa02", result)

    def test_requested_radiance_paths(self):
        """Test requested_radiance_paths method."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), band_names=["Oa01", "Oa02", "Oa03"])
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.radiance_paths(["Oa01", "Oa03"])
            self.assertEqual(len(result), 2)
            self.assertIn("Oa01", result)
            self.assertIn("Oa03", result)
            self.assertNotIn("Oa02", result)

    def test_requested_uncertainty_paths(self):
        """Test requested_uncertainty_paths method."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_radiance_bands=False)
            # Create uncertainty files
            (sen3_path / "Oa01_unc.nc").write_text("netcdf", encoding="utf-8")
            (sen3_path / "Oa02_unc.nc").write_text("netcdf", encoding="utf-8")
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.requested_uncertainty_paths(["Oa01", "Oa02"])
            self.assertEqual(len(result), 2)
            self.assertIn("Oa01", result)
            self.assertIn("Oa02", result)

    def test_default_meas_returns_available_bands(self):
        """Test that default_meas returns available bands."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), band_names=["Oa01", "Oa02", "Oa03"])
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.default_meas()
            # API now returns a list of available measurement vars
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)

    def test_geo_coordinates_path_exists(self):
        """Test geo_coordinates_path returns path when file exists."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_geo_coords=True)
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.geo_coordinates_path()
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("geo_coordinates.nc"))

    def test_geo_coordinates_path_missing(self):
        """Test geo_coordinates_path returns None when file missing."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_geo_coords=False)
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.geo_coordinates_path()
            self.assertIsNone(result)

    def test_quality_flags_path_exists(self):
        """Test quality_flags_path returns path when file exists."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_quality_flags=True)
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.quality_flags_path()
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("qualityFlags.nc"))

    def test_quality_flags_path_missing(self):
        """Test quality_flags_path returns None when file missing."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_quality_flags=False)
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.quality_flags_path()
            self.assertIsNone(result)

    def test_tie_geo_coordinates_path_exists(self):
        """Test tie_geo_coordinates_path returns path when file exists."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_tie_geo=True)
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.tie_geo_coordinates_path()
            self.assertIsNotNone(result)

    def test_tie_geometries_path_exists(self):
        """Test tie_geometries_path returns path when file exists."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_tie_geom=True)
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.tie_geometries_path()
            self.assertIsNotNone(result)

    def test_removed_pixels_path_exists(self):
        """Test removed_pixels_path returns path when file exists."""
        with TemporaryDirectory() as td:
            sen3_path = self._make_sen3_structure(Path(td), with_removed_pixels=True)
            layout = S3OLCILayout(path=str(sen3_path))
            result = layout.removed_pixels_path()
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
