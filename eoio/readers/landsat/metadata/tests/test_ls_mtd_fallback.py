"""eoio.readers.landsat.metadata.tests.test_ls_mtd_fallback - tests for
eoio.readers.landsat.metadata.ls_mtd_fallback"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np
import rasterio
from rasterio.transform import from_origin

from eoio.readers.landsat.metadata.ls_mtd_fallback import (
    NOMINAL_BAND_WAVELENGTHS_NM,
    LSMTDFallbackReader,
)


def _make_band_tif(path, *, epsg=32633, width=4, height=3):
    """A real, minimal single-band GeoTIFF with a known CRS/transform/shape, standing in for
    one of a Landsat product's own band files."""
    transform = from_origin(500000, 4000000, 30, 30)  # 30 m pixels, arbitrary UTM origin
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=f"EPSG:{epsg}",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((height, width), dtype="uint16"), 1)


def _fake_layout(tif_path):
    layout = MagicMock()
    layout.tif_band_files.return_value = {"B2": str(tif_path)}
    return layout


def _fake_xml_reader(values: dict):
    xml_reader = MagicMock()
    xml_reader.find_value.side_effect = lambda key, default=None: values.get(key, default)
    return xml_reader


class TestFindAllBandCentralWavelengths(unittest.TestCase):
    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    def test_returns_the_nominal_oli_tirs_wavelengths(self):
        tif = self.tmp_path / "B2.TIF"
        _make_band_tif(tif)
        reader = LSMTDFallbackReader(_fake_layout(tif), _fake_xml_reader({}))

        self.assertEqual(reader.find_all_band_central_wavelengths(), NOMINAL_BAND_WAVELENGTHS_NM)
        # a defensive copy, not the module-level dict itself
        self.assertIsNot(reader.find_all_band_central_wavelengths(), NOMINAL_BAND_WAVELENGTHS_NM)


class TestFindAllBandGsds(unittest.TestCase):
    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    def test_maps_grid_cell_sizes_to_the_right_band_groups(self):
        tif = self.tmp_path / "B2.TIF"
        _make_band_tif(tif)
        xml_reader = _fake_xml_reader(
            {"grid_cell_size_reflective": 30.0, "grid_cell_size_panchromatic": 15.0, "grid_cell_size_thermal": 100.0}
        )
        reader = LSMTDFallbackReader(_fake_layout(tif), xml_reader)

        gsds = reader.find_all_band_gsds()

        self.assertEqual(gsds["B2"], 30.0)  # reflective
        self.assertEqual(gsds["B1"], 30.0)
        self.assertEqual(gsds["B8"], 15.0)  # panchromatic
        self.assertEqual(gsds["B10"], 100.0)  # thermal
        self.assertEqual(gsds["B11"], 100.0)

    def test_missing_grid_cell_size_fields_omit_those_bands(self):
        tif = self.tmp_path / "B2.TIF"
        _make_band_tif(tif)
        reader = LSMTDFallbackReader(_fake_layout(tif), _fake_xml_reader({}))

        self.assertEqual(reader.find_all_band_gsds(), {})


class TestFindEpsgShapeTransform(unittest.TestCase):
    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    def test_reads_epsg_shape_transform_from_a_real_geotiff(self):
        tif = self.tmp_path / "B2.TIF"
        _make_band_tif(tif, epsg=32633, width=4, height=3)
        reader = LSMTDFallbackReader(_fake_layout(tif), _fake_xml_reader({}))

        self.assertEqual(reader.find_epsg(), 32633)
        self.assertEqual(reader.find_proj_shape(), [3, 4])  # [rows, cols]
        transform = reader.find_proj_transform()
        self.assertEqual(len(transform), 6)
        self.assertAlmostEqual(transform[0], 30.0)  # pixel width

    def test_raster_metadata_is_cached_after_first_call(self):
        """Only one representative band file needs opening, and only once."""
        tif = self.tmp_path / "B2.TIF"
        _make_band_tif(tif)
        layout = _fake_layout(tif)
        reader = LSMTDFallbackReader(layout, _fake_xml_reader({}))

        reader.find_epsg()
        reader.find_proj_shape()
        reader.find_proj_transform()

        self.assertEqual(layout.tif_band_files.call_count, 1)


if __name__ == "__main__":
    unittest.main()
