"""eoio.readers.landsat.tests.test_reader - unit tests for eoio.readers.landsat.reader"""

import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import xarray as xr
from eoio.readers.landsat.reader import LandsatReader


class TestLandsatReader(unittest.TestCase):
    def _make_reader_minimal(self, product_metadata=None):
        """
        Construct a reader instance without calling the real __init__.
        """
        reader = object.__new__(LandsatReader)

        reader.path = None
        reader.layout = Mock(name="layout")
        reader.mtd = Mock(name="mtd")
        reader.aux = Mock(name="aux")
        reader.aux.attach_aux.side_effect = lambda ds, aux: ds

        if product_metadata:
            reader.mtd.basic_metadata = product_metadata
        else:
            reader.mtd.basic_metadata = {}

        return reader

    # -------------------------
    # resolve_subset() tests
    # -------------------------

    @patch("eoio.readers.landsat.reader.ROISubsetResolver")
    def test_resolve_subset_ReturnsNoneWhenSubsetIsNone(self, ROISubsetResolver):
        reader = self._make_reader_minimal()
        self.assertIsNone(reader.resolve_subset(None))
        ROISubsetResolver.assert_not_called()

    @patch("eoio.readers.landsat.reader.ROISubsetResolver")
    def test_resolve_subset_RaisesIfImageCrsMissing(self, ROISubsetResolver):
        reader = self._make_reader_minimal(product_metadata={"epsg": None})

        with self.assertRaises(ValueError) as ctx:
            reader.resolve_subset({"roi": [0, 0, 1, 1], "roi_crs": 4326})

        self.assertIn("product CRS", str(ctx.exception))

    @patch("eoio.readers.landsat.reader.ROISubsetResolver")
    def test_resolve_subset_CallsResolver(self, ROISubsetResolver):
        reader = self._make_reader_minimal(product_metadata={"epsg": 32630})

        subset = {"roi": [0, 0, 1, 1], "roi_crs": 4326}
        reader.resolve_subset(subset)

        ROISubsetResolver.assert_called_once_with(
            roi=[0, 0, 1, 1],
            roi_crs_epsg=4326,
            image_crs_epsg=32630,
            image_bounds=None,
        )

    # -------------------------
    # open_dataset() tests
    # -------------------------

    @patch("eoio.readers.landsat.reader.apply_conventions")
    @patch("eoio.readers.landsat.reader.read_bands_into_dataset")
    def test_open_dataset_ReadsBands(self, read_bands_into_dataset, apply_conventions):
        reader = self._make_reader_minimal()
        reader.resolved_config = SimpleNamespace(
            vars_sel={"meas": ["B1"], "aux": []},
            subset=None,
            read_params={"metadata_level": None, "use_chunks": False, "chunks": None},
        )

        ds1 = xr.Dataset({"B1": (("y", "x"), [[1]])})
        ds2 = xr.Dataset({"B1": (("y", "x"), [[1]])})  # Conventions applied

        read_bands_into_dataset.return_value = ds1
        apply_conventions.return_value = ds2

        out = reader.open_dataset()
        self.assertIs(out, ds2)

        read_bands_into_dataset.assert_called_once()
        apply_conventions.assert_called_once()


if __name__ == "__main__":
    unittest.main()
