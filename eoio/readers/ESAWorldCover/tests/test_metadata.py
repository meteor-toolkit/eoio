import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import xarray as xr

from eoio.readers.ESAWorldCover.metadata import (
    ESAWorldCoverMetadataExtractor,
    FLAG_VALUES,
    FLAG_MEANINGS,
    FLAG_COLORS,
)


class TestESAWorldCoverMetadataExtractor(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock()
        self.reader.path = Path(
            "ESA_WorldCover_10m_2021_v200_Map.tif"
        )
        self.reader.config = SimpleNamespace(
            subset={"roi": ((0, 0), 100)}
        )

        self.ds = xr.Dataset(
            {
                "landcover_map": (
                    ("y", "x"),
                    [[10, 20], [30, 40]],
                )
            }
        )

    @patch("eoio.readers.ESAWorldCover.metadata.normalize_footprint")
    def test_get_basic_metadata(self, mock_normalize):
        mock_normalize.return_value = {}

        extractor = ESAWorldCoverMetadataExtractor(
            self.reader,
            self.ds,
        )

        md = extractor.get_basic_metadata()

        self.assertEqual(md["collection_name"], "ESA WorldCover")
        self.assertEqual(md["product_date"], "2021")
        self.assertEqual(md["product_bounds"], "")


    def test_variable_metadata(self):
        extractor = ESAWorldCoverMetadataExtractor(
            self.reader,
            self.ds,
        )

        md = extractor.get_variable_basic_metadata(
            "landcover_map"
        )

        self.assertEqual(md["units"], "1")
        self.assertEqual(
            md["long_name"],
            "Land cover classification",
        )
        self.assertEqual(
            md["flag_values"],
            FLAG_VALUES,
        )
        self.assertEqual(
            md["flag_meanings"],
            FLAG_MEANINGS,
        )
        self.assertEqual(
            md["flag_colors"],
            FLAG_COLORS,
        )

    def test_extract_year(self):
        extractor = ESAWorldCoverMetadataExtractor(
            self.reader,
            self.ds,
        )

        self.assertEqual(
            extractor._extract_year(),
            "2021",
        )

    def test_extract_year_no_year_found(self):
        self.reader.path = Path(
            "ESA_WorldCover_Map.tif"
        )

        extractor = ESAWorldCoverMetadataExtractor(
            self.reader,
            self.ds,
        )

        self.assertEqual(
            extractor._extract_year(),
            "",
        )

    def test_product_metadata_matches_basic(self):
        extractor = ESAWorldCoverMetadataExtractor(
            self.reader,
            self.ds,
        )

        self.assertEqual(
            extractor.get_product_metadata(),
            extractor.get_basic_metadata(),
        )

    @patch(
        "eoio.readers.ESAWorldCover.metadata.normalize_footprint"
    )
    def test_basic_metadata_without_rio(self, mock_normalize):
        mock_normalize.return_value = {}

        extractor = ESAWorldCoverMetadataExtractor(
            self.reader,
            self.ds,
        )

        md = extractor.get_basic_metadata()

        self.assertEqual(md["product_bounds"], "")
        self.assertEqual(
            md["collection_name"],
            "ESA WorldCover",
        )


class TestWorldCoverConstants(unittest.TestCase):

    def test_flag_metadata_lengths_match(self):
        self.assertEqual(
            len(FLAG_VALUES),
            len(FLAG_MEANINGS),
        )

        self.assertEqual(
            len(FLAG_VALUES),
            len(FLAG_COLORS),
        )

    def test_flag_values_unique(self):
        self.assertEqual(
            len(FLAG_VALUES),
            len(set(FLAG_VALUES)),
        )


if __name__ == "__main__":
    unittest.main()
