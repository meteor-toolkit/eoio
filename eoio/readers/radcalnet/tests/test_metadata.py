import unittest
import numpy as np
import xarray as xr

from eoio.readers.radcalnet.metadata import RadCalNetMetadataExtractor


class _FakeConfig:
    subset = None


class _FakeReader:
    path = "fake_radcalnet_file.txt"
    config = _FakeConfig()


def _make_ds(site="GONA01", lat=-23.59999, lon=15.119215, alt=510.0):
    return xr.Dataset(
        coords={"time": [np.datetime64("2021-10-16T08:00:00")]},
        attrs={
            "Site": site,
            "Lattitude": lat,
            "Longitude": lon,
            "Altitude": alt,
            "collection": "RadCalNet",
        },
    )


class testGetBasicMetadata(unittest.TestCase):
    """Regression test: get_basic_metadata()'s WKT footprint used to be built
    from list-wrapped lat/lon (e.g. [15.119215]), producing invalid WKT like
    "POINT ([15.119215] [-23.59999])" that only surfaced as a downstream
    parse failure far from the actual bug (see data_io.py's read_file(),
    which now unwraps Site/Lat/Lon/Alt to scalars at the source)."""

    def test_footprint_geometry_parses_to_correct_point(self):
        """footprint['geometry'] used to be None -- normalize_footprint's
        wkt_to_geometry() silently swallows the ParseException from the
        malformed, list-wrapped WKT and returns None, so the previously
        broken behaviour was a silently empty footprint, not a raised error."""
        extractor = RadCalNetMetadataExtractor(_FakeReader(), _make_ds())
        md = extractor.get_basic_metadata()

        geometry = md["footprint"]["geometry"]
        self.assertIsNotNone(geometry)
        self.assertAlmostEqual(geometry.x, 15.119215)
        self.assertAlmostEqual(geometry.y, -23.59999)

    def test_name_and_collection_name_use_site_prefix(self):
        extractor = RadCalNetMetadataExtractor(_FakeReader(), _make_ds())
        md = extractor.get_basic_metadata()

        self.assertEqual(md["name"], "GONA01")
        self.assertEqual(md["collection_name"], "GONA RadCalNet")
        self.assertEqual(md["product_name"], "GONA_RadCalNet_L1")

    def test_product_bounds_has_no_stray_brackets(self):
        extractor = RadCalNetMetadataExtractor(_FakeReader(), _make_ds())
        md = extractor.get_basic_metadata()

        self.assertNotIn("[", md["product_bounds"])
        self.assertNotIn("]", md["product_bounds"])


if __name__ == "__main__":
    unittest.main()
