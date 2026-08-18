import json
import unittest
import warnings
import numpy as np
import xarray as xr
from pathlib import Path

from eoio.readers.radcalnet.metadata import ROI_DEFINITIONS, RadCalNetMetadataExtractor


class _FakeConfig:
    subset = None
    read_params: dict = {}


class _FakeReader:
    path = Path("fake_radcalnet_file.txt")
    config = _FakeConfig()
    resolved_config = None

    def list_include_vars(self):
        return []

    def list_selected_meas(self):
        return []


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
        self.assertEqual(md["product_name"], "fake_radcalnet_file.txt")

    def test_product_bounds_has_no_stray_brackets(self):
        extractor = RadCalNetMetadataExtractor(_FakeReader(), _make_ds())
        md = extractor.get_basic_metadata()

        self.assertNotIn("[", md["product_bounds"])
        self.assertNotIn("]", md["product_bounds"])

    def test_product_bounds_is_valid_wkt_point(self):
        """product_bounds used to smuggle the site's ROI radius into a non-WKT string
        (e.g. "500 (15.1 -23.6)"), unlike every other reader (which uses real WKT/geometry
        for this field). It should now be a plain WKT POINT, matching hypernets -- the
        closest sibling in-situ reader."""
        extractor = RadCalNetMetadataExtractor(_FakeReader(), _make_ds())
        md = extractor.get_basic_metadata()

        self.assertEqual(md["product_bounds"], "POINT (15.119215 -23.59999)")

    def test_spatial_resolution_is_numeric_and_site_specific(self):
        """spatial_resolution used to be hardcoded to the literal string "30 m disk"
        regardless of site, even though each RadCalNet site has its own real ROI size in
        ROI_DEFINITIONS (e.g. GONA=500m) -- so the value was both the wrong type (a
        free-text string, unlike every other reader's numeric + *_units convention) and
        factually wrong for every site except one that happens to be 30m."""
        extractor = RadCalNetMetadataExtractor(_FakeReader(), _make_ds(site="GONA01"))
        md = extractor.get_basic_metadata()

        self.assertEqual(md["spatial_resolution"], ROI_DEFINITIONS["GONA"])
        self.assertEqual(md["spatial_resolution_units"], "m")

    def test_eoio_subset_is_empty_string_when_no_subset(self):
        extractor = RadCalNetMetadataExtractor(_FakeReader(), _make_ds())
        md = extractor.get_basic_metadata()

        # "" (not None) -- an attr value of None can't be written to netCDF.
        self.assertEqual(md["eoio:subset"], "")

    def test_eoio_subset_is_valid_json_when_subset_present(self):
        """Regression test: eoio:subset used to be repr(self.subset), Python-only
        syntax (e.g. single quotes) that isn't parseable JSON."""

        class _ConfigWithSubset(_FakeConfig):
            subset = {"roi": [0.0, 1.0, 2.0, 3.0], "roi_crs_epsg": "EPSG:4326"}

        class _ReaderWithSubset(_FakeReader):
            config = _ConfigWithSubset()

        extractor = RadCalNetMetadataExtractor(_ReaderWithSubset(), _make_ds())
        md = extractor.get_basic_metadata()

        parsed = json.loads(md["eoio:subset"])
        self.assertEqual(parsed["roi_crs_epsg"], "EPSG:4326")


class testGetVariableProductMetadata(unittest.TestCase):
    """Regression test: get_variable_product_metadata() used to be named
    get_variable_metadata(), which doesn't match BaseMetadataExtractor's expected override
    point -- so it was silently never called, and any variable attrs not covered by the
    min/optional basic-key lists were dropped rather than ending up under the variable's
    product_metadata."""

    def _make_reflectance_ds(self):
        return xr.Dataset(
            {
                "reflectance": (
                    ["wavelength", "time"],
                    np.zeros((2, 1)),
                    {"long_name": "refl", "standard_name": "refl", "units": "1", "extra_key": "extra_value"},
                )
            },
            coords={"time": [np.datetime64("2021-10-16T08:00:00")], "wavelength": [400.0, 410.0]},
            attrs={
                "Site": "GONA01",
                "Lattitude": -23.59999,
                "Longitude": 15.119215,
                "Altitude": 510.0,
                "collection": "RadCalNet",
            },
        )

    def test_variable_product_metadata_reaches_attach_metadata_output(self):
        ds = self._make_reflectance_ds()

        class _ReflectanceReader(_FakeReader):
            def list_include_vars(self):
                return ["reflectance"]

        extractor = RadCalNetMetadataExtractor(_ReflectanceReader(), ds)
        cleared = extractor.clear_metadata(ds.copy(deep=False))
        out = extractor.attach_metadata(cleared, level="all")

        self.assertEqual(out["reflectance"].attrs.get("product_metadata"), {"extra_key": "extra_value"})


class testGetVariableBasicMetadata(unittest.TestCase):
    """Regression test: a missing expected basic-metadata key used to construct a Warning
    object without raising/emitting it (`Warning(...)` instead of `warnings.warn(...)`), so
    the warning silently never surfaced."""

    def test_missing_key_actually_warns(self):
        ds = xr.Dataset({"x": (["t"], [1.0])})
        extractor = RadCalNetMetadataExtractor(_FakeReader(), ds)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            extractor.get_variable_basic_metadata("x")

        self.assertTrue(any("missing expected metadata key" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
