import unittest
from unittest.mock import MagicMock, patch

import xarray as xr

# Update this import to match your package structure
from eoio.readers.metadata import BaseMetadataExtractor


class _DummyExtractor(BaseMetadataExtractor):
    """
    Concrete extractor for testing BaseMetadataExtractor behaviour.
    """

    def __init__(self, reader):
        super().__init__(reader)
        self.calls = {
            "basic": 0,
            "product": 0,
            "var_product": {},
            "var_basic": {},
        }
        # Mapping of band names to spatial resolutions and geometry IDs for testing
        self.band_resolutions = {
            "B02": {"spatial_resolution": 10, "geometry_id": "10m"},
            "B03": {"spatial_resolution": 10, "geometry_id": "10m"},
            "B04": {"spatial_resolution": 10, "geometry_id": "10m"},
            "B05": {"spatial_resolution": 20, "geometry_id": "20m"},
            "B11": {"spatial_resolution": 20, "geometry_id": "20m"},
        }

    def get_basic_metadata(self) -> dict:
        self.calls["basic"] += 1
        # Return all resolutions; they should be filtered by meas_list in extract_metadata
        all_resolutions = [v["spatial_resolution"] for v in self.band_resolutions.values()]
        all_geom_ids = [v["geometry_id"] for v in self.band_resolutions.values()]
        return {
            "collection_name": "TEST",
            "eoio:reader": "dummy",
            "spatial_resolution": all_resolutions,
            "geometry_ids": all_geom_ids,
        }

    def get_product_metadata(self) -> dict:
        self.calls["product"] += 1
        return {"orbit": 137}

    def get_variable_product_metadata(self, var: str) -> dict:
        self.calls["var_product"][var] = self.calls["var_product"].get(var, 0) + 1
        # include "unc_comps" to exercise the removal logic
        result = {"a": 1, "unc_comps": ["u1", "u2"]}
        # Add spatial metadata for bands (if available)
        if var in self.band_resolutions:
            result.update(self.band_resolutions[var])
        return result

    def get_variable_basic_metadata(self, var: str) -> dict:
        self.calls["var_basic"][var] = self.calls["var_basic"].get(var, 0) + 1
        return {"units": "1", "long_name": f"var {var}"}


class TestBaseMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.reader = MagicMock()
        self.reader.path = "/tmp/TEST.SAFE"

        self.reader.list_include_vars.return_value = ["B02", "B03"]
        self.reader.list_selected_meas.return_value = ["B02", "B03"]

        self.reader.config.read_params = {"include_uncertainties": True}

        self.extractor = _DummyExtractor(self.reader)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def test_basic_metadata_is_cached(self):
        a = self.extractor.basic_metadata
        b = self.extractor.basic_metadata
        self.assertEqual(a["collection_name"], "TEST")
        self.assertEqual(a["eoio:reader"], "dummy")
        self.assertIn("spatial_resolution", a)
        self.assertIn("geometry_ids", a)
        self.assertIs(a, b)
        self.assertEqual(self.extractor.calls["basic"], 1)

    def test_product_metadata_is_cached(self):
        a = self.extractor.product_metadata
        b = self.extractor.product_metadata
        self.assertEqual(a, {"orbit": 137})
        self.assertIs(a, b)
        self.assertEqual(self.extractor.calls["product"], 1)

    def test_variable_product_metadata_is_cached_per_var(self):
        m1 = self.extractor.variable_product_metadata("B02")
        m2 = self.extractor.variable_product_metadata("B02")
        m3 = self.extractor.variable_product_metadata("B03")

        self.assertEqual(m1["a"], 1)
        self.assertEqual(m3["a"], 1)
        self.assertIs(m1, m2)

        self.assertEqual(self.extractor.calls["var_product"]["B02"], 1)
        self.assertEqual(self.extractor.calls["var_product"]["B03"], 1)

    def test_variable_basic_metadata_cached_returns_units(self):
        m1 = self.extractor.variable_basic_metadata("B02")
        m2 = self.extractor.variable_basic_metadata("B02")
        m3 = self.extractor.variable_basic_metadata("B03")

        self.assertEqual(m1["units"], "1")
        self.assertEqual(m3["units"], "1")
        self.assertIs(m1, m2)

        self.assertEqual(self.extractor.calls["var_basic"]["B02"], 1)
        self.assertEqual(self.extractor.calls["var_basic"]["B03"], 1)

    # ------------------------------------------------------------------
    # extract_metadata
    # ------------------------------------------------------------------

    def test_extract_metadata_default_level_returns_basic_only(self):
        basic_md, md, var_basic_md, var_md = self.extractor.extract_metadata(level=None)

        self.assertEqual(basic_md["collection_name"], "TEST")
        # Without meas_list, spatial lists should include all bands
        self.assertIn("spatial_resolution", basic_md)
        self.assertIn("geometry_ids", basic_md)
        self.assertEqual(md, {})
        self.assertEqual(var_basic_md, {})
        self.assertEqual(var_md, {})

    def test_extract_metadata_all_level_returns_everything(self):
        basic_md, md, var_basic_md, var_md = self.extractor.extract_metadata(level="all")

        self.assertEqual(basic_md["collection_name"], "TEST")
        self.assertIn("spatial_resolution", basic_md)
        self.assertIn("geometry_ids", basic_md)
        self.assertEqual(md, {"orbit": 137})

        self.assertEqual(set(var_md.keys()), {"B02", "B03"})
        self.assertEqual(set(var_basic_md.keys()), {"B02", "B03"})

        self.assertEqual(var_md["B02"]["a"], 1)
        self.assertEqual(var_basic_md["B02"]["units"], "1")

    def test_extract_metadata_all_level_clears_unc_comps_when_uncertainties_disabled(
        self,
    ):
        self.reader.config.read_params = {"include_uncertainties": False}

        basic_md, md, var_basic_md, var_md = self.extractor.extract_metadata(level="all")

        # Your current logic clears unc_comps in var_basic_md when unc_comps exists in var_md
        self.assertIn("unc_comps", var_basic_md["B02"])
        self.assertEqual(var_basic_md["B02"]["unc_comps"], [])

    def test_extract_metadata_filters_spatial_lists_by_meas_list(self):
        """Verify spatial_resolution and geometry_ids are filtered when meas_list is provided."""
        # Request only B02 and B05 (10m and 20m)
        basic_md, md, var_basic_md, var_md = self.extractor.extract_metadata(level=None, meas_list=["B02", "B05"])

        # Spatial lists should only contain values for B02 and B05
        self.assertEqual(basic_md["spatial_resolution"], [10, 20])
        self.assertEqual(basic_md["geometry_ids"], ["10m", "20m"])

    def test_extract_metadata_filters_spatial_lists_single_band(self):
        """Verify spatial filtering works with a single band."""
        basic_md, md, var_basic_md, var_md = self.extractor.extract_metadata(level=None, meas_list=["B11"])

        self.assertEqual(basic_md["spatial_resolution"], [20])
        self.assertEqual(basic_md["geometry_ids"], ["20m"])

    def test_extract_metadata_handles_empty_meas_list(self):
        """Verify empty meas_list results in empty spatial lists."""
        basic_md, md, var_basic_md, var_md = self.extractor.extract_metadata(level=None, meas_list=[])

        # Should preserve the original full lists when meas_list is empty
        self.assertIn("spatial_resolution", basic_md)
        self.assertIn("geometry_ids", basic_md)

    def test_extract_metadata_ignores_unknown_bands_in_meas_list(self):
        """Verify unknown band names are silently skipped in meas_list."""
        basic_md, md, var_basic_md, var_md = self.extractor.extract_metadata(
            level=None, meas_list=["B02", "UNKNOWN", "B04"]
        )

        # Only B02 and B04 should be included (UNKNOWN has no metadata)
        self.assertEqual(basic_md["spatial_resolution"], [10, 10])
        self.assertEqual(basic_md["geometry_ids"], ["10m", "10m"])

    def test_extract_metadata_filtering_does_not_inject_spurious_singular_key(self):
        """Regression test: the filtering trigger/output key used to be the singular
        "geometry_id" even though get_basic_metadata()'s own documented dataset-level key is
        plural "geometry_ids" -- so for any reader using the (correct, majority) plural
        convention, filtering silently left "geometry_ids" unfiltered and injected a spurious
        singular "geometry_id" key that was never part of the reader's own schema."""
        basic_md, _, _, _ = self.extractor.extract_metadata(level=None, meas_list=["B02", "B05"])

        self.assertNotIn("geometry_id", basic_md)
        self.assertEqual(basic_md["geometry_ids"], ["10m", "20m"])

    # ------------------------------------------------------------------
    # attach_metadata
    # ------------------------------------------------------------------

    @patch("eoio.readers.metadata.__version__", "9.9.9")
    def test_attach_metadata_basic_level_adds_basic_attrs_and_history(self):
        ds = xr.Dataset({"B02": xr.DataArray([1, 2, 3])})
        out = self.extractor.attach_metadata(ds, level=None)

        # returns a copy
        self.assertIsNot(ds, out)

        self.assertEqual(out.attrs["collection_name"], "TEST")
        self.assertEqual(out.attrs["eoio:reader"], "dummy")

        # should NOT add product_metadata at dataset level unless level == "all"
        self.assertNotIn("product_metadata", out.attrs)

        # always adds eoio version/path + history
        self.assertEqual(out.attrs["eoio:version"], "9.9.9")
        self.assertEqual(out.attrs["eoio:path"], "/tmp/TEST.SAFE")
        self.assertIn("history", out.attrs)

        # license/references/date_created default to "TBD"/now() when the reader doesn't
        # supply its own values (see _DummyExtractor.get_basic_metadata())
        self.assertEqual(out.attrs["license"], "TBD")
        self.assertEqual(out.attrs["references"], "TBD")
        self.assertIn("date_created", out.attrs)

    @patch("eoio.readers.metadata.__version__", "9.9.9")
    def test_attach_metadata_history_names_file_reader_and_version(self):
        out = self.extractor.attach_metadata(xr.Dataset({"B02": xr.DataArray([1])}), level=None)

        line = out.attrs["history"].split("\n")[-1]
        self.assertTrue(line.endswith("TEST.SAFE read in using eoio version 9.9.9 (dummy reader)"), line)

    def test_attach_metadata_appends_to_an_existing_history(self):
        ds = xr.Dataset({"B02": xr.DataArray([1])}, attrs={"history": "earlier step"})
        out = self.extractor.attach_metadata(ds, level=None)

        lines = out.attrs["history"].split("\n")
        self.assertEqual(lines[0], "earlier step")
        self.assertEqual(len(lines), 2)

    @patch("eoio.readers.metadata.__version__", "9.9.9")
    def test_attach_metadata_prefers_reader_supplied_license_references_date_created(self):
        """Regression test: license/references/date_created used to be unconditionally
        overwritten by attach_metadata() even when the reader's own get_basic_metadata()
        already supplied a real value -- making any reader-level value for these three keys
        dead code. They should now only fall back to the base class's default when the
        reader didn't provide one."""

        class _ExtractorWithLicense(_DummyExtractor):
            def get_basic_metadata(self) -> dict:
                md = super().get_basic_metadata()
                md.update(
                    {
                        "license": "CC-BY-4.0",
                        "references": "https://example.org/doi/123",
                        "date_created": "2020-01-01T00:00:00+00:00",
                    }
                )
                return md

        extractor = _ExtractorWithLicense(self.reader)
        ds = xr.Dataset({"B02": xr.DataArray([1, 2, 3])})
        out = extractor.attach_metadata(ds, level=None)

        self.assertEqual(out.attrs["license"], "CC-BY-4.0")
        self.assertEqual(out.attrs["references"], "https://example.org/doi/123")
        self.assertEqual(out.attrs["date_created"], "2020-01-01T00:00:00+00:00")

    @patch("eoio.readers.metadata.__version__", "9.9.9")
    def test_attach_metadata_all_level_adds_product_metadata_and_variable_metadata(
        self,
    ):
        ds = xr.Dataset(
            data_vars={
                # Use explicit dims so xarray doesn't try to align conflicting sizes on 'dim_0'
                "B02": xr.DataArray([1, 2, 3], dims=("x",), attrs={"existing": "x"}),
                "B03": xr.DataArray([4, 5, 6], dims=("x",)),
                "OTHER": xr.DataArray([7], dims=("y",)),  # different dim, won't collide
            }
        )

        out = self.extractor.attach_metadata(ds, level="all")

        # dataset-level product metadata
        self.assertIn("product_metadata", out.attrs)
        self.assertEqual(out.attrs["product_metadata"], {"orbit": 137})

        # variable-level metadata should be attached only for include vars that are present
        self.assertIn("product_metadata", out["B02"].attrs)
        self.assertEqual(out["B02"].attrs["product_metadata"]["a"], 1)
        self.assertEqual(out["B02"].attrs["units"], "1")

        # preserves existing attrs
        self.assertEqual(out["B02"].attrs["existing"], "x")

        self.assertIn("product_metadata", out["B03"].attrs)
        self.assertEqual(out["B03"].attrs["product_metadata"]["a"], 1)
        self.assertEqual(out["B03"].attrs["units"], "1")

        # not included => unchanged
        self.assertNotIn("product_metadata", out["OTHER"].attrs)

    @patch("eoio.readers.metadata.__version__", "9.9.9")
    def test_attach_metadata_filters_spatial_lists_to_bands_in_dataset(self):
        """Verify attach_metadata filters spatial_resolution/geometry_ids to bands present in ds."""
        # Set up reader to report B02, B03, B04 as requested/resolved
        self.reader.resolved_config = MagicMock()
        self.reader.resolved_config.vars_sel = {"meas": ["B02", "B03", "B04"]}

        # But dataset only contains B02 and B04
        ds = xr.Dataset(
            {
                "B02": xr.DataArray([1, 2, 3], dims=("x",)),
                "B04": xr.DataArray([4, 5, 6], dims=("x",)),
            }
        )

        out = self.extractor.attach_metadata(ds, level=None)

        # Spatial lists should only include B02 and B04 (both 10m)
        self.assertEqual(out.attrs["spatial_resolution"], [10, 10])
        self.assertEqual(out.attrs["geometry_ids"], ["10m", "10m"])

    @patch("eoio.readers.metadata.__version__", "9.9.9")
    def test_attach_metadata_all_level_filters_spatial_with_partial_bands(self):
        """Verify spatial filtering works with level='all' and partial band dataset."""
        self.reader.resolved_config = MagicMock()
        self.reader.resolved_config.vars_sel = {"meas": ["B02", "B05", "B11"]}

        # Dataset contains B02 (10m) and B05 (20m), missing B11
        ds = xr.Dataset(
            {
                "B02": xr.DataArray([1], dims=("x",)),
                "B05": xr.DataArray([2], dims=("x",)),
            }
        )

        out = self.extractor.attach_metadata(ds, level="all")

        # Spatial lists should only include B02 and B05
        self.assertEqual(out.attrs["spatial_resolution"], [10, 20])
        self.assertEqual(out.attrs["geometry_ids"], ["10m", "20m"])

    # ------------------------------------------------------------------
    # clear_metadata
    # ------------------------------------------------------------------

    def test_clear_metadata_wipes_dataset_and_variable_attrs_inplace(self):
        ds = xr.Dataset(
            {"B02": xr.DataArray([1], attrs={"a": "b"})},
            attrs={"x": "y"},
        )
        out = self.extractor.clear_metadata(ds)

        # same object returned (in-place)
        self.assertIs(ds, out)

        self.assertEqual(out.attrs, {})
        self.assertEqual(out["B02"].attrs, {})

    def test_clear_metadata_leaves_coordinate_attrs_untouched(self):
        """Coordinate attrs (e.g. a wavelength coordinate's units) must survive
        clear_metadata(): attach_metadata() only ever restores attrs for data variables
        (via self.reader.list_include_vars(), which coordinates are never part of), so
        wiping them here would discard them permanently rather than clearing-then-
        reattaching them."""
        ds = xr.Dataset(
            {"B02": xr.DataArray([1], dims=("wavelength",), attrs={"a": "b"})},
            coords={"wavelength": xr.DataArray([400], dims=("wavelength",), attrs={"units": "nm"})},
        )
        out = self.extractor.clear_metadata(ds)

        self.assertEqual(out["B02"].attrs, {})
        self.assertEqual(out["wavelength"].attrs, {"units": "nm"})


if __name__ == "__main__":
    unittest.main()
