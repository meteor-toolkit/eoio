"""Tests for eoio.processors.datatree.processor"""

import numpy as np
import pytest
import xarray as xr

from eoio.processors.datatree.processor import (
    ToDataTree,
    _detect_grid,
    _grid_suffix_from_dims,
    _rename_grid_dims,
    _rename_grid_vars,
    _rescope_per_band_attrs,
    _restore_grid_vars,
    build_datatree,
    from_datatree,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _da(dims, grid_id=None, spatial_resolution=None, extra_attrs=None):
    shape = tuple(4 for _ in dims)
    attrs = {}
    if grid_id is not None or spatial_resolution is not None:
        pm = {}
        if grid_id is not None:
            pm["geometry_id"] = grid_id
        if spatial_resolution is not None:
            pm["spatial_resolution"] = spatial_resolution
        attrs["product_metadata"] = pm
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.DataArray(np.ones(shape, dtype="float32"), dims=dims, attrs=attrs)


@pytest.fixture()
def s2_like_ds_whole_product_spatial_resolution():
    """Same shape as ``s2_like_ds``, but also carries the two real-world symptoms of a
    bug this module guards against: a dataset-level ``spatial_resolution``/
    ``geometry_ids`` attr set once for the *whole, still-flat* product -- one entry per
    originally-read band (B01's 60 m included), in read order, covering every eventual
    grid -- exactly as ``eoio.readers.metadata.extract_metadata`` builds it, before any
    split by grid. Each variable also carries its own correct per-variable
    ``product_metadata.spatial_resolution``, mirroring ``geometry_id``."""
    ds = xr.Dataset()
    ds["B02"] = _da(("y_10m", "x_10m"), grid_id="10m", spatial_resolution=10)
    ds["B03"] = _da(("y_10m", "x_10m"), grid_id="10m", spatial_resolution=10)
    ds["B05"] = _da(("y_20m", "x_20m"), grid_id="20m", spatial_resolution=20)
    ds["B06"] = _da(("y_20m", "x_20m"), grid_id="20m", spatial_resolution=20)
    ds["B01"] = _da(("y_60m", "x_60m"), grid_id="60m", spatial_resolution=60)
    ds.attrs["spatial_resolution"] = [60, 10, 10, 20, 20]
    ds.attrs["geometry_ids"] = ["60m", "10m", "10m", "20m", "20m"]
    return ds


@pytest.fixture()
def s2_like_ds():
    """Minimal multi-grid dataset resembling post-stack Sentinel-2 output."""
    ds = xr.Dataset()
    ds["B02"] = _da(("y_10m", "x_10m"), grid_id="10m")
    ds["B03"] = _da(("y_10m", "x_10m"), grid_id="10m")
    ds["B05"] = _da(("y_20m", "x_20m"), grid_id="20m")
    ds["B06"] = _da(("y_20m", "x_20m"), grid_id="20m")
    ds["B01"] = _da(("y_60m", "x_60m"), grid_id="60m")
    ds["SCL"] = _da(("y_20m", "x_20m"), grid_id="20m")
    ds.attrs["title"] = "S2MSI1C"
    return ds


@pytest.fixture()
def dim_only_ds():
    """Dataset where grid is detectable only from dimension names."""
    ds = xr.Dataset()
    ds["B02"] = xr.DataArray(np.ones((4, 4), dtype="float32"), dims=("y_10m", "x_10m"))
    ds["B05"] = xr.DataArray(np.ones((2, 2), dtype="float32"), dims=("y_20m", "x_20m"))
    ds["solar_zenith_angle"] = xr.DataArray(np.ones((2, 2), dtype="float32"), dims=("y_5000m", "x_5000m"))
    return ds


@pytest.fixture()
def mixed_ds():
    """Dataset with gridded and non-gridded (root) variables."""
    ds = xr.Dataset()
    ds["B02"] = _da(("y_10m", "x_10m"), grid_id="10m")
    ds["scalar"] = xr.DataArray(42.0)  # no spatial dims → root
    return ds


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestGridSuffixFromDims:
    def test_resolution_dims(self):
        assert _grid_suffix_from_dims(("y_10m", "x_10m")) == "10m"
        assert _grid_suffix_from_dims(("y_5000m", "x_5000m")) == "5000m"

    def test_generic_dims_returns_none(self):
        assert _grid_suffix_from_dims(("y", "x")) is None

    def test_no_dims_returns_none(self):
        assert _grid_suffix_from_dims(()) is None


class TestDetectGrid:
    def test_product_metadata_geometry_id(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("y_10m", "x_10m"),
            attrs={"product_metadata": {"geometry_id": "10m"}},
        )
        assert _detect_grid(da) == "10m"

    def test_dim_name_fallback(self):
        da = xr.DataArray(np.ones((2, 2)), dims=("y_20m", "x_20m"))
        assert _detect_grid(da) == "20m"

    def test_explicit_grid_attr(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("y", "x"),
            attrs={"my_grid": "5000m"},
        )
        assert _detect_grid(da, grid_attr="my_grid") == "5000m"

    def test_explicit_nested_grid_attr(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("y", "x"),
            attrs={"meta": {"grid": "300m"}},
        )
        assert _detect_grid(da, grid_attr="meta.grid") == "300m"

    def test_no_grid_returns_empty_string(self):
        da = xr.DataArray(42.0)
        assert _detect_grid(da) == ""

    def test_grid_attr_takes_priority_over_product_metadata(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("y", "x"),
            attrs={"my_grid": "100m", "product_metadata": {"geometry_id": "10m"}},
        )
        assert _detect_grid(da, grid_attr="my_grid") == "100m"


class TestRenameGridVars:
    def test_strips_suffix_from_data_vars(self):
        ds = xr.Dataset(
            {
                "reflectance_10m": xr.DataArray(np.ones((2, 2)), dims=("y", "x")),
                "B02": xr.DataArray(np.ones((2, 2)), dims=("y", "x")),
            }
        )
        result = _rename_grid_vars(ds, "10m")
        assert "reflectance" in result.data_vars
        assert "B02" in result.data_vars
        assert "reflectance_10m" not in result.data_vars

    def test_strips_suffix_from_coordinates(self):
        ds = xr.Dataset({"a": xr.DataArray(np.ones(3), dims=("band_10m",))})
        ds = ds.assign_coords(band_10m=["B02", "B03", "B04"])
        result = _rename_grid_vars(ds, "10m")
        assert "band" in result.coords
        assert "band_10m" not in result.coords

    def test_does_not_rename_spatial_dims(self):
        ds = xr.Dataset({"a": xr.DataArray(np.ones((2, 2)), dims=("y", "x"))})
        result = _rename_grid_vars(ds, "10m")
        assert "x" in result.dims
        assert "y" in result.dims

    def test_no_match_returns_unchanged(self):
        ds = xr.Dataset({"B02": xr.DataArray(np.ones((2, 2)), dims=("y", "x"))})
        result = _rename_grid_vars(ds, "10m")
        assert "B02" in result.data_vars


class TestRestoreGridVars:
    def test_appends_suffix_to_data_vars(self):
        ds = xr.Dataset(
            {
                "reflectance": xr.DataArray(np.ones((2, 2)), dims=("x_10m", "y_10m")),
                "B02": xr.DataArray(np.ones((2, 2)), dims=("x_10m", "y_10m")),
            }
        )
        result = _restore_grid_vars(ds, "10m")
        assert "reflectance_10m" in result.data_vars
        assert "B02_10m" in result.data_vars

    def test_does_not_double_suffix(self):
        ds = xr.Dataset({"reflectance_10m": xr.DataArray(np.ones(2), dims=("x",))})
        result = _restore_grid_vars(ds, "10m")
        assert "reflectance_10m" in result.data_vars
        assert "reflectance_10m_10m" not in result.data_vars

    def test_skips_spatial_dims(self):
        ds = xr.Dataset({"a": xr.DataArray(np.ones((2, 2)), dims=("y", "x"))})
        result = _restore_grid_vars(ds, "10m")
        assert "x" in result.dims
        assert "x_10m" not in result.dims


class TestRenameGridDims:
    def test_renames_suffixed_dims(self):
        ds = xr.Dataset({"a": xr.DataArray(np.ones((2, 2)), dims=("y_10m", "x_10m"))})
        result = _rename_grid_dims(ds, "10m")
        assert "x" in result.dims
        assert "y" in result.dims
        assert "x_10m" not in result.dims

    def test_no_match_returns_unchanged(self):
        ds = xr.Dataset({"a": xr.DataArray(np.ones((2, 2)), dims=("y_20m", "x_20m"))})
        result = _rename_grid_dims(ds, "10m")
        assert "x_20m" in result.dims

    def test_partial_match(self):
        ds = xr.Dataset(
            {
                "a": xr.DataArray(np.ones((2, 2)), dims=("y_10m", "x_10m")),
                "b": xr.DataArray(np.ones(3), dims=("band_10m",)),
            }
        )
        result = _rename_grid_dims(ds, "10m")
        assert "x" in result.dims
        assert "y" in result.dims
        assert "band_10m" in result.dims  # non-spatial dim left unchanged


# ---------------------------------------------------------------------------
# Integration tests — build_datatree
# ---------------------------------------------------------------------------


class TestBuildDataTree:
    def test_nodes_created_per_grid(self, s2_like_ds):
        dt = build_datatree(s2_like_ds)
        node_names = {path for path, _ in dt.subtree_with_keys if path}
        assert "10m" in node_names
        assert "20m" in node_names
        assert "60m" in node_names

    def test_variables_in_correct_nodes(self, s2_like_ds):
        dt = build_datatree(s2_like_ds)
        assert "B02" in dt["10m"].dataset.data_vars
        assert "B03" in dt["10m"].dataset.data_vars
        assert "B05" in dt["20m"].dataset.data_vars
        assert "B01" in dt["60m"].dataset.data_vars

    def test_dims_renamed_in_nodes(self, s2_like_ds):
        dt = build_datatree(s2_like_ds)
        assert "x" in dt["10m"].dataset.dims
        assert "y" in dt["10m"].dataset.dims
        assert "x_10m" not in dt["10m"].dataset.dims

    def test_rename_dims_false_preserves_original_names(self, s2_like_ds):
        dt = build_datatree(s2_like_ds, rename_dims=False)
        assert "x_10m" in dt["10m"].dataset.dims

    def test_root_attrs_preserved(self, s2_like_ds):
        dt = build_datatree(s2_like_ds)
        assert dt.root.dataset.attrs["title"] == "S2MSI1C"

    def test_non_gridded_variable_goes_to_root(self, mixed_ds):
        dt = build_datatree(mixed_ds)
        assert "scalar" in dt.root.dataset.data_vars
        assert "B02" not in dt.root.dataset.data_vars

    def test_dim_name_detection_when_no_attr(self, dim_only_ds):
        dt = build_datatree(dim_only_ds)
        node_names = {path for path, _ in dt.subtree_with_keys if path}
        assert "10m" in node_names
        assert "20m" in node_names
        assert "5000m" in node_names

    def test_custom_grid_attr(self):
        ds = xr.Dataset()
        ds["A"] = xr.DataArray(np.ones((2, 2)), dims=("y_a", "x_a"), attrs={"my_grid": "alpha"})
        ds["B"] = xr.DataArray(np.ones((4, 4)), dims=("y_b", "x_b"), attrs={"my_grid": "beta"})
        dt = build_datatree(ds, grid_attr="my_grid", rename_dims=False)
        node_names = {path for path, _ in dt.subtree_with_keys if path}
        assert "alpha" in node_names
        assert "beta" in node_names

    def test_rename_vars_strips_grid_suffix(self):
        ds = xr.Dataset()
        ds["reflectance_10m"] = xr.DataArray(
            np.ones((4, 4)), dims=("y_10m", "x_10m"), attrs={"product_metadata": {"geometry_id": "10m"}}
        )
        ds["B02"] = xr.DataArray(
            np.ones((4, 4)), dims=("y_10m", "x_10m"), attrs={"product_metadata": {"geometry_id": "10m"}}
        )
        dt = build_datatree(ds, rename_vars=True)
        assert "reflectance" in dt["10m"].dataset.data_vars
        assert "B02" in dt["10m"].dataset.data_vars
        assert "reflectance_10m" not in dt["10m"].dataset.data_vars

    def test_rename_vars_false_preserves_names(self):
        ds = xr.Dataset()
        ds["reflectance_10m"] = xr.DataArray(
            np.ones((4, 4)), dims=("y_10m", "x_10m"), attrs={"product_metadata": {"geometry_id": "10m"}}
        )
        dt = build_datatree(ds, rename_vars=False)
        assert "reflectance_10m" in dt["10m"].dataset.data_vars

    def test_variable_attrs_preserved_for_ungridded_data(self):
        """Point/time-series data (e.g. RadCalNet, Hypernets) has no x_/y_ spatial dims, so
        every variable lands at the root node untouched by grid grouping -- confirms
        per-variable attrs (long_name/standard_name/units) survive that path too, not just
        the gridded/renamed path covered by the other tests here."""
        ds = xr.Dataset(
            {
                "air_pressure": xr.DataArray(
                    [963.15],
                    dims=("time",),
                    attrs={"long_name": "Surface air pressure", "standard_name": "air_pressure", "units": "hPa"},
                )
            },
            coords={"time": [0]},
        )
        dt = build_datatree(ds)
        assert dt.root.dataset["air_pressure"].attrs == {
            "long_name": "Surface air pressure",
            "standard_name": "air_pressure",
            "units": "hPa",
        }

    def test_each_node_has_single_grid(self, s2_like_ds):
        dt = build_datatree(s2_like_ds)
        node_10m = dt["10m"].dataset
        for var in node_10m.data_vars.values():
            assert "x_20m" not in var.dims
            assert "x_60m" not in var.dims


# ---------------------------------------------------------------------------
# Unit tests — _rescope_per_band_attrs / build_datatree's use of it
# ---------------------------------------------------------------------------


class TestRescopePerBandAttrs:
    """Regression coverage for a real failure: ``spatial_resolution``/``geometry_ids``
    are set once, whole-product, as one list entry per originally-read band -- before
    ``build_datatree`` splits by grid. Left alone, every split node inherited the same
    whole-product list regardless of which bands it actually held (e.g. a real
    Sentinel-2 "10m" node came through with ``spatial_resolution=[60, 10, 10, 10, 20,
    20, 20]``, B01's 60 m included even though B01 lives on a different node)."""

    def test_each_node_gets_its_own_scalar_spatial_resolution(self, s2_like_ds_whole_product_spatial_resolution):
        dt = build_datatree(s2_like_ds_whole_product_spatial_resolution)
        assert dt["10m"].dataset.attrs["spatial_resolution"] == 10
        assert dt["20m"].dataset.attrs["spatial_resolution"] == 20
        assert dt["60m"].dataset.attrs["spatial_resolution"] == 60

    def test_each_node_gets_its_own_geometry_ids(self, s2_like_ds_whole_product_spatial_resolution):
        dt = build_datatree(s2_like_ds_whole_product_spatial_resolution)
        assert dt["10m"].dataset.attrs["geometry_ids"] == "10m"
        assert dt["20m"].dataset.attrs["geometry_ids"] == "20m"
        assert dt["60m"].dataset.attrs["geometry_ids"] == "60m"

    def test_mutating_one_nodes_attrs_does_not_leak_to_another(self, s2_like_ds_whole_product_spatial_resolution):
        dt = build_datatree(s2_like_ds_whole_product_spatial_resolution)
        dt["10m"].dataset.attrs["spatial_resolution"] = 999
        assert dt["20m"].dataset.attrs["spatial_resolution"] == 20
        assert dt["60m"].dataset.attrs["spatial_resolution"] == 60

    def test_root_attrs_left_as_the_whole_product_summary(self, s2_like_ds_whole_product_spatial_resolution):
        """Root holds no single grid, so its whole-product summary is legitimate as-is --
        only per-grid child nodes are rescoped."""
        dt = build_datatree(s2_like_ds_whole_product_spatial_resolution)
        assert dt.root.dataset.attrs["spatial_resolution"] == [60, 10, 10, 20, 20]
        assert dt.root.dataset.attrs["geometry_ids"] == ["60m", "10m", "10m", "20m", "20m"]

    def test_drops_stale_attr_when_no_per_variable_source(self):
        """A node whose own variables carry no per-variable spatial_resolution/
        geometry_id at all has the inherited whole-product value dropped, not kept --
        a missing attr is honest, a wrong one is not."""
        ds = xr.Dataset()
        ds["B02"] = xr.DataArray(np.ones((4, 4), dtype="float32"), dims=("y_10m", "x_10m"))
        ds.attrs["spatial_resolution"] = [60, 10, 10, 20, 20]
        dt = build_datatree(ds)
        assert "spatial_resolution" not in dt["10m"].dataset.attrs

    def test_noop_when_attr_not_present(self, s2_like_ds):
        """s2_like_ds carries no dataset-level spatial_resolution/geometry_ids at all --
        confirms the common case (nothing to rescope) is a clean no-op."""
        dt = build_datatree(s2_like_ds)
        assert "spatial_resolution" not in dt["10m"].dataset.attrs

    def test_helper_called_directly(self):
        ds = xr.Dataset()
        ds["B02"] = _da(("y_10m", "x_10m"), grid_id="10m", spatial_resolution=10)
        node_ds = ds[["B02"]]
        node_ds.attrs = dict(spatial_resolution=[60, 10])
        out = _rescope_per_band_attrs(ds, node_ds, ["B02"])
        assert out.attrs["spatial_resolution"] == 10


# ---------------------------------------------------------------------------
# Integration tests — from_datatree (round-trip)
# ---------------------------------------------------------------------------


class TestFromDataTree:
    def test_roundtrip_restores_variables(self, s2_like_ds):
        # rename_vars=True (default): variables without a grid suffix are unchanged
        # so all original names survive the round-trip
        dt = build_datatree(s2_like_ds)
        flat = from_datatree(dt)
        for name in s2_like_ds.data_vars:
            assert name in flat  # B02, B03, SCL etc. have no suffix → unchanged

    def test_roundtrip_restores_suffixed_dims(self, s2_like_ds):
        dt = build_datatree(s2_like_ds)
        flat = from_datatree(dt)
        assert "x_10m" in flat.dims
        assert "x_20m" in flat.dims
        assert "x_60m" in flat.dims

    def test_roundtrip_root_attrs_preserved(self, s2_like_ds):
        dt = build_datatree(s2_like_ds)
        flat = from_datatree(dt)
        assert flat.attrs["title"] == "S2MSI1C"

    def test_var_rename_is_one_way(self):
        # rename_vars strips the suffix in the node; from_datatree does not restore it
        ds = xr.Dataset()
        ds["reflectance_10m"] = xr.DataArray(
            np.ones((4, 4)), dims=("y_10m", "x_10m"), attrs={"product_metadata": {"geometry_id": "10m"}}
        )
        dt = build_datatree(ds, rename_vars=True)
        assert "reflectance" in dt["10m"].dataset.data_vars
        flat = from_datatree(dt)
        # stays as "reflectance" — variable renaming is not reversed
        assert "reflectance" in flat.data_vars
        assert "reflectance_10m" not in flat.data_vars

    def test_roundtrip_no_rename_is_identity(self, s2_like_ds):
        dt = build_datatree(s2_like_ds, rename_dims=False, rename_vars=False)
        flat = from_datatree(dt)
        for name in s2_like_ds.data_vars:
            assert name in flat
        assert "x_10m" in flat.dims


# ---------------------------------------------------------------------------
# Integration tests — ToDataTree processor
# ---------------------------------------------------------------------------


class TestToDataTreeProcessor:
    def test_returns_datatree(self, s2_like_ds):
        result = ToDataTree().run(s2_like_ds)
        assert isinstance(result, xr.DataTree)

    def test_params_forwarded(self, s2_like_ds):
        result = ToDataTree(params={"rename_dims": False}).run(s2_like_ds)
        assert "x_10m" in result["10m"].dataset.dims

    def test_registered_in_registry(self):
        from eoio.processors.registry import PROCESSOR_REGISTRY

        assert "to_datatree" in PROCESSOR_REGISTRY
