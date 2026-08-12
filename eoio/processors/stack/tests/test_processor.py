"""Tests for eoio.processors.stack.processor"""

import json

import numpy as np
import pytest
import xarray as xr

from eoio.processors.stack.processor import (
    StackCubes,
    _collect_per_band_ancillaries,
    _common_segment_prefix,
    _get_coord_units,
    _get_nested_attr,
    _grid_suffix_from_dims,
    _is_scalar_attr,
    _obs_concat,
    _output_var_name,
    _sort_group,
    _stack_dim_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _da(dims, measurand=None, units="1", extra_attrs=None):
    shape = tuple(4 for _ in dims)
    attrs = {}
    if measurand is not None:
        attrs["measurand"] = measurand
    attrs["units"] = units
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.DataArray(np.ones(shape, dtype="float32"), dims=dims, attrs=attrs)


@pytest.fixture()
def s2_like_ds():
    """Minimal dataset resembling Sentinel-2 reader output."""
    ds = xr.Dataset()
    for name in ["B02", "B03", "B04", "B08"]:
        ds[name] = _da(("y_10m", "x_10m"), measurand="toa_reflectance", extra_attrs={"spatial_resolution": "10m"})
    for name in ["B05", "B06", "B07"]:
        ds[name] = _da(("y_20m", "x_20m"), measurand="toa_reflectance", extra_attrs={"spatial_resolution": "20m"})
    for name in ["B01", "B09"]:
        ds[name] = _da(("y_60m", "x_60m"), measurand="toa_reflectance", extra_attrs={"spatial_resolution": "60m"})
    ds["SCL"] = _da(("y_20m", "x_20m"))
    return ds


@pytest.fixture()
def mixed_measurand_ds():
    ds = xr.Dataset()
    ds["B02"] = _da(("y_10m", "x_10m"), measurand="toa_reflectance")
    ds["B03"] = _da(("y_10m", "x_10m"), measurand="toa_reflectance")
    ds["SZA"] = _da(("y_10m", "x_10m"), measurand="solar_zenith_angle", units="degrees")
    ds["VZA"] = _da(("y_10m", "x_10m"), measurand="solar_zenith_angle", units="degrees")
    return ds


@pytest.fixture()
def ancillary_ds():
    """Dataset with per-band viewing angles as ancillary variables."""
    ds = xr.Dataset()
    wavelengths = {"B02": 490.0, "B03": 560.0, "B04": 665.0}
    for name, wl in wavelengths.items():
        anc_name = f"viewing_zenith_angle_{name}"
        ds[name] = _da(
            ("y_10m", "x_10m"),
            measurand="toa_reflectance",
            extra_attrs={
                "central_wavelength": wl,
                "ancillary_variables": f"solar_zenith_angle {anc_name}",
            },
        )
        ds[anc_name] = _da(
            ("y_10m", "x_10m"),
            measurand="angle",
            units="degrees",
        )
    # shared angle — appears in all bands' ancillary lists
    ds["solar_zenith_angle"] = _da(("y_10m", "x_10m"), measurand="angle", units="degrees")
    return ds


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestGridSuffixFromDims:
    def test_resolution_dims(self):
        assert _grid_suffix_from_dims(("y_10m", "x_10m")) == "10m"
        assert _grid_suffix_from_dims(("y_300m", "x_300m")) == "300m"

    def test_generic_dims_returns_none(self):
        assert _grid_suffix_from_dims(("y", "x")) is None

    def test_x_dim_suffix_preferred(self):
        assert _grid_suffix_from_dims(("time", "x_20m")) == "20m"


class TestGetNestedAttr:
    def _da(self, attrs):
        return xr.DataArray(np.ones((2, 2)), dims=("y", "x"), attrs=attrs)

    def test_flat_key(self):
        da = self._da({"central_wavelength": 490.0})
        assert _get_nested_attr(da, "central_wavelength") == 490.0

    def test_nested_key(self):
        da = self._da({"product_metadata": {"central_wavelength": 560.0}})
        assert _get_nested_attr(da, "product_metadata.central_wavelength") == 560.0

    def test_deeply_nested_key(self):
        da = self._da({"a": {"b": {"c": 42}}})
        assert _get_nested_attr(da, "a.b.c") == 42

    def test_missing_top_level_returns_none(self):
        da = self._da({})
        assert _get_nested_attr(da, "central_wavelength") is None

    def test_missing_nested_key_returns_none(self):
        da = self._da({"product_metadata": {}})
        assert _get_nested_attr(da, "product_metadata.central_wavelength") is None

    def test_intermediate_value_not_dict_returns_none(self):
        da = self._da({"product_metadata": "not_a_dict"})
        assert _get_nested_attr(da, "product_metadata.central_wavelength") is None

    def test_json_stringified_intermediate_dict_is_parsed(self):
        # A reader may have already JSON-stringified product_metadata to make it
        # netCDF-safe before this processor runs -- the lookup should still work.
        da = self._da({"product_metadata": json.dumps({"central_wavelength": 560.0})})
        assert _get_nested_attr(da, "product_metadata.central_wavelength") == 560.0

    def test_invalid_json_string_returns_none(self):
        da = self._da({"product_metadata": "not valid json {"})
        assert _get_nested_attr(da, "product_metadata.central_wavelength") is None


class TestStackDimName:
    def test_appends_grid_suffix(self):
        assert _stack_dim_name("stack_dim", ("y_10m", "x_10m")) == "stack_dim_10m"
        assert _stack_dim_name("band", ("y_300m", "x_300m")) == "band_300m"

    def test_no_suffix_when_no_grid_in_dims(self):
        assert _stack_dim_name("stack_dim", ("y", "x")) == "stack_dim"


class TestOutputVarName:
    def test_with_grid(self):
        assert _output_var_name("toa_reflectance", ("y_10m", "x_10m")) == "toa_reflectance_10m"

    def test_without_grid(self):
        assert _output_var_name("toa_reflectance", ("y", "x")) == "toa_reflectance"


class TestObsConcat:
    def test_concat_along_new_dim(self):
        a = xr.DataArray([[1, 2], [3, 4]], dims=["y", "x"]).expand_dims(dim={"band": ["B02"]})
        b = xr.DataArray([[5, 6], [7, 8]], dims=["y", "x"]).expand_dims(dim={"band": ["B03"]})
        result = _obs_concat([a, b], dim="band")
        assert result.dims == ("band", "y", "x")
        assert list(result.coords["band"].values) == ["B02", "B03"]


class TestIsScalarAttr:
    def test_numeric_types(self):
        assert _is_scalar_attr(1) is True
        assert _is_scalar_attr(1.5) is True
        assert _is_scalar_attr(True) is True

    def test_string(self):
        assert _is_scalar_attr("hello") is True

    def test_complex_types(self):
        assert _is_scalar_attr([1, 2]) is False
        assert _is_scalar_attr({"a": 1}) is False
        assert _is_scalar_attr(None) is False


class TestCommonSegmentPrefix:
    def test_strips_band_suffix(self):
        names = ["viewing_zenith_angle_B02", "viewing_zenith_angle_B03", "viewing_zenith_angle_B04"]
        assert _common_segment_prefix(names) == "viewing_zenith_angle"

    def test_no_common_prefix_returns_ancillary(self):
        assert _common_segment_prefix(["abc", "xyz"]) == "ancillary"

    def test_single_name_returns_full_name(self):
        assert _common_segment_prefix(["viewing_zenith_angle_B02"]) == "viewing_zenith_angle_B02"

    def test_partial_match(self):
        assert _common_segment_prefix(["u_radiance_Oa06", "u_radiance_Oa07"]) == "u_radiance"


# ---------------------------------------------------------------------------
# Integration tests — StackCubes processor
# ---------------------------------------------------------------------------


class TestAllOptions:
    def test_all_options_defined_as_class_attribute(self):
        assert hasattr(StackCubes, "_all_options")

    def test_all_options_covers_constructor_params(self):
        expected = {
            "stack_dim",
            "dim_coord_attr",
            "drop_originals",
            "stack_ancillaries",
            "coord_attrs",
            "measurand_attr",
            "dim_attrs",
        }
        assert expected <= set(StackCubes._all_options)


class TestStackCubes:
    def test_output_variables_created(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        assert "toa_reflectance_10m" in result
        assert "toa_reflectance_20m" in result
        assert "toa_reflectance_60m" in result

    def test_output_dims(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        assert result["toa_reflectance_10m"].dims == ("stack_dim_10m", "y_10m", "x_10m")

    def test_stack_dim_coordinate_contains_original_names(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        stack_10m = list(result["toa_reflectance_10m"].coords["stack_dim_10m"].values)
        assert stack_10m == sorted(["B02", "B03", "B04", "B08"])

    def test_variables_sorted_lexicographically(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        stack = list(result["toa_reflectance_10m"].coords["stack_dim_10m"].values)
        assert stack == sorted(stack)

    def test_original_variables_dropped_by_default(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        for name in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B01", "B09"]:
            assert name not in result

    def test_drop_originals_false_keeps_per_band_variables(self, s2_like_ds):
        result = StackCubes(params={"drop_originals": False}).run(s2_like_ds)
        assert "toa_reflectance_10m" in result
        assert "B02" in result

    def test_variable_without_measurand_left_unchanged(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        assert "SCL" in result
        assert result["SCL"].dims == ("y_20m", "x_20m")

    def test_groups_by_measurand(self, mixed_measurand_ds):
        result = StackCubes().run(mixed_measurand_ds)
        assert "toa_reflectance_10m" in result
        assert "solar_zenith_angle_10m" in result
        assert result["toa_reflectance_10m"].sizes["stack_dim_10m"] == 2
        assert result["solar_zenith_angle_10m"].sizes["stack_dim_10m"] == 2

    def test_single_variable_group_not_stacked(self):
        ds = xr.Dataset()
        ds["B02"] = _da(("y_10m", "x_10m"), measurand="toa_reflectance")
        result = StackCubes().run(ds)
        assert "toa_reflectance_10m" not in result
        assert "B02" in result

    def test_cube_attrs_contain_measurand(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        assert result["toa_reflectance_10m"].attrs["measurand"] == "toa_reflectance"

    def test_cube_attrs_shared_units_inherited(self, s2_like_ds):
        result = StackCubes().run(s2_like_ds)
        assert result["toa_reflectance_10m"].attrs["units"] == "1"

    def test_shared_long_name_preserved(self):
        ds = xr.Dataset()
        for name in ("B02", "B03", "B04"):
            ds[name] = _da(("y_10m", "x_10m"), measurand="toa_reflectance", extra_attrs={"long_name": "TOA HCRF"})
        result = StackCubes().run(ds)
        assert result["toa_reflectance_10m"].attrs["long_name"] == "TOA HCRF"

    def test_differing_long_name_falls_back_to_measurand(self):
        ds = xr.Dataset()
        for name, ln in [("B02", "Red"), ("B03", "Green"), ("B04", "Blue")]:
            ds[name] = _da(("y_10m", "x_10m"), measurand="toa_reflectance", extra_attrs={"long_name": ln})
        result = StackCubes().run(ds)
        assert result["toa_reflectance_10m"].attrs["long_name"] == "toa_reflectance"

    def test_per_band_scalar_attr_stored_as_list(self):
        ds = xr.Dataset()
        for name, wl in [("B02", 490.0), ("B03", 560.0), ("B04", 665.0)]:
            ds[name] = _da(("y_10m", "x_10m"), measurand="toa_reflectance", extra_attrs={"central_wavelength": wl})
        result = StackCubes().run(ds)
        wl_attr = result["toa_reflectance_10m"].attrs["central_wavelength"]
        assert isinstance(wl_attr, list)
        assert wl_attr == [490.0, 560.0, 665.0]

    def test_per_band_attr_promoted_to_coordinate(self):
        ds = xr.Dataset()
        for name, wl in [("B02", 490.0), ("B03", 560.0), ("B04", 665.0)]:
            ds[name] = _da(("y_10m", "x_10m"), measurand="toa_reflectance", extra_attrs={"central_wavelength": wl})
        result = StackCubes(params={"coord_attrs": ["central_wavelength"]}).run(ds)
        cube = result["toa_reflectance_10m"]
        assert "central_wavelength" in cube.coords
        assert list(cube.coords["central_wavelength"].values) == [490.0, 560.0, 665.0]
        assert "central_wavelength" not in cube.attrs

    def test_complex_per_band_attr_kept_as_list(self):
        ds = xr.Dataset()
        ds["B02"] = _da(
            ("y_10m", "x_10m"),
            measurand="toa_reflectance",
            extra_attrs={"product_metadata": {"band": "B02", "central_wavelength": 490}},
        )
        ds["B03"] = _da(
            ("y_10m", "x_10m"),
            measurand="toa_reflectance",
            extra_attrs={"product_metadata": {"band": "B03", "central_wavelength": 560}},
        )
        result = StackCubes().run(ds)
        pm = result["toa_reflectance_10m"].attrs["product_metadata"]
        assert isinstance(pm, list)
        assert pm[0]["band"] == "B02"
        assert pm[1]["band"] == "B03"

    def test_custom_stack_dim_name(self, s2_like_ds):
        result = StackCubes(params={"stack_dim": "band"}).run(s2_like_ds)
        assert "band_10m" in result["toa_reflectance_10m"].dims

    def test_custom_measurand_attr(self):
        ds = xr.Dataset()
        ds["B02"] = _da(("y_10m", "x_10m"), extra_attrs={"standard_name": "toa_reflectance"})
        ds["B03"] = _da(("y_10m", "x_10m"), extra_attrs={"standard_name": "toa_reflectance"})
        result = StackCubes(params={"measurand_attr": "standard_name"}).run(ds)
        assert "toa_reflectance_10m" in result


# ---------------------------------------------------------------------------
# Unit tests — _sort_group
# ---------------------------------------------------------------------------


class TestSortGroup:
    def _ds(self, names_wavelengths):
        ds = xr.Dataset()
        for name, wl in names_wavelengths:
            attrs = {"measurand": "toa_reflectance"}
            if wl is not None:
                attrs["central_wavelength"] = wl
            ds[name] = _da(("y_10m", "x_10m"), extra_attrs=attrs)
        return ds

    def test_default_sorts_lexicographically(self):
        ds = self._ds([("B04", 665.0), ("B02", 490.0), ("B03", 560.0)])
        names, coords = _sort_group(ds, ["B04", "B02", "B03"], dim_coord_attr=None)
        assert names == ["B02", "B03", "B04"]
        assert coords == ["B02", "B03", "B04"]

    def test_dim_coord_attr_sorts_by_value(self):
        ds = self._ds([("B04", 665.0), ("B02", 490.0), ("B03", 560.0)])
        names, coords = _sort_group(ds, ["B04", "B02", "B03"], dim_coord_attr="central_wavelength")
        assert names == ["B02", "B03", "B04"]
        assert coords == [490.0, 560.0, 665.0]

    def test_dim_coord_attr_missing_falls_back_to_variable_names(self):
        ds = self._ds([("B02", 490.0), ("B03", None)])
        names, coords = _sort_group(ds, ["B02", "B03"], dim_coord_attr="central_wavelength")
        assert names == ["B02", "B03"]
        assert coords == ["B02", "B03"]

    def test_dim_coord_attr_missing_raises_warning(self):
        ds = self._ds([("B02", 490.0), ("B03", None)])
        with pytest.warns(UserWarning, match="central_wavelength.*B03"):
            _sort_group(ds, ["B02", "B03"], dim_coord_attr="central_wavelength")


# ---------------------------------------------------------------------------
# Integration tests — dim_coord_attr
# ---------------------------------------------------------------------------


class TestCoordFromAttr:
    @pytest.fixture()
    def wavelength_ds(self):
        ds = xr.Dataset()
        wavelengths = {"B02": 490.0, "B03": 560.0, "B04": 665.0, "B08": 842.0}
        for name, wl in wavelengths.items():
            ds[name] = _da(
                ("y_10m", "x_10m"),
                measurand="toa_reflectance",
                extra_attrs={"central_wavelength": wl, "units": "nm"},
            )
        return ds

    def test_coord_values_are_wavelengths(self, wavelength_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        coord = result["toa_reflectance_10m"].coords["stack_dim_10m"].values
        assert list(coord) == [490.0, 560.0, 665.0, 842.0]

    def test_coord_sorted_ascending(self, wavelength_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        coord = list(result["toa_reflectance_10m"].coords["stack_dim_10m"].values)
        assert coord == sorted(coord)

    def test_sel_by_wavelength(self, wavelength_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        band = result["toa_reflectance_10m"].sel(stack_dim_10m=665.0)
        assert band.shape == (4, 4)

    def test_missing_attr_falls_back_to_variable_names(self, wavelength_ds):
        wavelength_ds["B08"].attrs.pop("central_wavelength")
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        coord = list(result["toa_reflectance_10m"].coords["stack_dim_10m"].values)
        assert all(isinstance(c, str) for c in coord)

    def test_nested_dim_coord_attr(self):
        ds = xr.Dataset()
        for name, wl in [("B02", 490.0), ("B03", 560.0)]:
            ds[name] = _da(
                ("y_10m", "x_10m"),
                measurand="toa_reflectance",
                extra_attrs={"product_metadata": {"central_wavelength": wl}},
            )
        result = StackCubes(params={"dim_coord_attr": "product_metadata.central_wavelength"}).run(ds)
        coord = list(result["toa_reflectance_10m"].coords["stack_dim_10m"].values)
        assert coord == [490.0, 560.0]

    def test_dim_coord_attr_stored_as_list_in_attrs(self, wavelength_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        # central_wavelength differs per band — kept as a list attr in stacking order
        wl_attr = result["toa_reflectance_10m"].attrs["central_wavelength"]
        assert isinstance(wl_attr, list)
        assert wl_attr == [490.0, 560.0, 665.0, 842.0]

    def test_original_names_kept_as_auxiliary_coord(self, wavelength_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        name_coord = list(result["toa_reflectance_10m"].coords["stack_dim_10m_name"].values)
        assert name_coord == ["B02", "B03", "B04", "B08"]

    def test_auxiliary_name_coord_aligned_with_dim_coord(self, wavelength_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        cube = result["toa_reflectance_10m"]
        idx = list(cube.coords["stack_dim_10m"].values).index(665.0)
        assert cube.coords["stack_dim_10m_name"].values[idx] == "B04"

    def test_auxiliary_name_coord_has_long_name(self, wavelength_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        assert result["toa_reflectance_10m"].coords["stack_dim_10m_name"].attrs["long_name"]

    def test_no_auxiliary_name_coord_when_fallback_to_names(self, wavelength_ds):
        wavelength_ds["B08"].attrs.pop("central_wavelength")
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(wavelength_ds)
        assert "stack_dim_10m_name" not in result["toa_reflectance_10m"].coords

    def test_no_auxiliary_name_coord_when_dim_coord_attr_unset(self, wavelength_ds):
        result = StackCubes().run(wavelength_ds)
        assert "stack_dim_10m_name" not in result["toa_reflectance_10m"].coords

    def test_dim_coord_attr_tolerates_json_stringified_metadata(self):
        # Regression test: a reader may leave product_metadata as a JSON string
        # (netCDF-safe) rather than a raw dict by the time this processor runs.
        ds = xr.Dataset()
        for name, wl in [("B02", 490.0), ("B03", 560.0)]:
            ds[name] = _da(
                ("y_10m", "x_10m"),
                measurand="toa_reflectance",
                extra_attrs={"product_metadata": json.dumps({"central_wavelength": wl})},
            )
        result = StackCubes(params={"dim_coord_attr": "product_metadata.central_wavelength"}).run(ds)
        coord = list(result["toa_reflectance_10m"].coords["stack_dim_10m"].values)
        assert coord == [490.0, 560.0]


# ---------------------------------------------------------------------------
# Unit tests — _collect_per_band_ancillaries
# ---------------------------------------------------------------------------


class TestCollectPerBandAncillaries:
    def test_finds_per_band_ancillaries(self, ancillary_ds):
        ordered = ["B02", "B03", "B04"]
        groups = _collect_per_band_ancillaries(ancillary_ds, ordered)
        assert len(groups) == 1
        assert groups[0] == [
            "viewing_zenith_angle_B02",
            "viewing_zenith_angle_B03",
            "viewing_zenith_angle_B04",
        ]

    def test_shared_ancillary_excluded(self, ancillary_ds):
        ordered = ["B02", "B03", "B04"]
        groups = _collect_per_band_ancillaries(ancillary_ds, ordered)
        flat = [v for group in groups for v in group]
        assert "solar_zenith_angle" not in flat

    def test_no_ancillary_variables_returns_empty(self, s2_like_ds):
        groups = _collect_per_band_ancillaries(s2_like_ds, ["B02", "B03", "B04", "B08"])
        assert groups == []

    def test_ancillary_variable_not_in_ds_ignored(self):
        ds = xr.Dataset()
        ds["B02"] = _da(
            ("y_10m", "x_10m"), measurand="toa_reflectance", extra_attrs={"ancillary_variables": "missing_var"}
        )
        ds["B03"] = _da(
            ("y_10m", "x_10m"), measurand="toa_reflectance", extra_attrs={"ancillary_variables": "also_missing"}
        )
        groups = _collect_per_band_ancillaries(ds, ["B02", "B03"])
        assert groups == []


# ---------------------------------------------------------------------------
# Integration tests — ancillary stacking
# ---------------------------------------------------------------------------


class TestAncillaryStacking:
    def test_per_band_ancillaries_stacked(self, ancillary_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(ancillary_ds)
        assert "viewing_zenith_angle_10m" in result

    def test_ancillary_cube_shares_stacking_coord(self, ancillary_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(ancillary_ds)
        refl_coord = list(result["toa_reflectance_10m"].coords["stack_dim_10m"].values)
        anc_coord = list(result["viewing_zenith_angle_10m"].coords["stack_dim_10m"].values)
        assert refl_coord == anc_coord

    def test_ancillary_coord_inherits_wavelength_values(self, ancillary_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(ancillary_ds)
        coord = list(result["viewing_zenith_angle_10m"].coords["stack_dim_10m"].values)
        assert coord == [490.0, 560.0, 665.0]

    def test_ancillary_cube_also_gets_auxiliary_name_coord(self, ancillary_ds):
        # stack_dim_10m_name is a dataset-level coordinate on the shared dim, so the
        # ancillary cube picks it up too even though only the parent cube set it.
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(ancillary_ds)
        name_coord = list(result["viewing_zenith_angle_10m"].coords["stack_dim_10m_name"].values)
        assert name_coord == ["B02", "B03", "B04"]

    def test_shared_ancillary_not_stacked_into_band_cube(self, ancillary_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(ancillary_ds)
        assert "solar_zenith_angle" in result
        assert result["solar_zenith_angle"].dims == ("y_10m", "x_10m")

    def test_ancillary_originals_dropped(self, ancillary_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(ancillary_ds)
        for name in ["viewing_zenith_angle_B02", "viewing_zenith_angle_B03", "viewing_zenith_angle_B04"]:
            assert name not in result

    def test_ancillary_originals_kept_when_drop_originals_false(self, ancillary_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength", "drop_originals": False}).run(ancillary_ds)
        assert "viewing_zenith_angle_B02" in result
        assert "viewing_zenith_angle_10m" in result

    def test_parent_cube_ancillary_variables_attr_updated(self, ancillary_ds):
        result = StackCubes(params={"dim_coord_attr": "central_wavelength"}).run(ancillary_ds)
        anc_attr = result["toa_reflectance_10m"].attrs.get("ancillary_variables", "")
        assert "viewing_zenith_angle_10m" in anc_attr

    def test_stack_ancillaries_false_disables_feature(self, ancillary_ds):
        result = StackCubes(params={"stack_ancillaries": False}).run(ancillary_ds)
        # Ancillary-aware cube not created
        assert "viewing_zenith_angle_10m" not in result
        # Angle vars get lumped into a single measurand-based group instead
        assert "angle_10m" in result


class TestGetCoordUnits:
    def _da(self, attrs):
        return xr.DataArray(np.ones(2), dims=("x",), attrs=attrs)

    def test_flat_unit_attr(self):
        da = self._da({"central_wavelength": 490.0, "central_wavelength_unit": "nm"})
        assert _get_coord_units(da, "central_wavelength") == "nm"

    def test_flat_units_attr_fallback(self):
        da = self._da({"central_wavelength": 490.0, "central_wavelength_units": "nm"})
        assert _get_coord_units(da, "central_wavelength") == "nm"

    def test_nested_unit_attr(self):
        da = self._da({"product_metadata": {"band_central_wavelength": 492.7, "band_central_wavelength_unit": "nm"}})
        assert _get_coord_units(da, "product_metadata.band_central_wavelength") == "nm"

    def test_missing_unit_returns_none(self):
        da = self._da({"central_wavelength": 490.0})
        assert _get_coord_units(da, "central_wavelength") is None

    def test_dim_attrs_applied_to_stacking_coord(self):
        ds = xr.Dataset()
        for name, wl in [("B02", 492.7), ("B03", 559.8)]:
            ds[name] = xr.DataArray(
                np.ones((2, 2)),
                dims=("y_10m", "x_10m"),
                attrs={"measurand": "reflectance", "central_wavelength": wl},
            )
        result = StackCubes(
            params={
                "dim_coord_attr": "central_wavelength",
                "dim_attrs": {"long_name": "centre wavelength", "standard_name": "radiation_wavelength"},
            }
        ).run(ds)
        dim = [d for d in result["reflectance_10m"].dims if d.startswith("stack")][0]
        assert result["reflectance_10m"][dim].attrs["long_name"] == "centre wavelength"
        assert result["reflectance_10m"][dim].attrs["standard_name"] == "radiation_wavelength"

    def test_dim_attrs_override_auto_units(self):
        ds = xr.Dataset()
        for name, wl in [("B02", 492.7), ("B03", 559.8)]:
            ds[name] = xr.DataArray(
                np.ones((2, 2)),
                dims=("y_10m", "x_10m"),
                attrs={"measurand": "reflectance", "central_wavelength": wl, "central_wavelength_unit": "nm"},
            )
        result = StackCubes(
            params={
                "dim_coord_attr": "central_wavelength",
                "dim_attrs": {"units": "Angstrom"},
            }
        ).run(ds)
        dim = [d for d in result["reflectance_10m"].dims if d.startswith("stack")][0]
        assert result["reflectance_10m"][dim].attrs["units"] == "Angstrom"

    def test_coord_units_set_on_stacking_dim(self):
        ds = xr.Dataset()
        for name, wl in [("B02", 492.7), ("B03", 559.8)]:
            ds[name] = xr.DataArray(
                np.ones((2, 2)),
                dims=("y_10m", "x_10m"),
                attrs={
                    "measurand": "reflectance",
                    "product_metadata": {"band_central_wavelength": wl, "band_central_wavelength_unit": "nm"},
                },
            )
        result = StackCubes(params={"dim_coord_attr": "product_metadata.band_central_wavelength"}).run(ds)
        dim = [d for d in result["reflectance_10m"].dims if d.startswith("stack")][0]
        assert result["reflectance_10m"][dim].attrs.get("units") == "nm"
