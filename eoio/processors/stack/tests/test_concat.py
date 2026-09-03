"""Tests for eoio.processors.stack._concat"""

import numpy as np
import pytest
import xarray as xr

from eoio.processors.stack._concat import (
    _concat_datasets,
    _dataset_schema,
    _dedupe_dims,
    _reconcile_attrs,
    _resolve_concat_coord_values,
    _restore_repeated_dims,
    _split_repeated_dims,
    _validate_and_filter_datasets,
)


def _time_ds(n_time, value=1.0, x_size=3):
    return xr.Dataset(
        {"spectra": (("time", "x"), np.full((n_time, x_size), value, dtype="float32"))},
        coords={"time": np.arange(n_time)},
    )


def _raster_ds(value, y_size=2, x_size=2, sensing_time=None):
    ds = xr.Dataset({"reflectance_10m": (("y_10m", "x_10m"), np.full((y_size, x_size), value, dtype="float32"))})
    if sensing_time is not None:
        ds.attrs["product_metadata"] = {"sensing_time": sensing_time}
    return ds


class TestDatasetSchema:
    def test_ignores_concat_dim(self):
        a = _time_ds(2)
        b = _time_ds(5)
        assert _dataset_schema(a, "time") == _dataset_schema(b, "time")

    def test_differs_on_other_dim_size(self):
        a = _time_ds(2, x_size=3)
        b = _time_ds(2, x_size=4)
        assert _dataset_schema(a, "time") != _dataset_schema(b, "time")

    def test_differs_on_variable_set(self):
        a = _time_ds(2)
        b = _time_ds(2)
        b["extra"] = ("time", np.arange(2))
        assert _dataset_schema(a, "time") != _dataset_schema(b, "time")


class TestValidateAndFilterDatasets:
    def test_all_matching_kept(self):
        datasets = [_time_ds(2), _time_ds(3), _time_ds(1)]
        paths = ["a.nc", "b.nc", "c.nc"]
        kept_ds, kept_paths = _validate_and_filter_datasets(datasets, paths, "time", "error")
        assert kept_ds == datasets
        assert kept_paths == paths

    def test_error_on_mismatch(self):
        datasets = [_time_ds(2, x_size=3), _time_ds(2, x_size=4)]
        paths = ["a.nc", "b.nc"]
        with pytest.raises(ValueError, match="b.nc"):
            _validate_and_filter_datasets(datasets, paths, "time", "error")

    def test_skip_drops_mismatched(self):
        datasets = [_time_ds(2, x_size=3), _time_ds(2, x_size=4), _time_ds(2, x_size=3)]
        paths = ["a.nc", "b.nc", "c.nc"]
        with pytest.warns(UserWarning, match="b.nc"):
            kept_ds, kept_paths = _validate_and_filter_datasets(datasets, paths, "time", "skip")
        assert kept_paths == ["a.nc", "c.nc"]
        assert kept_ds == [datasets[0], datasets[2]]

    def test_skip_raises_if_fewer_than_two_remain(self):
        datasets = [_time_ds(2, x_size=3), _time_ds(2, x_size=4), _time_ds(2, x_size=5)]
        paths = ["a.nc", "b.nc", "c.nc"]
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="fewer than 2"):
                _validate_and_filter_datasets(datasets, paths, "time", "skip")

    def test_union_skips_validation(self):
        datasets = [_time_ds(2, x_size=3), _time_ds(2, x_size=4)]
        paths = ["a.nc", "b.nc"]
        kept_ds, kept_paths = _validate_and_filter_datasets(datasets, paths, "time", "union")
        assert kept_ds == datasets
        assert kept_paths == paths


class TestConcatDatasets:
    def test_concat_along_existing_dim(self):
        datasets = [_time_ds(2, value=1.0), _time_ds(3, value=2.0)]
        paths = ["a.nc", "b.nc"]
        result = _concat_datasets(datasets, paths, "time", concat_coord=None, on_mismatch="error")
        assert result.sizes["time"] == 5
        assert list(result["spectra"].isel(x=0).values) == [1.0, 1.0, 2.0, 2.0, 2.0]

    def test_invalid_on_mismatch_raises(self):
        datasets = [_time_ds(2), _time_ds(2)]
        with pytest.raises(ValueError, match="on_mismatch"):
            _concat_datasets(datasets, ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="bogus")

    def test_on_mismatch_union_fills_missing_variable_with_nan(self):
        a = _time_ds(2)
        b = _time_ds(2)
        b["extra"] = ("time", np.array([9.0, 9.0]))
        result = _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="union")
        assert result.sizes["time"] == 4
        assert np.isnan(result["extra"].isel(time=0).item())
        assert result["extra"].isel(time=2).item() == 9.0

    # -- raster case: new dimension via expand_dims --------------------------

    def test_raster_stack_with_str_concat_coord(self):
        datasets = [
            _raster_ds(1.0, sensing_time="2026-07-01"),
            _raster_ds(2.0, sensing_time="2026-07-02"),
        ]
        paths = ["a.nc", "b.nc"]
        result = _concat_datasets(
            datasets, paths, "time", concat_coord="product_metadata.sensing_time", on_mismatch="error"
        )
        assert result["reflectance_10m"].dims == ("time", "y_10m", "x_10m")
        assert list(result.coords["time"].values) == ["2026-07-01", "2026-07-02"]

    def test_raster_stack_with_list_concat_coord(self):
        datasets = [_raster_ds(1.0), _raster_ds(2.0)]
        result = _concat_datasets(
            datasets, ["a.nc", "b.nc"], "time", concat_coord=["2026-07-01", "2026-07-02"], on_mismatch="error"
        )
        assert list(result.coords["time"].values) == ["2026-07-01", "2026-07-02"]

    def test_raster_stack_with_callable_concat_coord(self):
        datasets = [_raster_ds(1.0), _raster_ds(2.0)]
        paths = ["scene_2026-07-01.nc", "scene_2026-07-02.nc"]
        coord_fn = lambda path, ds: path.split("_")[1].removesuffix(".nc")  # noqa: E731
        result = _concat_datasets(datasets, paths, "time", concat_coord=coord_fn, on_mismatch="error")
        assert list(result.coords["time"].values) == ["2026-07-01", "2026-07-02"]

    def test_missing_dim_no_concat_coord_falls_back_to_index_with_warning(self):
        datasets = [_raster_ds(1.0), _raster_ds(2.0)]
        with pytest.warns(UserWarning, match="integer index"):
            result = _concat_datasets(datasets, ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error")
        assert list(result.coords["time"].values) == [0, 1]

    def test_raster_grid_mismatch_governed_by_on_mismatch(self):
        datasets = [
            _raster_ds(1.0, sensing_time="2026-07-01"),
            _raster_ds(2.0, y_size=3, sensing_time="2026-07-02"),  # different grid
        ]
        with pytest.raises(ValueError, match="b.nc"):
            _concat_datasets(
                datasets, ["a.nc", "b.nc"], "time", concat_coord="product_metadata.sensing_time", on_mismatch="error"
            )


class TestResolveConcatCoordValues:
    def test_str_reads_nested_ds_attr(self):
        datasets = [_raster_ds(1.0, sensing_time="2026-07-01"), _raster_ds(2.0, sensing_time="2026-07-02")]
        values = _resolve_concat_coord_values(datasets, ["a.nc", "b.nc"], "time", "product_metadata.sensing_time")
        assert values == ["2026-07-01", "2026-07-02"]

    def test_str_missing_attr_raises(self):
        datasets = [_raster_ds(1.0), _raster_ds(2.0)]
        with pytest.raises(ValueError, match="a.nc"):
            _resolve_concat_coord_values(datasets, ["a.nc", "b.nc"], "time", "product_metadata.sensing_time")

    def test_list_wrong_length_raises(self):
        datasets = [_raster_ds(1.0), _raster_ds(2.0)]
        with pytest.raises(ValueError, match="2 files"):
            _resolve_concat_coord_values(datasets, ["a.nc", "b.nc"], "time", ["only_one"])

    def test_invalid_type_raises(self):
        datasets = [_raster_ds(1.0), _raster_ds(2.0)]
        with pytest.raises(TypeError, match="concat_coord"):
            _resolve_concat_coord_values(datasets, ["a.nc", "b.nc"], "time", 123)


class TestReconcileAttrs:
    def test_shared_attr_kept_as_scalar(self):
        attrs, to_promote = _reconcile_attrs([{"a": 1}, {"a": 1}, {"a": 1}])
        assert attrs == {"a": 1}
        assert to_promote == {}

    def test_differing_attr_kept_as_per_file_list(self):
        attrs, to_promote = _reconcile_attrs([{"a": 1}, {"a": 2}])
        assert attrs == {"a": [1, 2]}
        assert to_promote == {}

    def test_attr_missing_from_first_file_still_kept(self):
        # regression case: _cube_attrs_and_coords only looks at the first
        # object's keys and would silently drop this; _reconcile_attrs must not.
        attrs, to_promote = _reconcile_attrs([{}, {"b": 2}])
        assert attrs == {"b": [None, 2]}

    def test_promoted_attr_excluded_from_attrs(self):
        attrs, to_promote = _reconcile_attrs([{"a": 1}, {"a": 2}], coord_attrs=["a"])
        assert "a" not in attrs
        assert to_promote == {"a": [1, 2]}

    def test_promotion_repeats_value_by_chunk_size(self):
        _, to_promote = _reconcile_attrs([{"a": 1}, {"a": 2}], coord_attrs=["a"], chunk_sizes=[2, 1])
        assert to_promote == {"a": [1, 1, 2]}

    def test_array_valued_attr_does_not_raise(self):
        # regression case: real EO products (e.g. Hypernets) carry array-valued
        # attrs, such as empty placeholder arrays for unused uncertainty-effect
        # parameters. Plain `==` on two arrays returns an array, not a bool,
        # which used to raise inside `all(...)`.
        shared = np.array([], dtype="float64")
        attrs, _ = _reconcile_attrs([{"err_corr_1_units": shared}, {"err_corr_1_units": shared.copy()}])
        assert np.array_equal(attrs["err_corr_1_units"], shared)

    def test_differing_array_valued_attr_kept_as_list(self):
        attrs, _ = _reconcile_attrs([{"a": np.array([1, 2])}, {"a": np.array([3, 4])}])
        assert len(attrs["a"]) == 2
        np.testing.assert_array_equal(attrs["a"][0], [1, 2])
        np.testing.assert_array_equal(attrs["a"][1], [3, 4])

    def test_non_scalar_differing_attr_not_promoted_even_if_named(self):
        attrs, to_promote = _reconcile_attrs([{"a": {"x": 1}}, {"a": {"x": 2}}], coord_attrs=["a"])
        assert to_promote == {}
        assert attrs == {"a": [{"x": 1}, {"x": 2}]}

    def test_empty_input_returns_empty(self):
        assert _reconcile_attrs([]) == ({}, {})


class TestConcatDatasetsAttrs:
    def test_reconcile_is_default_and_keeps_both_files_values(self):
        a = _time_ds(1, value=1.0)
        a.attrs["sensing_time"] = "A"
        a["spectra"].attrs["scale_factor"] = 1.1
        b = _time_ds(1, value=2.0)
        b.attrs["sensing_time"] = "B"
        b["spectra"].attrs["scale_factor"] = 2.2

        result = _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error")
        assert result.attrs["sensing_time"] == ["A", "B"]
        assert result["spectra"].attrs["scale_factor"] == [1.1, 2.2]

    def test_attrs_first_keeps_only_first_files_values(self):
        a = _time_ds(1, value=1.0)
        a.attrs["sensing_time"] = "A"
        b = _time_ds(1, value=2.0)
        b.attrs["sensing_time"] = "B"

        result = _concat_datasets(
            [a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error", attrs="first"
        )
        assert result.attrs["sensing_time"] == "A"

    def test_invalid_attrs_mode_raises(self):
        a, b = _time_ds(1), _time_ds(1)
        with pytest.raises(ValueError, match="attrs"):
            _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error", attrs="bogus")

    def test_coord_attrs_promotes_to_real_coordinate(self):
        a = _time_ds(1, value=1.0)
        a.attrs["product_name"] = "scene_A"
        b = _time_ds(1, value=2.0)
        b.attrs["product_name"] = "scene_B"

        result = _concat_datasets(
            [a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error", coord_attrs=["product_name"]
        )
        assert "product_name" not in result.attrs
        assert list(result.coords["product_name"].values) == ["scene_A", "scene_B"]

    def test_coord_attrs_repeats_value_across_multi_step_file(self):
        # file `a` contributes 2 time steps, `b` contributes 1 - the promoted
        # coordinate must repeat `a`'s value across both of its time steps.
        a = _time_ds(2, value=1.0)
        a.attrs["product_name"] = "scene_A"
        b = _time_ds(1, value=2.0)
        b["time"] = [2]
        b.attrs["product_name"] = "scene_B"

        result = _concat_datasets(
            [a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error", coord_attrs=["product_name"]
        )
        assert list(result.coords["product_name"].values) == ["scene_A", "scene_A", "scene_B"]

    def test_processing_steps_merged_by_concatenation_not_nested(self):
        a = _time_ds(1, value=1.0)
        a.attrs["eoio:processing_steps"] = [{"processor": "reader_a"}]
        b = _time_ds(1, value=2.0)
        b["time"] = [1]
        b.attrs["eoio:processing_steps"] = [{"processor": "reader_b"}]

        result = _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error")
        assert result.attrs["eoio:processing_steps"] == [{"processor": "reader_a"}, {"processor": "reader_b"}]


def _corr_ds(n_meas, value, with_time=None, sensing_time=None):
    corr = np.eye(n_meas, dtype="float32") * value
    coords = {"measurement": np.arange(n_meas)}
    data = {
        "radiance": ("measurement", np.full(n_meas, value, dtype="float32")),
        "err_corr_radiance": (("measurement", "measurement"), corr),
    }
    if with_time is not None:
        coords["time"] = [with_time]
    ds = xr.Dataset(data, coords=coords)
    if sensing_time is not None:
        ds.attrs["sensing_time"] = sensing_time
    return ds


class TestDedupeDims:
    def test_no_repeats_unchanged(self):
        assert _dedupe_dims(("time", "y", "x")) == ("time", "y", "x")

    def test_single_repeat_suffixed(self):
        assert _dedupe_dims(("measurement", "measurement")) == ("measurement", "measurement__dup1")

    def test_repeat_with_other_dims(self):
        assert _dedupe_dims(("time", "measurement", "measurement")) == ("time", "measurement", "measurement__dup1")

    def test_triple_repeat(self):
        assert _dedupe_dims(("m", "m", "m")) == ("m", "m__dup1", "m__dup2")


class TestSplitAndRestoreRepeatedDims:
    def test_split_renames_second_occurrence(self):
        ds = _corr_ds(3, 1.0)
        split_ds, touched = _split_repeated_dims(ds)
        assert touched == ["err_corr_radiance"]
        assert split_ds["err_corr_radiance"].dims == ("measurement", "measurement__dup1")
        # unaffected variable/original untouched
        assert split_ds["radiance"].dims == ("measurement",)
        assert ds["err_corr_radiance"].dims == ("measurement", "measurement")  # original not mutated

    def test_no_repeats_returns_untouched_list(self):
        ds = _time_ds(2)
        _, touched = _split_repeated_dims(ds)
        assert touched == []

    def test_split_then_restore_round_trips(self):
        ds = _corr_ds(3, 1.0)
        split_ds, touched = _split_repeated_dims(ds)
        restored = _restore_repeated_dims(split_ds, touched)
        assert restored["err_corr_radiance"].dims == ("measurement", "measurement")
        assert dict(restored.sizes) == {"measurement": 3}  # no phantom dup dim left over
        np.testing.assert_array_equal(restored["err_corr_radiance"].values, ds["err_corr_radiance"].values)

    def test_restore_no_touched_is_noop(self):
        ds = _time_ds(2)
        assert _restore_repeated_dims(ds, []) is ds


class TestConcatDatasetsRepeatedDims:
    def test_insitu_concat_of_repeated_dim_variable(self):
        a = _corr_ds(3, 1.0, with_time=0)
        b = _corr_ds(3, 2.0, with_time=1)

        result = _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error")

        assert result["err_corr_radiance"].dims == ("time", "measurement", "measurement")
        assert dict(result.sizes) == {"time": 2, "measurement": 3}
        np.testing.assert_array_equal(result["err_corr_radiance"].isel(time=0).values, np.eye(3, dtype="float32"))
        np.testing.assert_array_equal(result["err_corr_radiance"].isel(time=1).values, np.eye(3, dtype="float32") * 2.0)

    def test_raster_style_expand_dims_of_repeated_dim_variable(self):
        a = _corr_ds(3, 1.0, sensing_time="2026-07-01")
        b = _corr_ds(3, 2.0, sensing_time="2026-07-02")

        result = _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord="sensing_time", on_mismatch="error")

        assert result["err_corr_radiance"].dims == ("time", "measurement", "measurement")
        assert list(result["time"].values) == ["2026-07-01", "2026-07-02"]

    def test_result_survives_netcdf_round_trip(self, tmp_path):
        a = _corr_ds(3, 1.0, with_time=0)
        b = _corr_ds(3, 2.0, with_time=1)
        result = _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error")

        fp = tmp_path / "repeated_dim_result.nc"
        result.to_netcdf(fp)
        roundtripped = xr.open_dataset(fp)

        assert roundtripped["err_corr_radiance"].dims == ("time", "measurement", "measurement")
        np.testing.assert_array_equal(roundtripped["err_corr_radiance"].values, result["err_corr_radiance"].values)

    def test_mixed_repeated_and_plain_variables_alongside_each_other(self):
        # radiance has no repeated dim, err_corr_radiance does - both must
        # concatenate correctly in the same call.
        a = _corr_ds(3, 1.0, with_time=0)
        b = _corr_ds(3, 2.0, with_time=1)

        result = _concat_datasets([a, b], ["a.nc", "b.nc"], "time", concat_coord=None, on_mismatch="error")

        assert result["radiance"].dims == ("time", "measurement")
        assert list(result["radiance"].isel(time=1).values) == [2.0, 2.0, 2.0]
