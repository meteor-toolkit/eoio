"""Tests for eoio.serialisation.attrs"""

import datetime as dt
import json
import warnings

import numpy as np
import pytest
import xarray as xr

from eoio.serialisation import decode_attrs, encode_attrs
from eoio.serialisation.codecs import MARKER_ATTR


@pytest.fixture
def rich_attrs():
    """Attributes representative of what eoio readers actually produce."""
    return {
        "title": "Sentinel-2A MSI Level-1C",
        # From the controlled vocabulary. Must survive as a string: the deprecated
        # revert_netcdf_compatible() turns this into the float 3.01.
        "product_version": "03.01",
        "date_created": dt.datetime(2023, 7, 2, 10, 56, 19, tzinfo=dt.timezone.utc),
        "product_metadata": {
            "sensing_time": dt.datetime(2023, 7, 2, 10, 56, 19),
            "datatake": {"id": "GS2A_20230702", "orbit": 65},
        },
        # Produced by read_multi, and the reason its output cannot be written today.
        "eoio:processing_steps": [{"processor": "read_multi", "n_files": 4}],
        "quality_passed": True,
        "missing": None,
        "roi_bounds": [1.0, 2.0, 3.0, 4.0],
    }


@pytest.fixture
def rich_dataset(rich_attrs):
    ds = xr.Dataset(
        data_vars={"B02": (("y", "x"), np.arange(4, dtype="float32").reshape(2, 2))},
        coords={"x": ("x", [0, 1]), "y": ("y", [0, 1])},
        attrs=dict(rich_attrs),
    )
    ds["B02"].attrs = {"units": "1", "product_metadata": {"band": {"id": "B02", "centre_nm": 492.7}}}
    ds["x"].attrs = {"standard_name": "projection_x_coordinate", "created": dt.date(2023, 7, 2)}
    return ds


def _assert_attrs_restored(restored, original):
    """Compare decoded attrs to the originals by value and type."""
    assert set(restored) == set(original)
    for key, expected in original.items():
        assert restored[key] == expected, f"{key}: {restored[key]!r} != {expected!r}"
        assert type(restored[key]) is type(expected), f"{key}: type {type(restored[key])} != {type(expected)}"


# ---------------------------------------------------------------------------
# In-memory round trip
# ---------------------------------------------------------------------------


def test_round_trip_in_memory(rich_dataset, rich_attrs):
    restored = decode_attrs(encode_attrs(rich_dataset))
    _assert_attrs_restored(restored.attrs, rich_attrs)


def test_round_trip_covers_variable_and_coordinate_scopes(rich_dataset):
    restored = decode_attrs(encode_attrs(rich_dataset))
    assert restored["B02"].attrs["product_metadata"] == {"band": {"id": "B02", "centre_nm": 492.7}}
    assert restored["x"].attrs["created"] == dt.date(2023, 7, 2)
    assert restored["x"].attrs["standard_name"] == "projection_x_coordinate"


def test_encode_does_not_mutate_the_input(rich_dataset):
    before = dict(rich_dataset.attrs)
    before_var = dict(rich_dataset["B02"].attrs)
    encode_attrs(rich_dataset)
    assert rich_dataset.attrs == before
    assert rich_dataset["B02"].attrs == before_var
    assert isinstance(rich_dataset.attrs["date_created"], dt.datetime)


def test_encode_inplace_mutates(rich_dataset):
    returned = encode_attrs(rich_dataset, inplace=True)
    assert returned is rich_dataset
    assert MARKER_ATTR in rich_dataset.attrs


# ---------------------------------------------------------------------------
# Dot flattening
# ---------------------------------------------------------------------------


def test_nested_dicts_are_dot_flattened(rich_dataset):
    encoded = encode_attrs(rich_dataset)
    assert "product_metadata.sensing_time" in encoded.attrs
    assert "product_metadata.datatake.id" in encoded.attrs
    assert encoded.attrs["product_metadata.datatake.orbit"] == 65
    assert "product_metadata" not in encoded.attrs


def test_marker_records_the_codec_for_each_encoded_attribute(rich_dataset):
    marker = json.loads(encode_attrs(rich_dataset).attrs[MARKER_ATTR])
    assert marker["product_metadata"] == "dict_dot"
    assert marker["product_metadata.sensing_time"] == "datetime"
    assert marker["quality_passed"] == "bool"
    assert marker["missing"] == "none"
    assert marker["eoio:processing_steps"] == "json"
    # Attributes needing no encoding are not recorded.
    assert "title" not in marker
    assert "product_version" not in marker


def test_dict_style_json_keeps_the_dict_as_one_attribute(rich_dataset):
    encoded = encode_attrs(rich_dataset, dict_style="json")
    assert encoded.attrs["product_metadata"].startswith("{")
    assert "product_metadata.datatake.id" not in encoded.attrs
    assert decode_attrs(encoded).attrs["product_metadata"]["datatake"]["orbit"] == 65


def test_invalid_dict_style_raises(rich_dataset):
    with pytest.raises(ValueError, match="dict_style"):
        encode_attrs(rich_dataset, dict_style="flat")


@pytest.mark.parametrize(
    "attrs, reason",
    [
        ({"md": {"a.b": 1}}, "a key containing a dot is ambiguous once flattened"),
        ({"md": {1: "x"}}, "non-string keys cannot be addressed by dot-path"),
        ({"md": {}}, "flattening an empty dict would lose the attribute"),
        ({"md": {"x": 1}, "md.x": "collides"}, "the flattened name already exists"),
    ],
)
def test_unflattenable_dicts_fall_back_to_json(attrs, reason):
    ds = xr.Dataset(attrs=attrs)
    with warnings.catch_warnings():
        # The non-string-key case warns; that is asserted separately below.
        warnings.simplefilter("ignore")
        encoded = encode_attrs(ds)
    assert isinstance(encoded.attrs["md"], str), reason
    assert json.loads(encoded.attrs[MARKER_ATTR])["md"] == "json"


def test_dots_in_keys_still_round_trip_via_the_json_fallback():
    ds = xr.Dataset(attrs={"md": {"a.b": 1}})
    assert decode_attrs(encode_attrs(ds)).attrs["md"] == {"a.b": 1}


def test_colliding_flattened_name_round_trips_both_attributes():
    ds = xr.Dataset(attrs={"md": {"x": 1}, "md.x": "collides"})
    restored = decode_attrs(encode_attrs(ds)).attrs
    assert restored["md"] == {"x": 1}
    assert restored["md.x"] == "collides"


# ---------------------------------------------------------------------------
# Sniffing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"not": "really an object"} trailing',  # does not parse as JSON
        '{"cloud_cover": 0.4}',  # genuinely valid JSON, but stored as a string
        "[1, 2, 3]",
        "2023-07-02T10:56:19",  # genuinely an ISO datetime, but stored as a string
    ],
)
def test_string_that_sniffing_would_convert_survives_a_round_trip(text):
    # Encoding pins ambiguous strings in the marker, so a string that happens to be
    # valid JSON or a valid timestamp is not silently retyped on the way back.
    ds = xr.Dataset(attrs={"note": text})
    restored = decode_attrs(encode_attrs(ds)).attrs["note"]
    assert restored == text
    assert isinstance(restored, str)


def test_ambiguous_strings_are_pinned_in_the_marker():
    ds = xr.Dataset(attrs={"note": '{"a": 1}', "plain": "hello"})
    marker = json.loads(encode_attrs(ds).attrs[MARKER_ATTR])
    assert marker["note"] == "str"
    # Unambiguous strings stay out of the marker, so it does not bloat.
    assert "plain" not in marker


def test_string_pinning_survives_netcdf(tmp_path):
    ds = xr.Dataset(attrs={"note": '{"cloud_cover": 0.4}'})
    encode_attrs(ds).to_netcdf(tmp_path / "pinned.nc")
    with xr.open_dataset(tmp_path / "pinned.nc") as opened:
        assert decode_attrs(opened).attrs["note"] == '{"cloud_cover": 0.4}'


def test_numeric_looking_strings_are_not_coerced():
    ds = xr.Dataset(attrs={"product_version": "03.01", "count": "42"})
    restored = decode_attrs(ds).attrs
    assert restored["product_version"] == "03.01"
    assert restored["count"] == "42"


def test_sniffing_recognises_json_and_iso_datetimes_without_a_marker():
    # No marker: this is what a third-party netCDF looks like.
    ds = xr.Dataset(attrs={"blob": '{"a": 1}', "when": "2023-07-02T10:56:19", "list": "[1, 2]"})
    restored = decode_attrs(ds).attrs
    assert restored["blob"] == {"a": 1}
    assert restored["list"] == [1, 2]
    assert restored["when"] == dt.datetime(2023, 7, 2, 10, 56, 19)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2023-07-02T10:56:19", dt.datetime(2023, 7, 2, 10, 56, 19)),
        ("2023-07-02T10:56:19Z", dt.datetime(2023, 7, 2, 10, 56, 19, tzinfo=dt.timezone.utc)),
        ("2023-07-02T10:56:19+00:00", dt.datetime(2023, 7, 2, 10, 56, 19, tzinfo=dt.timezone.utc)),
        ("2023-07-02 10:56", dt.datetime(2023, 7, 2, 10, 56)),
    ],
)
def test_sniffing_handles_iso_variants_including_a_trailing_z(text, expected):
    assert decode_attrs(xr.Dataset(attrs={"when": text})).attrs["when"] == expected


def test_sniffing_leaves_bare_dates_alone():
    # Date-like strings are common as genuine string metadata, so a time component is
    # required before a value is treated as a datetime.
    restored = decode_attrs(xr.Dataset(attrs={"d": "2023-07-02"})).attrs
    assert restored["d"] == "2023-07-02"


def test_sniffing_can_be_disabled():
    ds = xr.Dataset(attrs={"blob": '{"a": 1}', "when": "2023-07-02T10:56:19"})
    restored = decode_attrs(ds, sniff=False).attrs
    assert restored["blob"] == '{"a": 1}'
    assert restored["when"] == "2023-07-02T10:56:19"


def test_marker_takes_precedence_over_sniffing():
    # The marker says this is a plain string, so it must not be sniffed into a dict.
    ds = xr.Dataset(attrs={"note": '{"a": 1}', MARKER_ATTR: json.dumps({"note": "bytes"})})
    assert decode_attrs(ds).attrs["note"] == b'{"a": 1}'


# ---------------------------------------------------------------------------
# Marker robustness
# ---------------------------------------------------------------------------


def test_unreadable_marker_warns_and_falls_back_to_sniffing():
    ds = xr.Dataset(attrs={"blob": '{"a": 1}', MARKER_ATTR: "not json at all"})
    with pytest.warns(UserWarning, match="unreadable"):
        restored = decode_attrs(ds).attrs
    assert restored["blob"] == {"a": 1}


def test_unknown_codec_in_marker_warns_and_keeps_the_encoded_value():
    ds = xr.Dataset(attrs={"v": "x", MARKER_ATTR: json.dumps({"v": "no_such_codec"})})
    with pytest.warns(UserWarning, match="Could not decode"):
        restored = decode_attrs(ds).attrs
    assert restored["v"] == "x"


def test_decode_removes_the_marker(rich_dataset):
    assert MARKER_ATTR not in decode_attrs(encode_attrs(rich_dataset)).attrs


def test_re_encoding_preserves_the_marker(rich_dataset, rich_attrs):
    twice = encode_attrs(encode_attrs(rich_dataset))
    assert MARKER_ATTR in twice.attrs
    _assert_attrs_restored(decode_attrs(twice).attrs, rich_attrs)


# ---------------------------------------------------------------------------
# Unencodable values
# ---------------------------------------------------------------------------


class _Unsupported:
    def __repr__(self):
        return "<unsupported>"


def test_strict_raises_on_an_unencodable_value():
    ds = xr.Dataset(attrs={"weird": _Unsupported()})
    with pytest.raises(TypeError, match="No codec for attribute 'weird'"):
        encode_attrs(ds, strict=True)


def test_lenient_falls_back_to_str_with_a_warning():
    ds = xr.Dataset(attrs={"weird": _Unsupported()})
    with pytest.warns(UserWarning, match="falling back to str"):
        encoded = encode_attrs(ds)
    assert encoded.attrs["weird"] == "<unsupported>"
    # No marker entry, because it cannot be restored.
    assert "weird" not in json.loads(encoded.attrs.get(MARKER_ATTR, "{}"))


# ---------------------------------------------------------------------------
# Round trips through real IO
# ---------------------------------------------------------------------------


def test_round_trip_through_netcdf(tmp_path, rich_dataset, rich_attrs):
    target = tmp_path / "out.nc"
    encode_attrs(rich_dataset).to_netcdf(target)

    with xr.open_dataset(target) as opened:
        restored = decode_attrs(opened)

    _assert_attrs_restored(restored.attrs, rich_attrs)
    assert restored["B02"].attrs["product_metadata"]["band"]["id"] == "B02"
    assert restored["x"].attrs["created"] == dt.date(2023, 7, 2)


def test_unencoded_dataset_cannot_be_written_to_netcdf(tmp_path, rich_dataset):
    # The problem this module exists to solve.
    with pytest.raises(TypeError):
        rich_dataset.to_netcdf(tmp_path / "fails.nc")


def test_round_trip_through_zarr(tmp_path, rich_dataset, rich_attrs):
    pytest.importorskip("zarr")
    target = tmp_path / "out.zarr"
    encode_attrs(rich_dataset).to_zarr(target)

    with xr.open_zarr(target) as opened:
        restored = decode_attrs(opened)

    _assert_attrs_restored(restored.attrs, rich_attrs)


def test_round_trip_of_a_read_multi_style_dataset(tmp_path):
    # read_multi writes list-of-dict attrs and eoio:processing_steps, which is what
    # currently blocks writing its output to netCDF.
    steps = [
        {"processor": "read_multi", "concat_dim": "time", "on_mismatch": "error", "n_files": 3},
        {"processor": "stack", "dim": "band"},
    ]
    ds = xr.Dataset(
        data_vars={"reflectance": (("time", "band"), np.zeros((3, 2), dtype="float32"))},
        coords={"time": ("time", [0, 1, 2]), "band": ("band", ["B02", "B03"])},
        attrs={
            "eoio:processing_steps": steps,
            "product_name": ["S2A_a.SAFE", "S2A_b.SAFE", "S2A_c.SAFE"],
            "eoio:subset": {"roi": [1.0, 2.0, 3.0, 4.0], "roi_crs": 4326},
        },
    )

    target = tmp_path / "multi.nc"
    encode_attrs(ds).to_netcdf(target)
    with xr.open_dataset(target) as opened:
        restored = decode_attrs(opened)

    assert restored.attrs["eoio:processing_steps"] == steps
    assert restored.attrs["product_name"] == ["S2A_a.SAFE", "S2A_b.SAFE", "S2A_c.SAFE"]
    assert restored.attrs["eoio:subset"] == {"roi": [1.0, 2.0, 3.0, 4.0], "roi_crs": 4326}


# ---------------------------------------------------------------------------
# Object types
# ---------------------------------------------------------------------------


def test_data_array_is_supported():
    da = xr.DataArray([1, 2], dims="x", coords={"x": [0, 1]}, attrs={"when": dt.date(2023, 7, 2)})
    da.coords["x"].attrs["md"] = {"a": 1}
    restored = decode_attrs(encode_attrs(da))
    assert restored.attrs["when"] == dt.date(2023, 7, 2)
    assert restored.coords["x"].attrs["md"] == {"a": 1}


def test_data_tree_round_trips_every_node(tmp_path):
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"root_md": {"campaign": "test"}}),
            "/s2": xr.Dataset({"v": ("x", [1, 2])}, attrs={"acquired": dt.datetime(2023, 7, 2, 10, 0)}),
            "/l8": xr.Dataset({"w": ("y", [3.0])}, attrs={"cloudy": False}),
        }
    )

    target = tmp_path / "tree.nc"
    encode_attrs(tree).to_netcdf(target)
    with xr.open_datatree(target) as opened:
        restored = decode_attrs(opened)

    assert restored.attrs["root_md"] == {"campaign": "test"}
    assert restored["s2"].attrs["acquired"] == dt.datetime(2023, 7, 2, 10, 0)
    assert restored["l8"].attrs["cloudy"] is False


def test_unsupported_object_type_raises():
    with pytest.raises(TypeError, match="Expected an xarray"):
        encode_attrs({"not": "an xarray object"})


# -------------------------
# Types beyond the original codec set
# -------------------------


def _round_trip(tmp_path, value):
    ds = xr.Dataset(attrs={"v": value})
    path = tmp_path / "rt.nc"
    encode_attrs(ds).to_netcdf(path)
    with xr.open_dataset(path) as reopened:
        return decode_attrs(reopened).attrs["v"]


@pytest.mark.parametrize(
    "value",
    [
        np.arange(6.0).reshape(2, 3),
        np.arange(24, dtype="int16").reshape(2, 3, 4),
        np.array([True, False, True]),
        np.array([1 + 2j, 3 - 4j]),
        np.array(["2021-02-08", "2021-02-09"], dtype="datetime64[ns]"),
        np.array([5, 10], dtype="timedelta64[s]"),
        np.array([1, "a", None], dtype=object),
        np.array(3.5),
    ],
)
def test_arrays_netcdf_cannot_store_round_trip_with_dtype_and_shape(tmp_path, value):
    restored = _round_trip(tmp_path, value)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == value.dtype
    assert restored.shape == value.shape
    assert restored.tolist() == value.tolist()


def test_one_dimensional_numeric_array_is_still_stored_natively():
    encoded = encode_attrs(xr.Dataset(attrs={"bbox": np.array([1.0, 2.0, 3.0, 4.0])}))
    assert isinstance(encoded.attrs["bbox"], np.ndarray)
    assert MARKER_ATTR not in encoded.attrs


@pytest.mark.parametrize(
    "value",
    [np.datetime64("2021-02-08T09:17:09"), np.datetime64("2021-02-08T09:17:09.123456789"), np.timedelta64(5, "s")],
)
def test_numpy_time_scalars_round_trip(tmp_path, value):
    restored = _round_trip(tmp_path, value)
    assert restored == value
    assert restored.dtype == value.dtype


@pytest.mark.parametrize(
    "value",
    [
        [dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 2, 3, 4, 5)],
        (dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),),
        [dt.date(2021, 1, 1), dt.date(2021, 1, 2)],
    ],
)
def test_lists_of_datetimes_and_dates_keep_their_types(tmp_path, value):
    restored = _round_trip(tmp_path, value)
    assert restored == value
    assert type(restored) is type(value)


def test_mixed_list_with_datetimes_still_falls_back_to_json(tmp_path):
    restored = _round_trip(tmp_path, [dt.datetime(2021, 1, 1), "x"])
    assert restored == ["2021-01-01T00:00:00", "x"]


def test_pyproj_crs_round_trips(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    crs = pyproj.CRS.from_epsg(32630)
    restored = _round_trip(tmp_path, crs)
    assert isinstance(restored, pyproj.CRS)
    assert restored == crs


def test_rasterio_crs_round_trips(tmp_path):
    rasterio_crs = pytest.importorskip("rasterio.crs")
    crs = rasterio_crs.CRS.from_epsg(32630)
    restored = _round_trip(tmp_path, crs)
    assert isinstance(restored, rasterio_crs.CRS)
    assert restored == crs


def test_affine_round_trips_and_stays_readable(tmp_path):
    affine = pytest.importorskip("affine")
    transform = affine.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 7390300.0)
    restored = _round_trip(tmp_path, transform)
    assert isinstance(restored, affine.Affine)
    assert restored == transform
    stored = encode_attrs(xr.Dataset(attrs={"t": transform})).attrs["t"]
    assert list(stored) == [10.0, 0.0, 500000.0, 0.0, -10.0, 7390300.0]
