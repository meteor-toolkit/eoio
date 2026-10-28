"""Tests for eoio.serialisation.codecs"""

import datetime as dt
from pathlib import Path, PurePath

import numpy as np
import pytest

from eoio.serialisation.codecs import (
    CODEC_REGISTRY,
    get_codec,
    is_serialisable,
    list_codecs,
    register_codec,
    select_codec,
    unregister_codec,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class _DummyType:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DummyType) and other.payload == self.payload


class _DummyCodec:
    name = "dummy"

    def applies(self, value):
        return isinstance(value, _DummyType)

    def encode(self, value):
        return f"dummy:{value.payload}"

    def decode(self, encoded):
        return _DummyType(str(encoded).split(":", 1)[1])


@pytest.fixture
def dummy_codec():
    """Register a third-party codec and clean it up afterwards."""
    codec = register_codec(_DummyCodec(), priority=5)
    yield codec
    unregister_codec(codec.name)


def test_register_codec_extends_the_registry_without_core_changes(dummy_codec):
    assert select_codec(_DummyType("x")) is dummy_codec
    assert get_codec("dummy") is dummy_codec
    assert "dummy" in list_codecs()


def test_registering_a_duplicate_name_raises():
    with pytest.raises(KeyError, match="already registered"):
        register_codec(_DummyCodec(), priority=5)
        register_codec(_DummyCodec(), priority=5)
    unregister_codec("dummy")


def test_register_codec_overwrite_replaces():
    first = register_codec(_DummyCodec(), priority=5)
    second = register_codec(_DummyCodec(), priority=5, overwrite=True)
    assert get_codec("dummy") is second
    assert get_codec("dummy") is not first
    unregister_codec("dummy")


def test_get_codec_reports_available_names_when_unknown():
    with pytest.raises(KeyError, match="Unknown codec"):
        get_codec("no_such_codec")


def test_list_codecs_is_in_priority_order():
    names = list_codecs()
    # bool must precede the numeric codecs, and datetime must precede date, or the
    # subclass relationships resolve to the wrong codec.
    assert names.index("bool") < names.index("numpy_scalar")
    assert names.index("datetime") < names.index("date")
    assert names.index("json") == len(names) - 1


def test_unregister_codec_removes_it():
    register_codec(_DummyCodec(), priority=5)
    unregister_codec("dummy")
    assert "dummy" not in CODEC_REGISTRY
    with pytest.raises(KeyError):
        get_codec("dummy")


# ---------------------------------------------------------------------------
# is_serialisable - grounded in what xarray/netCDF4 actually accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["text", 42, 3.14, np.int64(7), np.float32(1.5), np.array([1, 2, 3]), np.array(["a", "b"])],
)
def test_is_serialisable_accepts_values_netcdf_can_store(value):
    assert is_serialisable(value)


@pytest.mark.parametrize(
    "value",
    [
        True,  # netCDF4 rejects bool outright
        np.bool_(True),
        None,  # netCDF4 rejects None outright
        [1, 2, 3],  # returned as ndarray, so routed through a codec to stay a list
        (1, 2),
        [1, "a"],  # silently coerced to ['1', 'a'] if passed through
        {"a": 1},
        dt.datetime(2023, 7, 2),
        Path("/tmp/x"),
        {1, 2},
        3 + 4j,
    ],
)
def test_is_serialisable_rejects_values_needing_encoding(value):
    assert not is_serialisable(value)


def test_object_dtype_arrays_are_not_serialisable():
    assert not is_serialisable(np.array([{"a": 1}], dtype=object))


# ---------------------------------------------------------------------------
# Codec selection resolves subclass ambiguity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "none"),
        (True, "bool"),  # bool is a subclass of int
        (np.bool_(False), "bool"),
        (dt.datetime(2023, 7, 2, 10, 0), "datetime"),  # datetime is a subclass of date
        (dt.date(2023, 7, 2), "date"),
        (dt.time(10, 0), "time"),
        (dt.timedelta(hours=1), "timedelta"),
        (3 + 4j, "complex"),
        (b"bytes", "bytes"),
        (Path("/tmp/x"), "path"),
        ({1, 2}, "set"),
        (np.int64(3), "numpy_scalar"),
        ([1, 2], "list"),
        ((1, 2), "tuple"),
        ([1, "a"], "json"),  # mixed sequences must not be passed through
        ([{"a": 1}], "json"),
        ({"a": 1}, "json"),
    ],
)
def test_select_codec_picks_the_expected_codec(value, expected):
    codec = select_codec(value)
    assert codec is not None
    assert codec.name == expected


def test_select_codec_returns_none_for_unknown_types():
    assert select_codec(object()) is None


# ---------------------------------------------------------------------------
# Individual codec round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        dt.datetime(2023, 7, 2, 10, 56, 19),
        dt.datetime(2023, 7, 2, 10, 56, 19, tzinfo=dt.timezone.utc),
        dt.date(2023, 7, 2),
        dt.time(10, 56, 19),
        dt.timedelta(hours=2, minutes=30),
        3 + 4j,
        b"some bytes",
        Path("/tmp/product.SAFE"),
        {1, 2, 3},
        [1.0, 2.0],
        [],
        (1, 2),
        [1, "a"],
        [{"processor": "read_multi", "n_files": 4}],
        {"a": {"b": 1}},
    ],
)
def test_codec_round_trip(value):
    codec = select_codec(value)
    assert codec is not None
    assert codec.decode(codec.encode(value)) == value


def test_complex_codec_handles_negative_imaginary_part():
    codec = get_codec("complex")
    assert codec.decode(codec.encode(3 - 4j)) == 3 - 4j


def test_numpy_bool_round_trips_by_value_not_type():
    codec = select_codec(np.bool_(True))
    restored = codec.decode(codec.encode(np.bool_(True)))
    assert restored is True
    assert isinstance(restored, bool)


def test_shapely_codec_is_selected_by_duck_typing():
    shapely_geometry = pytest.importorskip("shapely.geometry")
    geom = shapely_geometry.box(0, 0, 1, 1)
    codec = select_codec(geom)
    assert codec.name == "wkt"
    assert codec.decode(codec.encode(geom)).equals(geom)


def test_path_codec_returns_a_path():
    codec = get_codec("path")
    assert isinstance(codec.decode(codec.encode(Path("/tmp/x"))), PurePath)


def test_json_codec_warns_on_non_string_keys():
    codec = get_codec("json")
    with pytest.warns(UserWarning, match="non-string keys"):
        encoded = codec.encode({1: "x"})
    # JSON genuinely cannot represent integer keys, so this is lossy by nature.
    assert codec.decode(encoded) == {"1": "x"}


def test_json_codec_encodes_nested_unserialisable_values_via_their_codecs():
    codec = get_codec("json")
    encoded = codec.encode([{"when": dt.datetime(2023, 7, 2, 10, 0)}])
    assert codec.decode(encoded) == [{"when": "2023-07-02T10:00:00"}]
