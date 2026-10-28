"""eoio.serialisation.codecs - codec registry for attribute (de)serialisation

A codec converts a single Python attribute value to something a serialised format
(netCDF, Zarr) can store, and back again. Codecs are registered under a stable name;
that name is what :mod:`eoio.serialisation.attrs` records in the ``eoio:attr_encoding``
marker so decoding is exact rather than guessed.

New types are supported by registering a codec, without modifying this module::

    from eoio.serialisation import register_codec

    class MyCodec:
        name = "my_type"

        def applies(self, value):
            return isinstance(value, MyType)

        def encode(self, value):
            return str(value)

        def decode(self, encoded):
            return MyType(encoded)

    register_codec(MyCodec(), priority=45)

Codecs are consulted in ascending priority order and the first match wins, so
priority resolves subclass ambiguity: ``bool`` is a subclass of ``int`` and
``datetime`` is a subclass of ``date``, and in both cases the more specific codec
must be offered first.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
from pathlib import PurePath
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CODEC_REGISTRY",
    "MARKER_ATTR",
    "AttrCodec",
    "get_codec",
    "is_serialisable",
    "list_codecs",
    "register_codec",
    "select_codec",
    "unregister_codec",
]


#: Attribute name holding the per-scope JSON map of attribute name -> codec name.
MARKER_ATTR = "eoio:attr_encoding"


@runtime_checkable
class AttrCodec(Protocol):
    """Protocol implemented by attribute codecs."""

    #: Stable identifier recorded in the encoding marker.
    name: str

    def applies(self, value: Any) -> bool:
        """
        Report whether this codec handles ``value``.

        :param value: Candidate attribute value.
        :returns: ``True`` when this codec can encode the value.
        """
        ...

    def encode(self, value: Any) -> Any:
        """
        Convert ``value`` to a serialisation-safe representation.

        :param value: Attribute value to encode.
        :returns: A value acceptable as a netCDF/Zarr attribute.
        """
        ...

    def decode(self, encoded: Any) -> Any:
        """
        Reverse :meth:`encode`.

        :param encoded: Previously encoded value.
        :returns: The restored Python object.
        """
        ...


#: Registered codecs, keyed by name. Ordering is maintained separately by priority.
CODEC_REGISTRY: Dict[str, AttrCodec] = {}

#: Codec name -> priority. Lower values are consulted first.
_CODEC_PRIORITY: Dict[str, int] = {}


def register_codec(codec: AttrCodec, priority: int = 100, overwrite: bool = False) -> AttrCodec:
    """
    Register a codec under its own ``name``.

    :param codec: Codec instance implementing :class:`AttrCodec`.
    :param priority: Consultation order; lower values are offered the value first.
    :param overwrite: Whether replacing an already-registered name is permitted.
    :returns: The registered codec, so it can be assigned in the same statement.
    :raises KeyError: If the name is already registered and ``overwrite`` is False.
    """
    name = codec.name
    if name in CODEC_REGISTRY and not overwrite:
        raise KeyError(
            f"Codec {name!r} already registered (existing: {type(CODEC_REGISTRY[name]).__name__}). "
            "Pass overwrite=True to replace it."
        )
    CODEC_REGISTRY[name] = codec
    _CODEC_PRIORITY[name] = priority
    return codec


def unregister_codec(name: str) -> AttrCodec:
    """
    Remove a registered codec.

    :param name: Registered codec name.
    :returns: The removed codec.
    :raises KeyError: If no codec is registered under that name.
    """
    codec = get_codec(name)
    del CODEC_REGISTRY[name]
    _CODEC_PRIORITY.pop(name, None)
    return codec


def get_codec(name: str) -> AttrCodec:
    """
    Look up a codec by name.

    :param name: Registered codec name.
    :returns: The codec.
    :raises KeyError: If no codec is registered under that name.
    """
    try:
        return CODEC_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown codec {name!r}. Registered codecs: {sorted(CODEC_REGISTRY)}") from None


def list_codecs() -> List[str]:
    """
    List registered codec names in consultation order.

    :returns: Codec names, lowest priority value first.
    """
    return sorted(CODEC_REGISTRY, key=lambda n: (_CODEC_PRIORITY[n], n))


def select_codec(value: Any) -> AttrCodec | None:
    """
    Find the codec responsible for ``value``.

    :param value: Attribute value.
    :returns: The first matching codec in priority order, or ``None`` if the value
        needs no encoding or no codec handles it.
    """
    for name in list_codecs():
        codec = CODEC_REGISTRY[name]
        if codec.applies(value):
            return codec
    return None


def is_serialisable(value: Any) -> bool:
    """
    Report whether a value can be written as an attribute without any encoding.

    Grounded in what xarray/netCDF4 actually accepts rather than what looks
    reasonable: ``bool`` and ``None`` are rejected outright by the netCDF4 backend,
    and a mixed-type list such as ``[1, "a"]`` is silently coerced to ``['1', 'a']``,
    so neither is treated as safe here.

    :param value: Attribute value.
    :returns: ``True`` when the value can be stored as-is.
    """
    # bool must be tested before int - bool is a subclass of int.
    if isinstance(value, bool) or value is None:
        return False

    if isinstance(value, (str, int, float)):
        return True

    if isinstance(value, np.generic):
        # np.timedelta64 subclasses np.signedinteger but netCDF cannot store it.
        return isinstance(value, (np.integer, np.floating, np.str_)) and not isinstance(value, np.timedelta64)

    if isinstance(value, np.ndarray):
        # netCDF attributes are 1-D; anything else is routed through the ndarray codec.
        return value.ndim == 1 and value.dtype.kind in "iufSU"

    if isinstance(value, (list, tuple)):
        # Homogeneous scalar sequences survive, but lose their list/tuple identity
        # (they come back as ndarray), so they are still routed through a codec.
        return False

    return False


class _NoneCodec:
    """Encode ``None``, which netCDF4 rejects outright."""

    name = "none"
    _SENTINEL = ""

    def applies(self, value: Any) -> bool:
        return value is None

    def encode(self, value: Any) -> str:
        return self._SENTINEL

    def decode(self, encoded: Any) -> None:
        return None


class _BoolCodec:
    """Encode Python and numpy booleans, which the netCDF4 backend refuses."""

    name = "bool"

    def applies(self, value: Any) -> bool:
        return isinstance(value, (bool, np.bool_))

    def encode(self, value: Any) -> int:
        return int(value)

    def decode(self, encoded: Any) -> bool:
        return bool(encoded)


class _DatetimeCodec:
    """Encode ``datetime`` as ISO-8601. Must precede :class:`_DateCodec`."""

    name = "datetime"

    def applies(self, value: Any) -> bool:
        return isinstance(value, dt.datetime)

    def encode(self, value: dt.datetime) -> str:
        return value.isoformat()

    def decode(self, encoded: Any) -> dt.datetime:
        return dt.datetime.fromisoformat(str(encoded))


class _DateCodec:
    """Encode ``date`` as ISO-8601."""

    name = "date"

    def applies(self, value: Any) -> bool:
        return isinstance(value, dt.date) and not isinstance(value, dt.datetime)

    def encode(self, value: dt.date) -> str:
        return value.isoformat()

    def decode(self, encoded: Any) -> dt.date:
        return dt.date.fromisoformat(str(encoded))


class _TimeCodec:
    """Encode ``time`` as ISO-8601."""

    name = "time"

    def applies(self, value: Any) -> bool:
        return isinstance(value, dt.time)

    def encode(self, value: dt.time) -> str:
        return value.isoformat()

    def decode(self, encoded: Any) -> dt.time:
        return dt.time.fromisoformat(str(encoded))


class _TimedeltaCodec:
    """Encode ``timedelta`` as total seconds."""

    name = "timedelta"

    def applies(self, value: Any) -> bool:
        return isinstance(value, dt.timedelta)

    def encode(self, value: dt.timedelta) -> float:
        return value.total_seconds()

    def decode(self, encoded: Any) -> dt.timedelta:
        return dt.timedelta(seconds=float(encoded))


class _ComplexCodec:
    """Encode complex numbers as their string repr."""

    name = "complex"

    def applies(self, value: Any) -> bool:
        return isinstance(value, (complex, np.complexfloating))

    def encode(self, value: Any) -> str:
        value = complex(value)
        return f"{value.real}+{value.imag}j"

    def decode(self, encoded: Any) -> complex:
        return complex(str(encoded).replace("+-", "-"))


class _PathCodec:
    """Encode filesystem paths as strings."""

    name = "path"

    def applies(self, value: Any) -> bool:
        return isinstance(value, PurePath)

    def encode(self, value: PurePath) -> str:
        return str(value)

    def decode(self, encoded: Any) -> PurePath:
        from pathlib import Path

        return Path(str(encoded))


class _StrCodec:
    """
    Identity codec used to pin a string that sniffing would otherwise convert.

    Never selected automatically - ``applies`` always returns ``False``. It exists so
    that a string which happens to be valid JSON, or to look like an ISO-8601 datetime,
    can be recorded in the marker as genuinely a string and left alone on decode.
    """

    name = "str"

    def applies(self, value: Any) -> bool:
        return False

    def encode(self, value: Any) -> str:
        return str(value)

    def decode(self, encoded: Any) -> str:
        return str(encoded)


class _BytesCodec:
    """Decode byte strings to text, as produced by some XML metadata parsers."""

    name = "bytes"

    def applies(self, value: Any) -> bool:
        return isinstance(value, (bytes, bytearray))

    def encode(self, value: Any) -> str:
        return bytes(value).decode("utf-8", errors="replace")

    def decode(self, encoded: Any) -> bytes:
        return str(encoded).encode("utf-8")


class _ShapelyCodec:
    """
    Encode shapely geometries as WKT.

    Detected by duck typing (``__geo_interface__`` plus ``wkt``) so that importing
    this module never imports shapely; the import happens only when a geometry is
    actually decoded.
    """

    name = "wkt"

    def applies(self, value: Any) -> bool:
        return hasattr(value, "__geo_interface__") and hasattr(value, "wkt")

    def encode(self, value: Any) -> str:
        return str(value.wkt)

    def decode(self, encoded: Any) -> Any:
        from shapely import wkt  # type: ignore[import-untyped]

        return wkt.loads(str(encoded))


class _SetCodec:
    """Encode sets as a JSON list, preserving set identity through the marker."""

    name = "set"

    def applies(self, value: Any) -> bool:
        return isinstance(value, (set, frozenset))

    def encode(self, value: Any) -> str:
        return json.dumps(sorted(value, key=repr))

    def decode(self, encoded: Any) -> set:
        if isinstance(encoded, str):
            return set(json.loads(encoded))
        return set(np.asarray(encoded).tolist())


class _NumpyDatetimeCodec:
    """Encode ``np.datetime64`` as an ISO-8601 string; the unit survives in its precision."""

    name = "datetime64"

    def applies(self, value: Any) -> bool:
        return isinstance(value, np.datetime64)

    def encode(self, value: np.datetime64) -> str:
        return str(np.datetime_as_string(value))

    def decode(self, encoded: Any) -> np.datetime64:
        return np.datetime64(str(encoded))


class _NumpyTimedeltaCodec:
    """Encode ``np.timedelta64`` as ``"<count> <unit>"``, e.g. ``"5 s"``."""

    name = "timedelta64"

    def applies(self, value: Any) -> bool:
        return isinstance(value, np.timedelta64)

    def encode(self, value: np.timedelta64) -> str:
        unit, _ = np.datetime_data(value.dtype)
        return f"{int(value.astype('int64'))} {unit}"

    def decode(self, encoded: Any) -> np.timedelta64:
        count, unit = str(encoded).split()
        return np.timedelta64(int(count), unit)  # type: ignore[call-overload]


def _ndarray_to_json_data(value: np.ndarray) -> Any:
    """
    Convert an array's contents to nested-free JSON data (flat, C order).

    :param value: Array to convert.
    :returns: A flat list of JSON-compatible items.
    """
    kind = value.dtype.kind
    flat = value.ravel()
    if kind == "M":
        return [str(x) for x in np.datetime_as_string(flat)]
    if kind == "m":
        return flat.astype("int64").tolist()
    if kind == "c":
        return [[float(x.real), float(x.imag)] for x in flat]
    if kind == "O":
        return list(flat)
    return flat.tolist()


class _NdarrayCodec:
    """
    Encode arrays netCDF cannot store as an attribute: multi-dimensional, 0-d, boolean,
    complex, object, ``datetime64`` and ``timedelta64`` arrays.

    The array is written as a JSON string holding its dtype, shape and flat contents, so
    the round trip restores the original dtype and shape. Plain 1-D numeric and string
    arrays never reach this codec - see :func:`is_serialisable`.
    """

    name = "ndarray"

    def applies(self, value: Any) -> bool:
        return isinstance(value, np.ndarray)

    def encode(self, value: np.ndarray) -> str:
        return json.dumps(
            {"dtype": value.dtype.str, "shape": list(value.shape), "data": _ndarray_to_json_data(value)},
            default=_json_default,
        )

    def decode(self, encoded: Any) -> np.ndarray:
        payload = json.loads(str(encoded))
        dtype = np.dtype(payload["dtype"])
        data = payload["data"]
        if dtype.kind == "c":
            data = [complex(re, im) for re, im in data]
        if dtype.kind == "O":
            out = np.empty(len(data), dtype=object)
            for i, item in enumerate(data):
                out[i] = item
            return out.reshape(payload["shape"])
        return np.array(data, dtype=dtype).reshape(payload["shape"])


class _DatetimeListCodec:
    """
    Preserve lists and tuples whose elements are all ``datetime`` or all ``date``.

    Without this they fall through to the JSON catch-all and come back as ISO strings.
    """

    name = "datetime_list"

    @staticmethod
    def _element_kind(value: Any) -> str | None:
        if not isinstance(value, (list, tuple)) or not value:
            return None
        if all(isinstance(i, dt.datetime) for i in value):
            return "datetime"
        if all(isinstance(i, dt.date) and not isinstance(i, dt.datetime) for i in value):
            return "date"
        return None

    def applies(self, value: Any) -> bool:
        return self._element_kind(value) is not None

    def encode(self, value: Any) -> str:
        return json.dumps(
            {
                "container": "tuple" if isinstance(value, tuple) else "list",
                "element": self._element_kind(value),
                "values": [i.isoformat() for i in value],
            }
        )

    def decode(self, encoded: Any) -> Any:
        payload = json.loads(str(encoded))
        parse = dt.datetime.fromisoformat if payload["element"] == "datetime" else dt.date.fromisoformat
        values = [parse(i) for i in payload["values"]]
        return tuple(values) if payload["container"] == "tuple" else values


def _module_root(value: Any) -> str:
    return type(value).__module__.split(".")[0]


class _PyprojCrsCodec:
    """
    Encode ``pyproj.CRS`` as WKT2.

    Detected by module and class name so importing this module never imports pyproj;
    the import happens only when a CRS is actually decoded.
    """

    name = "crs_pyproj"

    def applies(self, value: Any) -> bool:
        return _module_root(value) == "pyproj" and type(value).__name__ == "CRS"

    def encode(self, value: Any) -> str:
        return str(value.to_wkt())

    def decode(self, encoded: Any) -> Any:
        from pyproj import CRS  # type: ignore[import-untyped]

        return CRS.from_wkt(str(encoded))


class _RasterioCrsCodec:
    """Encode ``rasterio.crs.CRS`` as WKT, restored as a rasterio CRS. Imports lazily."""

    name = "crs_rasterio"

    def applies(self, value: Any) -> bool:
        return _module_root(value) == "rasterio" and type(value).__name__ == "CRS"

    def encode(self, value: Any) -> str:
        return str(value.to_wkt())

    def decode(self, encoded: Any) -> Any:
        from rasterio.crs import CRS  # type: ignore[import-untyped]

        return CRS.from_wkt(str(encoded))


class _AffineCodec:
    """
    Encode ``affine.Affine`` as its six coefficients ``(a, b, c, d, e, f)``.

    Stored as a plain float array, so it stays readable under ``ncdump``. Imports lazily.
    """

    name = "affine"

    def applies(self, value: Any) -> bool:
        return _module_root(value) == "affine" and type(value).__name__ == "Affine"

    def encode(self, value: Any) -> list:
        return [float(c) for c in tuple(value)[:6]]

    def decode(self, encoded: Any) -> Any:
        from affine import Affine  # type: ignore[import-untyped]

        return Affine(*[float(c) for c in np.atleast_1d(encoded)])


class _NumpyScalarCodec:
    """Unwrap remaining numpy scalars to their native Python equivalents."""

    name = "numpy_scalar"

    def applies(self, value: Any) -> bool:
        return isinstance(value, np.generic)

    def encode(self, value: np.generic) -> Any:
        return value.item()

    def decode(self, encoded: Any) -> Any:
        return encoded


def _is_homogeneous_scalar_sequence(value: Any) -> bool:
    """
    Report whether a sequence is safe to store without JSON encoding.

    :param value: Sequence to test.
    :returns: ``True`` when every element is a scalar of a single kind. Mixed
        sequences are excluded because netCDF silently coerces them to strings.
    """
    items = list(value)
    if not items:
        return True
    if any(isinstance(i, bool) for i in items):
        return False
    if all(isinstance(i, str) for i in items):
        return True
    return all(isinstance(i, (int, float, np.integer, np.floating)) for i in items)


class _ListCodec:
    """
    Preserve list identity for homogeneous scalar lists.

    netCDF returns such attributes as an ``ndarray``; the marker lets them come back
    as a ``list``, including the empty-list case which otherwise returns as an empty
    float64 array.
    """

    name = "list"

    def applies(self, value: Any) -> bool:
        return isinstance(value, list) and _is_homogeneous_scalar_sequence(value)

    def encode(self, value: list) -> Any:
        return list(value)

    def decode(self, encoded: Any) -> list:
        if isinstance(encoded, np.ndarray):
            return encoded.tolist()
        if isinstance(encoded, (list, tuple)):
            return list(encoded)
        return [encoded]


class _TupleCodec:
    """As :class:`_ListCodec`, but restoring a tuple."""

    name = "tuple"

    def applies(self, value: Any) -> bool:
        return isinstance(value, tuple) and _is_homogeneous_scalar_sequence(value)

    def encode(self, value: tuple) -> Any:
        return list(value)

    def decode(self, encoded: Any) -> tuple:
        if isinstance(encoded, np.ndarray):
            return tuple(encoded.tolist())
        if isinstance(encoded, (list, tuple)):
            return tuple(encoded)
        return (encoded,)


class _JsonCodec:
    """
    Catch-all for structured values: mixed sequences, lists of dicts, and any dict
    that cannot be dot-flattened.

    This is the codec behind ``eoio:processing_steps`` (a list of dicts) produced by
    :func:`eoio.read_multi`.
    """

    name = "json"

    def applies(self, value: Any) -> bool:
        return isinstance(value, (list, tuple, dict))

    def encode(self, value: Any) -> str:
        if _has_non_string_keys(value):
            warnings.warn(
                "Dict has non-string keys; JSON encoding will return them as strings "
                "(e.g. {1: 'x'} decodes as {'1': 'x'}).",
                stacklevel=2,
            )
        return json.dumps(value, default=_json_default)

    def decode(self, encoded: Any) -> Any:
        if isinstance(encoded, str):
            return json.loads(encoded)
        return encoded


def _has_non_string_keys(value: Any) -> bool:
    """
    Check recursively for dict keys JSON cannot represent faithfully.

    :param value: Structure to check.
    :returns: ``True`` when any nested dict has a non-string key.
    """
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            return True
        return any(_has_non_string_keys(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_string_keys(v) for v in value)
    return False


def _json_default(value: Any) -> Any:
    """
    Serialise values JSON cannot handle natively, by delegating to their codec.

    :param value: Value encountered inside a structure being JSON-encoded.
    :returns: A JSON-serialisable representation.
    """
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        return value.tolist()
    codec = select_codec(value)
    if codec is not None and codec.name != _JsonCodec.name:
        return codec.encode(value)
    return str(value)


# Registration order encodes the subclass relationships that matter: bool before the
# numeric codecs, datetime before date, and the JSON catch-all last.
register_codec(_NoneCodec(), priority=10)
register_codec(_BoolCodec(), priority=20)
register_codec(_DatetimeCodec(), priority=30)
register_codec(_DateCodec(), priority=31)
register_codec(_TimeCodec(), priority=32)
register_codec(_TimedeltaCodec(), priority=33)
register_codec(_ComplexCodec(), priority=40)
register_codec(_StrCodec(), priority=44)
register_codec(_BytesCodec(), priority=45)
register_codec(_PathCodec(), priority=50)
register_codec(_ShapelyCodec(), priority=55)
register_codec(_PyprojCrsCodec(), priority=56)
register_codec(_RasterioCrsCodec(), priority=57)
register_codec(_AffineCodec(), priority=58)  # before tuple codecs: some Affine versions are tuples
register_codec(_SetCodec(), priority=60)
register_codec(_NumpyDatetimeCodec(), priority=65)
register_codec(_NumpyTimedeltaCodec(), priority=66)
register_codec(_NumpyScalarCodec(), priority=70)
register_codec(_NdarrayCodec(), priority=75)
register_codec(_DatetimeListCodec(), priority=79)  # before the plain list/tuple codecs
register_codec(_ListCodec(), priority=80)
register_codec(_TupleCodec(), priority=81)
register_codec(_JsonCodec(), priority=90)


if __name__ == "__main__":
    pass
