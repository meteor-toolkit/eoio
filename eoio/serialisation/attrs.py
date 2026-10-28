"""eoio.serialisation.attrs - encode and decode xarray attributes for serialisation

Readers populate ``attrs`` with rich Python objects (nested metadata dicts, datetimes,
geometries) that netCDF and Zarr cannot store. :func:`encode_attrs` converts them to
serialisable values before writing; :func:`decode_attrs` restores them after reading.

Two design points are worth knowing before using this module.

**Nested dicts are dot-flattened.** ``{"product_metadata": {"sensing_time": ...}}`` is
written as an attribute named ``product_metadata.sensing_time``, so it stays readable
under ``ncdump`` and matches the dot-path convention already used by
:func:`eoio.read_multi` and the stack processor. Dicts that cannot be flattened
unambiguously - non-string keys, keys containing a dot, or names that would collide with
an existing attribute - fall back to a single JSON blob instead.

**Decoding works on any file, not just eoio's own.** Sniffing is the primary path, so
third-party netCDF products decode on a best-effort basis. Files written by eoio also
carry an ``eoio:attr_encoding`` marker recording which codec encoded which attribute,
and when that marker is present it is preferred, making the round trip exact rather than
best-effort.

Sniffing is deliberately conservative: only JSON objects/arrays and ISO-8601 datetimes
are recognised. Numeric-looking strings are left alone, because coercing them corrupts
legitimate values - the controlled vocabulary's own ``product_version = "03.01"`` would
otherwise come back as the float ``3.01``.

Known limitations, none of which lose information:

* Plain Python scalars widen through netCDF itself - an ``int`` attribute is returned as
  ``np.int64`` and a ``float`` as ``np.float64``, whatever this module does. Values
  compare equal, but ``type()`` differs. Dot-flattened dict leaves are affected;
  ``dict_style="json"`` leaves are not, since they travel inside a JSON string.
* ``np.bool_`` is restored as a Python ``bool``.
* Dicts with non-string keys cannot be dot-flattened and fall back to JSON, which
  returns the keys as strings. This is warned about at encode time.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import warnings
from typing import Any, Callable, Dict, Mapping, Tuple, TypeVar, Union, cast

import xarray as xr

from eoio.serialisation.codecs import MARKER_ATTR, get_codec, is_serialisable, select_codec

__all__ = ["decode_attrs", "encode_attrs"]


#: Marker value recording that an attribute was dot-flattened from a nested dict.
DICT_DOT = "dict_dot"

#: Matches ISO-8601 values that include a time component. A bare date is deliberately
#: not sniffed - date-like strings are common as genuine string metadata.
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")

#: Bound rather than value-restricted, so that callers holding a union - a writer with a
#: ``Dataset | DataTree``, for example - can pass it without the type checker objecting.
T = TypeVar("T", bound=Union[xr.Dataset, xr.DataArray, xr.DataTree])


# -------------------------
# Public API
# -------------------------


def encode_attrs(
    obj: T,
    *,
    dict_style: str = "dot",
    strict: bool = False,
    inplace: bool = False,
) -> T:
    """
    Convert attributes to values that can be written to netCDF or Zarr.

    Applies to dataset-level, variable and coordinate attributes, and to every node of a
    ``DataTree``. An ``eoio:attr_encoding`` marker is added to each scope that needed
    encoding, so :func:`decode_attrs` can reverse it exactly.

    :param obj: Dataset, DataArray or DataTree whose attributes should be encoded.
    :param dict_style: ``"dot"`` to dot-flatten nested dicts, or ``"json"`` to encode
        each as a single JSON blob.
    :param strict: If ``True``, raise on a value no codec handles. If ``False``, fall
        back to ``str()`` with a warning.
    :param inplace: Whether to modify ``obj`` rather than a copy.
    :returns: The object with encoded attributes.
    :raises ValueError: If ``dict_style`` is not ``"dot"`` or ``"json"``.
    :raises TypeError: If ``strict`` is set and a value cannot be encoded.
    """
    if dict_style not in ("dot", "json"):
        raise ValueError(f"dict_style must be 'dot' or 'json', got {dict_style!r}")

    return _walk(obj, lambda attrs: _encode_scope(attrs, dict_style=dict_style, strict=strict), inplace=inplace)


def decode_attrs(obj: T, *, sniff: bool = True, inplace: bool = False) -> T:
    """
    Restore attributes previously encoded for serialisation.

    Where an ``eoio:attr_encoding`` marker is present its codecs are used, giving an
    exact round trip. Attributes with no marker entry are sniffed instead, recognising
    only JSON objects/arrays and ISO-8601 datetimes.

    :param obj: Dataset, DataArray or DataTree whose attributes should be decoded.
    :param sniff: Whether to attempt best-effort decoding of unmarked attributes. Set
        ``False`` to decode only what the marker describes.
    :param inplace: Whether to modify ``obj`` rather than a copy.
    :returns: The object with decoded attributes, and the marker attribute removed.
    """
    return _walk(obj, lambda attrs: _decode_scope(attrs, sniff=sniff), inplace=inplace)


# -------------------------
# Scope-level encode/decode
# -------------------------


def _encode_scope(attrs: Mapping[str, Any], *, dict_style: str, strict: bool) -> Dict[str, Any]:
    """
    Encode one attribute scope (a dataset, a variable, or a DataTree node).

    :param attrs: Attributes to encode.
    :param dict_style: ``"dot"`` or ``"json"``, see :func:`encode_attrs`.
    :param strict: Whether unencodable values raise rather than warn.
    :returns: New attribute mapping, including the encoding marker where needed.
    """
    previous_marker = _read_marker(attrs)
    out: Dict[str, Any] = {}
    marker: Dict[str, str] = {}

    for key, value in attrs.items():
        if key == MARKER_ATTR:
            continue

        if dict_style == "dot" and isinstance(value, dict) and _can_flatten(value, key, attrs):
            for leaf_key, leaf_value in _flatten_dict(value, key).items():
                encoded, codec_name = _encode_value(leaf_value, strict=strict, attr_name=leaf_key)
                out[leaf_key] = encoded
                codec_name = _resolve_codec_name(codec_name, leaf_key, encoded, previous_marker)
                if codec_name is not None:
                    marker[leaf_key] = codec_name
            marker[key] = DICT_DOT
            continue

        encoded, codec_name = _encode_value(value, strict=strict, attr_name=key)
        out[key] = encoded
        codec_name = _resolve_codec_name(codec_name, key, encoded, previous_marker)
        if codec_name is not None:
            marker[key] = codec_name

    # Re-encoding an already-encoded object finds nothing left to do, which would drop
    # the original marker and leave the data undecodable. Carry forward any entry that
    # still refers to an attribute we have.
    for key, codec_name in previous_marker.items():
        if key in out or codec_name == DICT_DOT:
            marker.setdefault(key, codec_name)

    if marker:
        out[MARKER_ATTR] = json.dumps(marker, sort_keys=True)

    return out


def _decode_scope(attrs: Mapping[str, Any], *, sniff: bool) -> Dict[str, Any]:
    """
    Decode one attribute scope.

    :param attrs: Attributes to decode.
    :param sniff: Whether to best-effort decode attributes absent from the marker.
    :returns: New attribute mapping, without the encoding marker.
    """
    marker = _read_marker(attrs)
    out: Dict[str, Any] = {}

    for key, value in attrs.items():
        if key == MARKER_ATTR:
            continue

        codec_name = marker.get(key)

        if codec_name == DICT_DOT:
            # Placeholder only - the flattened leaves are re-nested below.
            continue

        if codec_name is not None:
            try:
                out[key] = get_codec(codec_name).decode(value)
            except Exception as exc:  # noqa: BLE001 - a bad marker must not lose the value
                warnings.warn(
                    f"Could not decode attribute {key!r} with codec {codec_name!r} ({exc}); "
                    "leaving the encoded value in place.",
                    stacklevel=2,
                )
                out[key] = value
            continue

        out[key] = _sniff_value(value) if sniff else value

    for key, codec_name in marker.items():
        if codec_name != DICT_DOT:
            continue
        prefix = f"{key}."
        collected = {name: out.pop(name) for name in list(out) if name.startswith(prefix)}
        if collected:
            out[key] = _unflatten(collected, prefix)

    return out


def _encode_value(value: Any, *, strict: bool, attr_name: str) -> Tuple[Any, str | None]:
    """
    Encode a single attribute value.

    :param value: Value to encode.
    :param strict: Whether an unencodable value raises rather than warns.
    :param attr_name: Attribute name, used only for error messages.
    :returns: Tuple of the encoded value and the codec name, or ``None`` when the value
        needed no encoding.
    :raises TypeError: If ``strict`` and no codec handles the value.
    """
    if is_serialisable(value):
        return value, None

    codec = select_codec(value)
    if codec is None:
        message = f"No codec for attribute {attr_name!r} of type {type(value).__name__}"
        if strict:
            raise TypeError(f"{message}. Register one with eoio.serialisation.register_codec().")
        warnings.warn(f"{message}; falling back to str(). It will not be restored on decode.", stacklevel=2)
        return str(value), None

    return codec.encode(value), codec.name


def _resolve_codec_name(
    codec_name: str | None,
    key: str,
    encoded: Any,
    previous_marker: Mapping[str, str],
) -> str | None:
    """
    Decide what the marker should record for an attribute.

    Precedence matters when re-encoding an already-encoded object: a codec that just ran
    wins, then whatever the previous marker said, and only then the ambiguous-string
    pin. Without that order, re-encoding a datetime that is already an ISO string would
    pin it as a string and lose the original type.

    :param codec_name: Codec that encoded the value this pass, if any.
    :param key: Attribute name.
    :param encoded: The encoded value.
    :param previous_marker: Marker found on the object before encoding.
    :returns: Codec name to record, or ``None`` to leave the attribute unmarked.
    """
    if codec_name is not None:
        return codec_name
    return previous_marker.get(key) or _pin_if_ambiguous(encoded)


def _pin_if_ambiguous(encoded: Any) -> str | None:
    """
    Decide whether a string needs pinning as a string in the marker.

    A value that needed no encoding is normally left out of the marker. That is unsafe
    for a string sniffing would convert - valid JSON, or an ISO-8601 datetime - because
    decoding would then change its type. Recording it as ``"str"`` keeps the round trip
    exact without affecting anything else.

    :param encoded: The already-encoded value.
    :returns: ``"str"`` when the value must be pinned, otherwise ``None``.
    """
    if isinstance(encoded, str) and not isinstance(_sniff_value(encoded), str):
        return "str"
    return None


def _sniff_value(value: Any) -> Any:
    """
    Best-effort decode of an unmarked attribute.

    Only JSON objects/arrays and ISO-8601 datetimes are recognised - see the module
    docstring for why this is deliberately narrow.

    :param value: Encoded attribute value.
    :returns: The decoded value, or the input unchanged.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()

    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return value
        return parsed if isinstance(parsed, (dict, list)) else value

    if _ISO_DATETIME_RE.match(text):
        try:
            # fromisoformat handles a trailing "Z" natively on the Python versions
            # this package supports (>=3.11).
            return dt.datetime.fromisoformat(text)
        except ValueError:
            return value

    return value


# -------------------------
# Dict flattening
# -------------------------


def _can_flatten(value: Mapping[str, Any], key: str, attrs: Mapping[str, Any]) -> bool:
    """
    Report whether a dict can be dot-flattened without ambiguity.

    :param value: The dict attribute.
    :param key: Its attribute name, which becomes the flattened prefix.
    :param attrs: The full scope, checked for name collisions.
    :returns: ``True`` when flattening is unambiguous and reversible.
    """
    if not value:
        # Flattening an empty dict would emit nothing and lose the attribute.
        return False

    if not _keys_are_flattenable(value):
        return False

    flattened = set(_flatten_dict(value, key))
    return not (flattened & (set(attrs) - {key}))


def _keys_are_flattenable(value: Any) -> bool:
    """
    Check recursively that every key is a dot-free string.

    :param value: Dict (or leaf) to check.
    :returns: ``True`` when the whole structure can be addressed by dot-paths.
    """
    if not isinstance(value, dict):
        return True
    for key, sub in value.items():
        if not isinstance(key, str) or "." in key:
            return False
        if not _keys_are_flattenable(sub):
            return False
    return True


def _flatten_dict(value: Mapping[str, Any], parent: str) -> Dict[str, Any]:
    """
    Flatten a nested dict to dot-path attribute names.

    :param value: Dict to flatten.
    :param parent: Prefix for generated names.
    :returns: Mapping of dot-path name to leaf value. Empty dicts are kept as leaves.
    """
    flat: Dict[str, Any] = {}
    for key, sub in value.items():
        name = f"{parent}.{key}"
        if isinstance(sub, dict) and sub:
            flat.update(_flatten_dict(sub, name))
        else:
            flat[name] = sub
    return flat


def _unflatten(collected: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Rebuild a nested dict from dot-path attribute names.

    :param collected: Mapping of full dot-path name to value.
    :param prefix: Prefix to strip, including the trailing dot.
    :returns: The re-nested dict.
    """
    nested: Dict[str, Any] = {}
    for name, value in collected.items():
        parts = name[len(prefix) :].split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def _read_marker(attrs: Mapping[str, Any]) -> Dict[str, str]:
    """
    Read and parse the encoding marker from a scope.

    :param attrs: Attribute mapping.
    :returns: Mapping of attribute name to codec name; empty when absent or unreadable.
    """
    raw = attrs.get(MARKER_ATTR)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        warnings.warn(f"Ignoring unreadable {MARKER_ATTR!r} attribute; falling back to sniffing.", stacklevel=2)
        return {}
    return parsed if isinstance(parsed, dict) else {}


# -------------------------
# Object walking
# -------------------------


def _walk(obj: T, transform: Callable[[Mapping[str, Any]], Dict[str, Any]], *, inplace: bool) -> T:
    """
    Apply a scope transform to every attribute scope of an xarray object.

    :param obj: Dataset, DataArray or DataTree.
    :param transform: Callable applied to each scope's attributes.
    :param inplace: Whether to modify ``obj`` rather than a copy.
    :returns: The transformed object.
    :raises TypeError: If ``obj`` is not a supported xarray type.
    """
    # The casts below are needed because T is bound rather than value-restricted: each
    # branch narrows to a concrete type that the checker cannot see is the same T that
    # came in. The isinstance checks are what actually guarantee it.
    if isinstance(obj, xr.DataTree):
        tree = obj if inplace else obj.copy()
        for node in tree.subtree:
            node.dataset = _walk_dataset(node.to_dataset(), transform, inplace=True)
        return cast(T, tree)

    if isinstance(obj, xr.Dataset):
        return cast(T, _walk_dataset(obj if inplace else obj.copy(), transform, inplace=True))

    if isinstance(obj, xr.DataArray):
        out = obj if inplace else obj.copy()
        out.attrs = transform(out.attrs)
        for name in out.coords:
            out.coords[name].attrs = transform(out.coords[name].attrs)
        return cast(T, out)

    raise TypeError(f"Expected an xarray Dataset, DataArray or DataTree, got {type(obj).__name__}")


def _walk_dataset(
    ds: xr.Dataset,
    transform: Callable[[Mapping[str, Any]], Dict[str, Any]],
    *,
    inplace: bool,
) -> xr.Dataset:
    """
    Apply a scope transform to a dataset and every one of its variables.

    :param ds: Dataset to transform.
    :param transform: Callable applied to each scope's attributes.
    :param inplace: Whether to modify ``ds`` rather than a copy.
    :returns: The transformed dataset.
    """
    out = ds if inplace else ds.copy()
    out.attrs = transform(out.attrs)
    # Iterate Variable objects directly: these are the objects the dataset holds, so
    # assigning their attrs updates the dataset itself, for coordinates as well as data
    # variables.
    for variable in out.variables.values():
        variable.attrs = transform(variable.attrs)
    return out
