"""eoio.writers.zarr.writer - Zarr writer

Wraps ``to_zarr`` with the parts eoio needs on top of it: attribute encoding (handled by
:class:`~eoio.writers.base.BaseWriter`), explicit chunking, and appending along a
dimension as a first-class operation rather than something the caller assembles.

Unlike the netCDF writer, there are no compression defaults to add - zarr compresses by
default, and does it well (a 200k-element float array occupies about 2 kB rather than
800 kB). Pass ``encoding`` to change or disable that.

A ``DataTree`` is written as zarr groups, one per node. xarray implements neither
``append_dim`` nor ``group`` for a tree, so :meth:`ZarrWriter.can_write` refuses a tree
combined with either - at the first write rather than the second, since a first append
is an ordinary write and would otherwise succeed.

Appending is a mutation
-----------------------
An append is not a fresh write that happens to extend an array: it changes an artifact
that already exists, so what is already in the store has to be reconciled with what is
arriving. Two things follow, and both are handled here rather than left to the caller.

*Coordinates.* ``to_zarr`` extends a dimension unconditionally, so re-running a write
that partly succeeded appends the same time steps a second time. Nothing downstream can
tell the duplicates apart, and the artifact is usually permanent, so ``on_overlap``
governs it explicitly and defaults to refusing.

*Attributes.* Zarr replaces a group's attributes wholesale on every write, so an append
would otherwise leave the store carrying only the last-appended object's metadata - and
with ``update_history`` on by default, a stack's ``history`` would record the most recent
append instead of accumulating. Appends therefore merge into what the store already
holds instead of overwriting it.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from eoio.serialisation.codecs import MARKER_ATTR
from eoio.writers.base import BaseWriter, WritableObject, WriteCapability
from eoio.writers.factory import register_writer

__all__ = ["ZarrWriter"]


#: Attributes whose values accumulate across appends rather than being replaced.
HISTORY_ATTR = "history"
STEPS_ATTR = "eoio:processing_steps"


@register_writer("zarr")
class ZarrWriter(BaseWriter):
    """Write a Dataset or DataTree to a Zarr store."""

    default_write_params: Dict[str, Any] = {
        **BaseWriter.default_write_params,
        "mode": "w",
        "append_dim": None,
        "on_overlap": "error",
        "chunks": None,
        "consolidated": True,
        "zarr_format": None,
        "group": None,
        "encoding": None,
    }

    all_write_params: Dict[str, str] = {
        **BaseWriter.all_write_params,
        "mode": "w/w-/a/a-/r+",
        "append_dim": "dimension name to append along, or None (not supported for a DataTree)",
        "on_overlap": "error/skip/allow - what to do when appending coordinates the store already has",
        "chunks": "mapping of dimension name to chunk size, or None to leave chunking alone",
        "consolidated": "True/False",
        "zarr_format": "2/3, or None for the zarr default",
        "group": "group path within the store, or None for the root (not supported for a DataTree)",
        "encoding": "per-variable encoding dict, or None. Ignored when appending",
    }

    @staticmethod
    def get_extension() -> str:
        return ".zarr"

    def can_write(self, obj: Any) -> WriteCapability:
        """
        Report whether this writer can express ``obj`` with its current parameters.

        :param obj: Candidate object.
        :returns: Capability, with a reason when the answer is no.
        """
        capability = super().can_write(obj)
        if not capability.ok or not isinstance(obj, xr.DataTree):
            return capability

        # Both are accepted by DataTree.to_zarr's signature but raise NotImplementedError.
        params = self.config.write_params
        unsupported = [name for name in ("append_dim", "group") if params[name] is not None]
        if unsupported:
            return WriteCapability(
                False,
                f"xarray cannot write a DataTree with {' or '.join(unsupported)}; "
                "its own structure is the groups, and appending to a tree is not implemented",
            )
        return capability

    def write_dataset(self, obj: WritableObject, path: Path) -> None:
        """
        Write the object to a Zarr store.

        :param obj: Dataset or DataTree, already encoded by the base class.
        :param path: Destination store path.
        :raises ImportError: If the optional ``zarr`` package is not installed.
        """
        if importlib.util.find_spec("zarr") is None:
            raise ImportError(
                "The zarr writer needs the optional zarr package. Install it with: pip install eoio[zarr]"
            )

        params = self.config.write_params

        if params["chunks"]:
            obj = obj.chunk(params["chunks"])

        if isinstance(obj, xr.Dataset):
            obj = self._fill_values_to_encoding(obj)

        mode, append_dim = self._resolve_mode(path)

        if append_dim is not None:
            existing_coord, existing_attrs, existing_dims = self._read_store_state(path, append_dim)

            self._check_variables_match(obj, path, append_dim, existing_dims)
            trimmed = self._resolve_overlap(obj, path, append_dim, existing_coord)
            if trimmed is None:
                # Everything on the way in is already in the store; writing nothing is
                # what makes a re-run idempotent rather than merely non-duplicating.
                return
            obj = self._merge_store_attrs(trimmed, existing_attrs)

        kwargs: Dict[str, Any] = {"mode": mode, "consolidated": params["consolidated"]}

        if append_dim is not None:
            kwargs["append_dim"] = append_dim

        if params["group"] is not None:
            kwargs["group"] = params["group"]

        if params["zarr_format"] is not None:
            kwargs["zarr_format"] = params["zarr_format"]

        encoding = self._resolve_encoding(append_dim)
        obj, encoding = self._with_append_safe_time(obj, encoding, creating=mode != "a" or append_dim is None)
        if encoding:
            kwargs["encoding"] = encoding

        obj.to_zarr(path, **kwargs)

    # ---- mode and encoding ----

    @staticmethod
    def _fill_values_to_encoding(obj: xr.Dataset) -> xr.Dataset:
        """
        Move ``_FillValue`` from variable attributes to variable encoding.

        Readers commonly leave ``_FillValue`` in ``attrs``, and a fresh write accepts
        that. An append does not: xarray reloads the stored variable's encoding, which
        carries the fill value, finds the same key in ``attrs`` and refuses with "failed
        to prevent overwriting existing key _FillValue". Writing it as encoding on every
        write, first included, stores the same fill value and keeps the first write and
        the appends consistent. An explicit encoding already on the variable wins.

        :param obj: Dataset about to be written.
        :returns: A copy with the fill values moved; the input is not modified.
        """
        if not any("_FillValue" in var.attrs for var in obj.variables.values()):
            return obj

        out = obj.copy()
        for var in out.variables.values():
            if "_FillValue" not in var.attrs:
                continue
            attrs = dict(var.attrs)
            fill = attrs.pop("_FillValue")
            var.attrs = attrs
            var.encoding = {"_FillValue": fill, **var.encoding}
        return out

    def _resolve_mode(self, path: Path) -> Tuple[str, Optional[str]]:
        """
        Work out the mode and append dimension for this write.

        ``append_dim`` is a complete instruction on its own, and determines the mode:

        * the store does not exist yet - create it, and do not append. Appending to a
          store that is not there fails, because there is no existing dimension to
          extend. Handling it here is what lets the same call serve both the first and
          every later time step of a growing series, rather than making the caller
          special-case the first one.
        * the store exists but not the requested ``group`` - add the group, without
          touching the rest of the store, for the same reason as the first case.
        * the store and group exist - append to it, which requires ``mode="a"``.

        An explicitly supplied ``mode`` is therefore only meaningful when not appending.
        Asking for both is a contradiction - xarray rejects the combination outright -
        and silently resolving it either way would do something other than what was
        asked, so it raises instead. The default ``mode`` is not treated as a request.

        :param path: Destination store path.
        :returns: Tuple of the mode and the append dimension, the latter ``None`` when
            not appending.
        :raises ValueError: If ``mode`` was supplied alongside ``append_dim``.
        """
        params = self.config.write_params
        append_dim = params["append_dim"]

        if append_dim is None:
            return params["mode"], None

        if "mode" in self.config.provided_write_params and params["mode"] != "a":
            raise ValueError(
                f"mode={params['mode']!r} contradicts append_dim={append_dim!r}: appending requires "
                "mode='a', which append_dim already selects. Drop mode to append, or drop "
                "append_dim to write the store afresh."
            )

        if not path.exists():
            return "w", None

        # xarray cannot append to a group the store does not have yet, but adding one is
        # an ordinary write - in mode "a", since "w" would replace the whole store.
        group = params["group"]
        if group is not None and self._group_absent(path, group):
            return "a", None

        return "a", append_dim

    def _resolve_encoding(self, append_dim: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Work out the ``encoding`` argument for this write.

        Encoding cannot be supplied when appending: zarr rejects it with "variable
        already exists, but encoding was provided", because the variable's encoding was
        fixed when the store was created. Dropping it here means a caller can set
        ``encoding`` once and let every append reuse what the store already has.

        :param append_dim: The dimension being appended along, if any.
        :returns: Encoding mapping, or ``None``.
        """
        if append_dim is not None:
            return None

        encoding = self.config.write_params["encoding"]
        return dict(encoding) if encoding else None

    def _with_append_safe_time(
        self, obj: WritableObject, encoding: Optional[Dict[str, Any]], creating: bool
    ) -> Tuple[WritableObject, Optional[Dict[str, Any]]]:
        """
        Make datetime values on the append dimension safe to write and to append.

        xarray chooses a datetime coordinate's units from the first write alone, and
        later appends are encoded in those units. A first step of a single value lands on
        "days since <that instant>", so a second step half an hour later is stored as
        0.5 days truncated to an integer - and reads back as the wrong time, with no
        error. When the store is being created, nanosecond units are fixed up front so
        every later step is exact.

        The coordinate is also widened to nanosecond resolution on every write, because
        second-resolution values encoded in a finer unit come back as NaT.

        Only applies when ``append_dim`` is set and names a datetime coordinate; an
        explicit encoding for that coordinate is left alone.

        :param obj: Object being written.
        :param encoding: Encoding resolved so far, if any.
        :param creating: ``True`` when this write creates the store (or group) rather
            than appending to it.
        :returns: The object and encoding, changed only when this applies.
        """
        dim = self.config.write_params["append_dim"]
        if dim is None or not isinstance(obj, xr.Dataset) or dim not in obj.coords:
            return obj, encoding

        coord = obj.coords[dim]
        if coord.dtype.kind != "M":
            return obj, encoding

        obj = obj.assign_coords({dim: coord.astype("datetime64[ns]")})
        if not creating or "units" in coord.encoding or (encoding and dim in encoding):
            return obj, encoding
        return obj, {**(encoding or {}), dim: {"units": "nanoseconds since 1970-01-01", "dtype": "int64"}}

    # ---- reconciling an append with the store ----

    def _read_store_state(
        self, path: Path, append_dim: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any], Dict[Any, Tuple[Any, ...]]]:
        """
        Read the parts of an existing store an append has to reconcile with.

        All three are cheap: the coordinate is one small array and the attributes and
        variable dimensions are metadata, so a consolidated store answers this in a single
        read.

        The store is opened with ``consolidated=None`` rather than the write parameter:
        how a store was written says nothing about how the next append is configured, and
        insisting on a match would make a store written without consolidated metadata
        unreadable here.

        A store that cannot be read is an error, not "nothing to reconcile". Returning no
        state would skip the overlap check, so an unreadable store would be appended to
        without ever being compared against, duplicating whatever it holds.

        :param path: Store being appended to.
        :param append_dim: Dimension being appended along.
        :returns: Tuple of the existing coordinate values (``None`` when the dimension
            carries no coordinate), the store's root attributes, and a mapping of each
            variable name to its dimensions.
        :raises ValueError: If the store exists but cannot be read.
        """
        group = self.config.write_params["group"]
        open_kwargs: Dict[str, Any] = {"consolidated": None}
        if group is not None:
            open_kwargs["group"] = group

        try:
            with xr.open_zarr(path, **open_kwargs) as store:
                coord = np.asarray(store[append_dim].values) if append_dim in store.coords else None
                return coord, dict(store.attrs), {name: var.dims for name, var in store.variables.items()}
        except Exception as exc:  # noqa: BLE001 - re-raised below with context
            raise ValueError(
                f"Cannot read the existing store {path.name} to reconcile an append with it "
                f"({type(exc).__name__}: {exc}). Appending without that check could duplicate "
                "what the store already holds."
            ) from exc

    @staticmethod
    def _group_absent(path: Path, group: str) -> bool:
        """
        Report whether ``group`` is missing from an otherwise readable store.

        :param path: Store path.
        :param group: Group path within the store.
        :returns: ``True`` only when the store opens and the group is not in it.
        """
        import zarr

        try:
            root = zarr.open_group(path, mode="r")
        except Exception:  # noqa: BLE001 - an unopenable store is not a missing group
            return False
        return group.strip("/") not in root

    @staticmethod
    def _check_variables_match(
        obj: WritableObject,
        path: Path,
        append_dim: str,
        existing_dims: Mapping[Any, Tuple[Any, ...]],
    ) -> None:
        """
        Refuse an append whose variables along ``append_dim`` differ from the store's.

        xarray accepts such an append and returns normally, but a variable present on
        only one side ends up a different length from the rest of the dimension. The
        store is then unreadable, and so is every later append, so the mismatch has to be
        caught before anything is written. Variables that do not vary along ``append_dim``
        are not affected, and may be added freely.

        :param obj: Object about to be appended.
        :param path: Store being appended to.
        :param append_dim: Dimension being appended along.
        :param existing_dims: Mapping of each variable in the store to its dimensions.
        :raises ValueError: If a variable along ``append_dim`` is in only one of the two.
        """
        incoming = {name for name, var in obj.variables.items() if name != append_dim and append_dim in var.dims}
        stored = {name for name, dims in existing_dims.items() if name != append_dim and append_dim in dims}

        new, missing = sorted(map(str, incoming - stored)), sorted(map(str, stored - incoming))
        if not new and not missing:
            return

        parts = []
        if new:
            parts.append(f"not in the store: {', '.join(new)}")
        if missing:
            parts.append(f"in the store but not in the appended object: {', '.join(missing)}")
        raise ValueError(
            f"Cannot append along {append_dim!r} to {path.name}: variables along it differ ("
            f"{'; '.join(parts)}). Appending would leave them a different length from the "
            "rest of the dimension and make the store unreadable."
        )

    def _resolve_overlap(
        self,
        obj: WritableObject,
        path: Path,
        append_dim: str,
        existing: Optional[np.ndarray],
    ) -> Optional[WritableObject]:
        """
        Apply the ``on_overlap`` policy to coordinates the store already holds.

        A dimension with no coordinate carries no identity, so there is nothing to
        compare and the write proceeds untouched.

        :param obj: Object about to be appended.
        :param path: Store being appended to.
        :param append_dim: Dimension being appended along.
        :param existing: Coordinate values already in the store, if any.
        :returns: The object to write, trimmed of anything already present, or ``None``
            when nothing is left to write.
        :raises ValueError: If the policy is unrecognised, or is ``"error"`` and the
            incoming coordinates overlap what the store holds.
        """
        policy = self.config.write_params["on_overlap"]
        if policy not in ("error", "skip", "allow"):
            raise ValueError(f"on_overlap must be 'error', 'skip' or 'allow', got {policy!r}")

        if policy == "allow" or existing is None or append_dim not in obj.coords:
            return obj

        incoming = np.asarray(obj[append_dim].values)
        overlap = np.isin(incoming, existing)

        if not overlap.any():
            return obj

        if policy == "error":
            duplicated = ", ".join(str(v) for v in incoming[overlap][:3])
            more = "..." if overlap.sum() > 3 else ""
            raise ValueError(
                f"{int(overlap.sum())} of {incoming.size} {append_dim!r} value(s) are already in "
                f"{path.name} ({duplicated}{more}). Appending would duplicate them, which is what "
                "re-running a partly successful write does. Pass on_overlap='skip' to write only "
                "what is new, or 'allow' to append regardless."
            )

        if overlap.all():
            return None
        return obj.isel({append_dim: ~overlap})

    def _merge_store_attrs(self, obj: WritableObject, existing: Mapping[Any, Any]) -> WritableObject:
        """
        Merge the store's existing root attributes into the object being appended.

        Without this, zarr replaces the group's attributes with whatever the incoming
        object carries, so a stack ends up describing only its most recent append.

        :param obj: Object about to be appended, with attributes already encoded.
        :param existing: Root attributes currently in the store.
        :returns: The object with reconciled attributes. The input is not modified.
        """
        if not existing:
            return obj

        out = obj.copy()
        out.attrs = _reconcile_attrs(existing, out.attrs)
        return out


# -------------------------
# Attribute reconciliation
# -------------------------


def _reconcile_attrs(existing: Mapping[Any, Any], incoming: Mapping[Any, Any]) -> Dict[Any, Any]:
    """
    Combine a store's root attributes with those of an object being appended to it.

    Root attributes describe the whole store, so a per-object value cannot simply
    replace them. The store wins any disagreement - it describes everything written so
    far, where the incoming object describes only itself - and the conflict is warned
    about rather than resolved silently. ``history`` and ``eoio:processing_steps``
    accumulate instead, since those are records of what has happened rather than
    statements about the data.

    :param existing: Attributes already in the store, as stored (encoded).
    :param incoming: Attributes of the object being appended, already encoded.
    :returns: The merged attributes.
    """
    merged = dict(existing)
    conflicts: List[Any] = []

    for key, value in incoming.items():
        if key == HISTORY_ATTR:
            merged[key] = _merge_history(existing.get(key), value)
        elif key == STEPS_ATTR:
            merged[key] = _merge_steps(existing.get(key), value)
        elif key == MARKER_ATTR:
            merged[key] = _merge_marker(existing.get(key), value)
        elif key not in merged:
            merged[key] = value
        elif not _same(merged[key], value):
            conflicts.append(key)

    if conflicts:
        warnings.warn(
            f"Keeping the store's value for {', '.join(sorted(str(key) for key in conflicts))}: the appended object "
            "disagrees, and root attributes describe the whole store rather than any one object. "
            "Store per-object metadata as a variable along the append dimension instead.",
            stacklevel=4,
        )

    return merged


def _merge_history(existing: Any, incoming: Any) -> str:
    """
    Concatenate two ``history`` blocks, dropping lines already recorded.

    Deduplication matters because every object appended to a stack tends to carry the
    same preamble from the reader that produced it; without it a 200-scene stack repeats
    those lines 200 times.

    :param existing: History already in the store.
    :param incoming: History of the object being appended.
    :returns: The merged history, oldest block first.
    """
    lines: List[str] = []
    for block in (existing, incoming):
        if not block:
            continue
        for line in str(block).splitlines():
            if line and line not in lines:
                lines.append(line)
    return "\n".join(lines)


def _merge_steps(existing: Any, incoming: Any) -> Any:
    """
    Concatenate two ``eoio:processing_steps`` lists, dropping steps already recorded.

    :param existing: Steps already in the store.
    :param incoming: Steps of the object being appended.
    :returns: The merged steps, JSON-encoded when either input was.
    """
    merged: List[Any] = []
    for block in (existing, incoming):
        for step in _as_steps(block):
            if step not in merged:
                merged.append(step)

    if isinstance(existing, str) or isinstance(incoming, str):
        return json.dumps(merged)
    return merged


def _as_steps(value: Any) -> List[Any]:
    """
    Read a processing-steps attribute in either its encoded or decoded form.

    :param value: Attribute value, a JSON string or a list.
    :returns: The steps, empty when there are none or the value is unreadable.
    """
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _merge_marker(existing: Any, incoming: Any) -> str:
    """
    Merge two encoding markers so the merged attributes stay fully described.

    The store's entries win where both describe the same attribute, matching the
    precedence :func:`_reconcile_attrs` applies to the values themselves - otherwise the
    marker would name a codec for a value that was not kept.

    :param existing: Marker already in the store.
    :param incoming: Marker of the object being appended.
    :returns: The merged marker, JSON-encoded.
    """
    merged = {**_as_marker(incoming), **_as_marker(existing)}
    return json.dumps(merged, sort_keys=True)


def _as_marker(value: Any) -> Dict[str, str]:
    """
    Read an encoding marker in either its encoded or decoded form.

    :param value: Attribute value, a JSON string or a dict.
    :returns: Mapping of attribute name to codec name; empty when unreadable.
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _same(a: Any, b: Any) -> bool:
    """
    Compare two attribute values, tolerating the array forms zarr returns.

    :param a: First value.
    :param b: Second value.
    :returns: ``True`` when the values are equivalent.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    return bool(a == b)


if __name__ == "__main__":
    pass
