"""eoio.writers.base - base classes for eoio dataset writers

Design goals
------------
- Mirror `eoio.readers.base` so both halves of the library read the same way.
- Keep the base small: it owns configuration and orchestration, concrete writers own
  the format.
- Keep import-time dependencies light (format backends are imported lazily).

Writer contract
---------------
Concrete writers must implement:
- `write_dataset(obj, path) -> None`
- `get_extension() -> str`

and should set:
- `name` - the identifier the writer is registered under

Optionally override:
- `can_write(obj) -> WriteCapability` - to refuse objects the format cannot express
- `default_write_params` / `all_write_params` - to declare format-specific options

Typical usage
-------------
    writer = SomeWriter(write_params={...})
    writer.write(ds, "output.nc")

`write()` is the public entrypoint. Unlike a reader, which is constructed around a
single product path, a writer instance holds only configuration and can be reused
across many outputs - which is what `write_multi` relies on.

Note on history
---------------
This is not a revival of the writer package deleted in `8d1a300`. That one patched
existing products in place (copy a `.SAFE` tree, apply per-band corrections); this one
serialises datasets. The contracts have nothing in common.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Union

import xarray as xr

from eoio.serialisation import encode_attrs

__version__ = version("eoio")

__all__ = ["BaseWriter", "WritableObject", "WriteCapability", "WriterConfig"]


#: Objects a writer accepts.
WritableObject = Union[xr.Dataset, xr.DataTree]


# -------------------------
# Configuration
# -------------------------


@dataclass(frozen=True)
class WriterConfig:
    """Normalised, validated configuration for a writer instance."""

    write_params: Dict[str, Any]

    #: Names the caller actually supplied. Needed wherever a default and an explicitly
    #: requested value of the same parameter must behave differently: the Zarr writer
    #: refuses ``mode`` alongside ``append_dim``, which it can only do without also
    #: refusing its own default.
    provided_write_params: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class WriteCapability:
    """
    Whether a writer can express a given object in its format.

    :param ok: Whether the object can be written.
    :param reason: Why not, when ``ok`` is False. Should say what is missing, so the
        user can act on it.
    """

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# -------------------------
# Base writer
# -------------------------


class BaseWriter(ABC):
    """Base class for eoio writers.

    This class is intentionally small. It owns:
    - config merge/validation
    - a single public entrypoint: `write()`
    - the orchestration common to every format: capability check, conformance check,
      provenance, attribute encoding

    Concrete writers implement `write_dataset()` and `get_extension()`.
    """

    #: Name this writer is registered under. Concrete writers must set this.
    name: str = ""

    # Default values for user inputs, for all available parameters.
    # Concrete writers should extend these with their own format options.

    default_write_params: Dict[str, Any] = {
        "encode_attrs": True,
        "on_nonconformance": "warn",
        "update_history": True,
        "create_parents": True,
    }

    all_write_params: Dict[str, str] = {
        "encode_attrs": "True/False",
        "on_nonconformance": "warn/error/ignore",
        "update_history": "True/False",
        "create_parents": "True/False",
    }

    def __init__(self, write_params: Dict[str, Any] | None = None) -> None:
        self.config = WriterConfig(
            write_params=self._merge_and_validate(write_params, self.default_write_params, kind="writing"),
            provided_write_params=frozenset(write_params or {}),
        )

        self._all_options: Dict[str, Any] | None = None

    # ---- public API ----

    def write(self, obj: WritableObject, path: Path | str) -> str:
        """
        Public entrypoint: write an object to ``path``.

        Orchestration is the same for every format:

        1. ``can_write()`` - refuse early, with a reason the user can act on
        2. ``check_conformance()`` - CF-1.8 / ACDD / controlled vocabulary
        3. provenance - append to ``history`` and ``eoio:processing_steps``
        4. ``encode_attrs()`` - make attributes serialisable
        5. ``write_dataset()`` - the format-specific part

        Provenance is added *before* encoding and writing, so that it ends up in the
        written file rather than only on a discarded in-memory copy.

        The input object is never modified.

        :param obj: Dataset or DataTree to write.
        :param path: Destination path.
        :returns: The path written, as a string.
        :raises TypeError: If this writer cannot express ``obj``.
        """
        capability = self.can_write(obj)
        if not capability.ok:
            raise TypeError(f"{type(self).__name__} cannot write this object: {capability.reason}")

        self._apply_conformance_policy(obj)

        target = self._resolve_path(path)

        out = obj.copy()
        if self.config.write_params["update_history"]:
            out = self._add_provenance(out, target)

        if self.config.write_params["encode_attrs"]:
            out = encode_attrs(out)

        self.write_dataset(out, target)

        return str(target)

    @abstractmethod
    def write_dataset(self, obj: WritableObject, path: Path) -> None:
        """
        Write the object in this writer's format.

        Called with attributes already encoded and provenance already applied, so
        implementations only have to deal with the format itself.

        :param obj: Dataset or DataTree to write.
        :param path: Destination path, already resolved.
        """

    def can_write(self, obj: Any) -> WriteCapability:
        """
        Report whether this writer can express ``obj``.

        The default accepts any ``Dataset`` or ``DataTree``, which is correct for
        general-purpose formats: anything expressible in xarray is expressible in
        netCDF or Zarr. Format-specific writers should override this and explain what
        is missing - a Sentinel-2 SAFE writer needs an S2-shaped dataset, and saying so
        is more useful than emitting a broken product.

        :param obj: Candidate object.
        :returns: Capability, with a reason when the answer is no.
        """
        if not isinstance(obj, (xr.Dataset, xr.DataTree)):
            return WriteCapability(
                False,
                f"expected an xarray Dataset or DataTree, got {type(obj).__name__}",
            )
        return WriteCapability(True)

    def check_conformance(self, obj: WritableObject) -> List[str]:
        """
        Check the object against CF-1.8 / ACDD / the controlled vocabulary.

        Placeholder until the conformance checks land (see #205); returning no issues
        keeps the orchestration honest in the meantime rather than pretending to check.

        :param obj: Object about to be written.
        :returns: Human-readable descriptions of non-conformances; empty when it
            conforms.
        """
        return []

    @staticmethod
    @abstractmethod
    def get_extension() -> str:
        """
        Return the filename extension this writer produces, including the dot.

        Used to name outputs when the caller gives a directory rather than a full path
        (see `write_multi`).
        """

    @property
    def all_options(self) -> Dict[str, Any]:
        """
        Return the available options for this writer.

        :returns: Dictionary of available write parameters.
        """
        if self._all_options is None:
            self._all_options = {"write_params": self.all_write_params.copy()}
        return self._all_options

    # ---- internals ----

    def _apply_conformance_policy(self, obj: WritableObject) -> None:
        """
        Run the conformance check and act on it according to ``on_nonconformance``.

        :param obj: Object about to be written.
        :raises ValueError: If ``on_nonconformance`` is ``"error"`` and issues are found,
            or if the policy value is not recognised.
        """
        policy = self.config.write_params["on_nonconformance"]
        if policy not in ("warn", "error", "ignore"):
            raise ValueError(f"on_nonconformance must be 'warn', 'error' or 'ignore', got {policy!r}")

        if policy == "ignore":
            return

        issues = self.check_conformance(obj)
        if not issues:
            return

        summary = "; ".join(issues)
        if policy == "error":
            raise ValueError(f"Dataset does not conform: {summary}")
        warnings.warn(f"Dataset does not conform: {summary}", stacklevel=3)

    def _resolve_path(self, path: Path | str) -> Path:
        """
        Normalise the destination path, creating parent directories if configured.

        :param path: Destination path.
        :returns: The resolved path.
        """
        target = Path(path)
        if self.config.write_params["create_parents"] and target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _add_provenance(self, obj: WritableObject, path: Path) -> WritableObject:
        """
        Record this write in ``history`` and ``eoio:processing_steps``.

        The step entry uses the same shape `read_multi` writes, so a dataset's
        processing history reads consistently regardless of which end produced it.

        :param obj: Object about to be written.
        :param path: Destination path.
        :returns: The object with provenance attributes updated.
        """
        attrs = dict(obj.attrs)

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        filename = os.path.basename(str(path))
        entry = f"{timestamp}: written to {filename} using eoio {self.name or type(self).__name__} writer version {__version__}"
        attrs["history"] = f"{attrs['history']}\n{entry}" if attrs.get("history") else entry

        steps = self._as_step_list(attrs.get("eoio:processing_steps", []))
        attrs["eoio:processing_steps"] = [
            *steps,
            {"processor": "write", "writer": self.name or type(self).__name__, "path": filename},
        ]

        obj.attrs = attrs
        return obj

    @staticmethod
    def _as_step_list(value: Any) -> List[Any]:
        """
        Normalise an existing ``eoio:processing_steps`` value to a list of steps.

        An object read back from a file without ``decode_attrs`` carries the attribute
        as the JSON string it was encoded to; appending to it as-is would nest that
        string inside the list instead of extending it.

        :param value: The attribute's current value.
        :returns: The steps as a list. A string that is not a JSON list or object is kept
            whole as a single step.
        """
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return [value]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        return [value]

    # ---- config helpers ----

    @staticmethod
    def _merge_and_validate(
        provided: Dict[str, Any] | None,
        defaults: Dict[str, Any],
        *,
        kind: str,
    ) -> Dict[str, Any]:
        """
        Merge provided dict onto defaults and validate keys.

        Intentionally strict: unknown keys raise, matching
        :meth:`eoio.readers.base.BaseReader._merge_and_validate`.

        :param provided: Provided keys.
        :param defaults: Default keys.
        :param kind: Provided dictionary name (only used for clarifying errors).
        :return: Merged dictionary.
        :raises ValueError: If any provided key is not present in the defaults.
        """
        provided = provided or {}
        unknown = [k for k in provided.keys() if k not in defaults]
        if unknown:
            raise ValueError(f"Unknown {kind} parameter(s): {unknown}. Allowed: {sorted(defaults.keys())}")

        merged = dict(defaults)
        merged.update(provided)
        return merged


if __name__ == "__main__":
    pass
