"""eoio.writers.netcdf.writer - netCDF writer

Wraps ``to_netcdf`` with the parts eoio needs on top of it: attribute encoding (handled
by :class:`~eoio.writers.base.BaseWriter`), and compression applied by default rather
than left to the caller to remember.

A ``DataTree`` is written as netCDF4 groups, one per node.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Optional

import xarray as xr

from eoio.writers.base import BaseWriter, WritableObject, WriteCapability
from eoio.writers.factory import register_writer

__all__ = ["NetCDFWriter"]


#: Formats that support zlib compression. NETCDF3 variants do not.
_COMPRESSIBLE_FORMATS = {None, "NETCDF4", "NETCDF4_CLASSIC"}

#: Engines that accept the zlib/complevel encoding keys. h5netcdf translates them to
#: HDF5 gzip compression; scipy (NETCDF3 only) has no compression.
_COMPRESSIBLE_ENGINES = {None, "netcdf4", "h5netcdf"}


@lru_cache(maxsize=1)
def _default_engine() -> Optional[str]:
    """
    Pick the engine to use when none is requested.

    xarray chooses netCDF4 whenever the package is *installed*, even if it cannot be
    imported - a wheel built against a mismatched numpy ABI fails at import with
    ``numpy.dtype size changed``. Probing the import here lets such an environment fall
    back to h5netcdf instead of failing every write.

    :returns: ``"netcdf4"``, ``"h5netcdf"`` when netCDF4 is unusable, or ``None`` to
        leave the choice to xarray when neither can be imported.
    """
    try:
        import netCDF4  # noqa: F401
    except (ImportError, ValueError):
        pass
    else:
        return "netcdf4"

    try:
        import h5netcdf  # noqa: F401
    except (ImportError, ValueError):
        return None
    return "h5netcdf"


@register_writer("netcdf")
class NetCDFWriter(BaseWriter):
    """Write a Dataset or DataTree to netCDF."""

    default_write_params: Dict[str, Any] = {
        **BaseWriter.default_write_params,
        "engine": None,
        "format": "NETCDF4",
        "mode": "w",
        "group": None,
        "encoding": None,
        "compress": True,
        "complevel": 4,
        "compress_min_bytes": 16384,
        "unlimited_dims": None,
    }

    all_write_params: Dict[str, str] = {
        **BaseWriter.all_write_params,
        "engine": "netcdf4/scipy/h5netcdf, or None to use netcdf4 and fall back to h5netcdf if it cannot be imported",
        "format": "NETCDF4/NETCDF4_CLASSIC/NETCDF3_64BIT/NETCDF3_CLASSIC",
        "mode": "w/a",
        "group": "group path for a Dataset, or None for the root group (not supported for a DataTree)",
        "encoding": "per-variable encoding dict, or None to use the compression defaults",
        "compress": "True/False",
        "complevel": "1-9",
        "compress_min_bytes": "smallest variable size in bytes to compress; smaller ones are stored uncompressed",
        "unlimited_dims": "dimension name(s) to write as unlimited; for a DataTree, applied to every group containing them",
    }

    @staticmethod
    def get_extension() -> str:
        return ".nc"

    def can_write(self, obj: Any) -> WriteCapability:
        """
        Report whether this writer can express ``obj`` with its current parameters.

        :param obj: Candidate object.
        :returns: Capability, with a reason when the answer is no.
        """
        capability = super().can_write(obj)
        if not capability.ok:
            return capability

        # xarray accepts a ``group`` argument on DataTree.to_netcdf but raises
        # NotImplementedError for any value other than None.
        if isinstance(obj, xr.DataTree) and self.config.write_params["group"] is not None:
            return WriteCapability(
                False,
                "a DataTree cannot be written under a root group (the 'group' write param); "
                "its own structure is the groups",
            )
        return capability

    def _engine(self) -> Optional[str]:
        """
        Resolve the engine for this write.

        :returns: The requested engine, or the probed default when none was requested
            and the format is NETCDF4. Other formats are left to xarray, since h5netcdf
            cannot write NETCDF3 or the CLASSIC model.
        """
        params = self.config.write_params
        if params["engine"] is not None:
            return params["engine"]
        if params["format"] == "NETCDF4":
            return _default_engine()
        return None

    def write_dataset(self, obj: WritableObject, path: Path) -> None:
        """
        Write the object to netCDF.

        :param obj: Dataset or DataTree, already encoded by the base class.
        :param path: Destination path.
        """
        params = self.config.write_params

        kwargs: Dict[str, Any] = {
            "mode": params["mode"],
            "format": params["format"],
            "engine": self._engine(),
        }

        unlimited_dims = self._resolve_unlimited_dims(obj)
        if unlimited_dims:
            kwargs["unlimited_dims"] = unlimited_dims

        # A DataTree's structure is its groups; can_write() has already refused a group
        # for one, so this only ever applies to a Dataset.
        if isinstance(obj, xr.Dataset) and params["group"] is not None:
            kwargs["group"] = params["group"]

        encoding = self._resolve_encoding(obj)
        if encoding:
            kwargs["encoding"] = encoding

        obj.to_netcdf(path, **kwargs)

    def _resolve_unlimited_dims(self, obj: WritableObject) -> Any:
        """
        Work out the ``unlimited_dims`` argument for this write.

        ``Dataset.to_netcdf`` takes dimension names, but ``DataTree.to_netcdf`` takes a
        mapping of group path to dimension names. A list or single name is therefore
        expanded to every group that actually contains each dimension; a mapping is
        passed through untouched.

        :param obj: Object being written.
        :returns: The value to pass to ``to_netcdf``, or ``None`` when not requested.
        """
        requested = self.config.write_params["unlimited_dims"]
        if requested is None or not isinstance(obj, xr.DataTree) or isinstance(requested, Mapping):
            return requested

        names: List[Hashable] = [requested] if isinstance(requested, str) else list(requested)
        per_group: Dict[str, List[Hashable]] = {}
        for node in obj.subtree:
            present = [name for name in names if name in node.to_dataset(inherit=False).dims]
            if present:
                per_group[node.path] = present
        return per_group

    # ---- encoding ----

    def _resolve_encoding(self, obj: WritableObject) -> Optional[Dict[Hashable, Any]]:
        """
        Work out the ``encoding`` argument for this write.

        An explicit ``encoding`` write param is passed through untouched. Otherwise
        compression defaults are generated, unless they are switched off or the chosen
        engine/format cannot express them.

        :param obj: Object being written.
        :returns: Encoding mapping, or ``None`` to let xarray decide.
        """
        params = self.config.write_params

        if params["encoding"] is not None:
            return dict(params["encoding"])

        if not params["compress"]:
            return None

        if self._engine() not in _COMPRESSIBLE_ENGINES or params["format"] not in _COMPRESSIBLE_FORMATS:
            return None

        settings = {"zlib": True, "complevel": params["complevel"]}
        min_bytes = params["compress_min_bytes"]

        if isinstance(obj, xr.DataTree):
            # Keyed by absolute group path, and only over what each node actually owns:
            # naming an inherited coordinate here is rejected, because it belongs to the
            # group that defines it.
            tree_encoding: Dict[Hashable, Any] = {}
            for node in obj.subtree:
                node_encoding = self._compressible_vars(node.to_dataset(inherit=False), settings, min_bytes)
                if node_encoding:
                    tree_encoding[node.path] = node_encoding
            return tree_encoding or None

        return self._compressible_vars(obj, settings, min_bytes) or None

    @staticmethod
    def _compressible_vars(ds: xr.Dataset, settings: Dict[str, Any], min_bytes: int) -> Dict[Hashable, Any]:
        """
        Build per-variable compression settings for everything worth compressing.

        Scalars are skipped: compressing a 0-d variable achieves nothing and only adds
        chunking metadata to the file. So are variables smaller than ``min_bytes``: a
        compressed variable is chunked, which costs about 2 kB in the file however small it
        is, and that only pays back once the data is large enough - measured, the break-even
        for float64 is around 1000-2000 elements. Below it the file gets bigger, which is
        what a product made of many short in-situ series, such as RadCalNet, would otherwise
        do. ``nbytes`` is computed from the shape, so a lazy variable is not loaded.

        :param ds: Dataset whose variables to cover.
        :param settings: Encoding settings to apply to each.
        :param min_bytes: Smallest variable, in bytes, to compress.
        :returns: Mapping of variable name to encoding.
        """
        return {
            name: dict(settings)
            for name, variable in ds.variables.items()
            if variable.ndim >= 1 and variable.nbytes >= min_bytes
        }


if __name__ == "__main__":
    pass
