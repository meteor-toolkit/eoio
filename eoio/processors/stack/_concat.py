"""
eoio.processors.stack._concat
------------------------------

Dimension-agnostic concatenation and attribute-reconciliation helpers shared
by the ``stack`` processor (concatenating per-band DataArrays within one
Dataset) and ``eoio.read_multi`` (concatenating whole Datasets across files).

These helpers have no dependency on the ``stack`` processor's per-band
grouping logic, so they live here and are imported back into
``eoio.processors.stack.processor`` for backwards compatibility.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import xarray as xr


def _safe_equal(a: Any, b: Any) -> bool:
    """
    Compare two attribute values for equality without raising.

    Plain ``==`` breaks for array-valued attrs (``a == b`` returns an array,
    not a bool, so ``all(...)``/``if`` raises) — real EO products can carry
    these, e.g. empty placeholder arrays for unused uncertainty-effect
    parameters. ``np.array_equal`` handles scalars, strings, dicts, lists,
    and arrays uniformly and always returns a plain bool.
    """
    try:
        return bool(np.array_equal(a, b))
    except Exception:
        return False


def _get_nested_attr(obj: Union[xr.DataArray, xr.Dataset], attr_path: str) -> Any:
    """
    Retrieve a variable or dataset attribute using dot-notation for nested dicts.

    ``"central_wavelength"`` returns ``obj.attrs["central_wavelength"]``.
    ``"product_metadata.central_wavelength"`` returns
    ``obj.attrs["product_metadata"]["central_wavelength"]``.

    A dict-valued step that has already been JSON-stringified (e.g. a reader
    that serialised ``product_metadata`` to a netCDF-safe string before this
    processor ran) is transparently parsed back into a dict before descending
    further, so the lookup still succeeds either way.

    Returns ``None`` if any key in the path is missing.
    """
    val: Any = obj.attrs
    for key in attr_path.split("."):
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except ValueError:
                return None
        if not isinstance(val, dict):
            return None
        val = val.get(key)
        if val is None:
            return None
    return val


def _obs_concat(objs: Union[List[xr.DataArray], List[xr.Dataset]], dim: str) -> Union[xr.DataArray, xr.Dataset]:
    """
    Concatenate DataArrays or Datasets along ``dim`` using the obsarray interface.

    Currently delegates to ``xr.concat``. When ``obsarray.concat`` is
    available this function will be updated to use it, enabling
    uncertainty-aware concatenation that propagates associated uncertainty
    variables and metadata automatically.

    ``join="outer"`` is passed explicitly (rather than relying on xarray's
    current default) so that mismatched non-concat-dim coordinates keep
    being reconciled by padding with NaN — this is what ``read_multi``'s
    ``on_mismatch="union"`` depends on — regardless of xarray's own default
    changing in future releases.
    """
    return xr.concat(objs, dim=dim, join="outer")


def _is_scalar_attr(val: Any) -> bool:
    """Return True if val is a scalar numeric or string — safe to store as a list attr or coordinate."""
    return isinstance(val, (bool, int, float, str))


def _cube_attrs_and_coords(
    ds: xr.Dataset,
    var_names: List[str],
    measurand: str,
    measurand_attr: str,
    coord_attrs: List[str] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Build attribute dict and coordinate promotions for a stacked cube.

    Attributes are reconciled across constituent variables as follows:

    - **Shared** (identical across all variables): kept as scalar attrs.
    - **Per-band scalar** (differ but are all numeric or string scalars):
      promoted to a coordinate variable if the attribute name is in
      ``coord_attrs``; otherwise kept as a list attr in stacking order.
    - **Per-band complex** (dicts, lists, or mixed types): kept as a list attr
      in stacking order. Serialisation to NetCDF/Zarr is left to the caller.

    ``long_name`` and ``ancillary_variables`` are always replaced with
    cube-level equivalents.

    :returns:
        Tuple of ``(attrs, to_promote)`` where ``attrs`` is the cube attribute
        dict and ``to_promote`` maps attribute names to per-element value lists
        for coordinate assignment.
    """
    coord_attrs_set: set = set(coord_attrs or [])
    first_attrs = dict(ds[var_names[0]].attrs)

    attrs: Dict[str, Any] = {}
    to_promote: Dict[str, List[Any]] = {}

    for key, first_val in first_attrs.items():
        if key in ("ancillary_variables",):
            continue
        all_vals = [ds[name].attrs.get(key) for name in var_names]
        if all(_safe_equal(v, first_val) for v in all_vals[1:]):
            attrs[key] = first_val
        elif key == "long_name":
            pass  # falls back to the measurand-cube default below
        elif None not in all_vals:
            if all(_is_scalar_attr(v) for v in all_vals) and key in coord_attrs_set:
                to_promote[key] = all_vals
            else:
                attrs[key] = all_vals

    attrs[measurand_attr] = measurand
    if "long_name" not in attrs:
        attrs["long_name"] = measurand

    return attrs, to_promote


def _reconcile_attrs(
    attrs_list: List[Dict[str, Any]],
    coord_attrs: Optional[List[str]] = None,
    chunk_sizes: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Reconcile a list of attrs dicts, one per source file, in file order.

    Unlike :func:`_cube_attrs_and_coords` (which only looks at the first
    variable's attribute keys), this considers the union of keys across all
    files, so an attribute present on a later file but not the first is
    still kept rather than silently dropped.

    - **Shared** (identical across all files): kept as a scalar attr.
    - **Differing, named in ``coord_attrs``, scalar on every file**: promoted
      to a per-``concat_dim``-element coordinate. Each file's value is
      repeated ``chunk_sizes[i]`` times so the coordinate's length matches
      the concatenated dimension (one value per file if ``chunk_sizes`` is
      not given, e.g. one slice per file in the raster/``expand_dims`` case).
    - **Anything else that differs** (including attrs missing from some
      files, or non-scalar values): kept as a list attr, one entry per file,
      in file order. Serialisation to NetCDF/Zarr is left to the caller.

    :returns: ``(attrs, to_promote)`` — ``attrs`` is the reconciled attrs
        dict; ``to_promote`` maps attribute names to per-``concat_dim``-element
        value lists for coordinate assignment.
    """
    if not attrs_list:
        return {}, {}
    if chunk_sizes is None:
        chunk_sizes = [1] * len(attrs_list)

    coord_attrs_set = set(coord_attrs or [])
    keys: List[str] = []
    seen: set = set()
    for a in attrs_list:
        for key in a:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    attrs: Dict[str, Any] = {}
    to_promote: Dict[str, List[Any]] = {}

    for key in keys:
        all_vals = [a.get(key) for a in attrs_list]
        first_val = all_vals[0]
        if all(_safe_equal(v, first_val) for v in all_vals[1:]):
            attrs[key] = first_val
        elif key in coord_attrs_set and all(_is_scalar_attr(v) for v in all_vals):
            to_promote[key] = [v for v, n in zip(all_vals, chunk_sizes) for _ in range(n)]
        else:
            attrs[key] = all_vals

    return attrs, to_promote


def _apply_attrs_reconciliation(
    result: xr.Dataset,
    per_file_datasets: List[xr.Dataset],
    concat_dim: str,
    coord_attrs: Optional[List[str]],
) -> xr.Dataset:
    """
    Overwrite ``result``'s dataset-level and per-variable attrs (which
    ``xr.concat`` otherwise silently sets to just the first file's, via its
    default ``combine_attrs="override"``) with attrs reconciled across every
    contributing file.

    Dataset-level attrs support ``coord_attrs`` promotion (see
    :func:`_reconcile_attrs`); per-variable attrs do not — a variable's attrs
    describe the variable as a whole (e.g. ``scale_factor``), not a
    per-``concat_dim``-element quantity, so they only ever collapse to a
    shared scalar or a per-file list attr.

    ``eoio:processing_steps`` is itself a per-file *list* of step records, so
    it is merged by concatenation (in file order) rather than reconciled like
    a scalar attr — otherwise it would end up nested as a list-of-lists.
    """
    chunk_sizes = [ds.sizes.get(concat_dim, 1) for ds in per_file_datasets]

    steps_key = "eoio:processing_steps"
    all_ds_attrs = []
    merged_steps: List[Any] = []
    has_steps = False
    for ds in per_file_datasets:
        file_attrs = dict(ds.attrs)
        steps = file_attrs.pop(steps_key, None)
        if steps is not None:
            has_steps = True
            merged_steps.extend(steps if isinstance(steps, list) else [steps])
        all_ds_attrs.append(file_attrs)

    ds_attrs, ds_to_promote = _reconcile_attrs(all_ds_attrs, coord_attrs, chunk_sizes)
    if has_steps:
        ds_attrs[steps_key] = merged_steps
    result.attrs = ds_attrs
    for name, values in ds_to_promote.items():
        result = result.assign_coords({name: (concat_dim, values)})

    for var_name in result.data_vars:
        var_attrs_list = [ds[var_name].attrs for ds in per_file_datasets if var_name in ds]
        if not var_attrs_list:
            continue
        var_attrs, _ = _reconcile_attrs(var_attrs_list)
        result[var_name].attrs = var_attrs

    return result


def _dataset_schema(ds: xr.Dataset, concat_dim: str) -> Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """
    Return a per-variable ``(dims, shape)`` fingerprint, ignoring ``concat_dim``.

    Two datasets with equal schemas can be concatenated along ``concat_dim``
    without ambiguity: same variable names, same dims (other than the
    concat dim), same shape along those dims.
    """
    schema: Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]] = {}
    for name, da in ds.data_vars.items():
        dims = tuple(str(d) for d in da.dims if d != concat_dim)
        shape = tuple(da.sizes[d] for d in dims)
        schema[str(name)] = (dims, shape)
    return schema


def _validate_and_filter_datasets(
    datasets: List[xr.Dataset],
    paths: List[str],
    concat_dim: str,
    on_mismatch: str,
) -> Tuple[List[xr.Dataset], List[str]]:
    """
    Check that ``datasets`` share a consistent variable set/shape (other than
    ``concat_dim``), applying ``on_mismatch`` to resolve any inconsistency.

    - ``"error"``: raise ``ValueError`` naming the offending paths.
    - ``"skip"``: drop mismatched datasets/paths, with a warning.
    - ``"union"``: no validation — mismatches are resolved by an outer-join
      concat (missing variables/coordinates filled with NaN).

    Datasets are compared against ``datasets[0]``'s schema.
    """
    if on_mismatch == "union":
        return datasets, paths

    ref_schema = _dataset_schema(datasets[0], concat_dim)
    kept_datasets, kept_paths = [datasets[0]], [paths[0]]
    mismatched_paths: List[str] = []

    for ds, path in zip(datasets[1:], paths[1:]):
        if _dataset_schema(ds, concat_dim) == ref_schema:
            kept_datasets.append(ds)
            kept_paths.append(path)
        else:
            mismatched_paths.append(path)

    if not mismatched_paths:
        return kept_datasets, kept_paths

    if on_mismatch == "error":
        raise ValueError(
            f"read_multi: {len(mismatched_paths)} of {len(datasets)} files have a variable set/shape "
            f"that differs from {paths[0]!r}: {mismatched_paths}. Pass on_mismatch='skip' to drop "
            "mismatched files, or on_mismatch='union' to outer-join them (missing values filled with NaN)."
        )

    # on_mismatch == "skip"
    warnings.warn(
        f"read_multi: skipping {len(mismatched_paths)} file(s) whose variable set/shape differs from "
        f"{paths[0]!r}: {mismatched_paths}."
    )
    if len(kept_datasets) < 2:
        raise ValueError(
            "read_multi: fewer than 2 datasets remain after dropping mismatched files "
            f"({mismatched_paths}); nothing to concatenate."
        )
    return kept_datasets, kept_paths


ConcatCoord = Union[str, List[Any], Callable[[str, xr.Dataset], Any]]


def _resolve_concat_coord_values(
    datasets: List[xr.Dataset],
    paths: List[str],
    concat_dim: str,
    concat_coord: Optional[ConcatCoord],
) -> List[Any]:
    """
    Compute a per-dataset coordinate value for ``concat_dim``, one of:

    - ``None`` – no coordinate info available; falls back to file order as
      an integer index, with a warning.
    - ``str`` – dot-path into ``ds.attrs`` (reuses :func:`_get_nested_attr`,
      the same mechanism as the ``stack`` processor's ``dim_coord_attr``).
    - ``list``/``tuple`` – explicit values, one per path, same order as ``paths``.
    - ``callable(path, ds) -> value`` – escape hatch for e.g. parsing a
      timestamp out of the filename.
    """
    if concat_coord is None:
        warnings.warn(
            f"read_multi: {concat_dim!r} is not an existing dimension on the input datasets and no "
            "concat_coord was given; falling back to file order as an integer index coordinate."
        )
        return list(range(len(datasets)))

    if isinstance(concat_coord, str):
        values = []
        for ds, path in zip(datasets, paths):
            val = _get_nested_attr(ds, concat_coord)
            if val is None:
                raise ValueError(f"read_multi: concat_coord={concat_coord!r} not found in ds.attrs for {path!r}.")
            values.append(val)
        return values

    if callable(concat_coord):
        return [concat_coord(path, ds) for path, ds in zip(paths, datasets)]

    if isinstance(concat_coord, (list, tuple)):
        if len(concat_coord) != len(datasets):
            raise ValueError(
                f"read_multi: concat_coord has {len(concat_coord)} values but there are "
                f"{len(datasets)} files to concatenate."
            )
        return list(concat_coord)

    raise TypeError(
        f"read_multi: concat_coord must be None, a str, a list, or a callable; got {type(concat_coord).__name__}"
    )


def _dedupe_dims(dims: Tuple[str, ...]) -> Tuple[str, ...]:
    """Suffix repeated dim names to be unique, e.g. ``("m", "m")`` -> ``("m", "m__dup1")``."""
    seen: Dict[str, int] = {}
    new_dims = []
    for d in dims:
        if d in seen:
            seen[d] += 1
            new_dims.append(f"{d}__dup{seen[d]}")
        else:
            seen[d] = 0
            new_dims.append(d)
    return tuple(new_dims)


def _split_repeated_dims(ds: xr.Dataset) -> Tuple[xr.Dataset, List[str]]:
    """
    Return a copy of ``ds`` where any data variable whose dims tuple contains
    the same dimension name twice (e.g. a ``(measurement, measurement)``
    error-correlation matrix) has the repeat(s) uniquely suffixed, plus the
    names of the variables that were touched.

    ``xr.concat`` — and most of xarray — cannot handle a variable with a
    repeated dim, even though such variables are valid to store in, and read
    from, a single file (some real EO products do this for e.g.
    error-correlation matrices). Splitting the repeat into a distinct dim
    name lets concat proceed; pair with :func:`_restore_repeated_dims` to put
    the original (repeated) dim name back afterwards.

    Mutates each touched variable's ``.dims`` directly (xarray's own
    recommended workaround for duplicate dims — ``Dataset.rename`` refuses,
    since it always validates the result has no duplicates) on a shallow
    copy, so the input ``ds`` and its underlying data arrays are untouched.
    """
    ds = ds.copy()
    touched: List[str] = []
    for name in list(ds.data_vars):
        dims = tuple(str(d) for d in ds[name].variable.dims)
        new_dims = _dedupe_dims(dims)
        if new_dims != dims:
            ds[name].variable.dims = new_dims
            touched.append(str(name))
    return ds, touched


def _restore_repeated_dims(ds: xr.Dataset, touched: List[str]) -> xr.Dataset:
    """
    Undo :func:`_split_repeated_dims`: strip the ``__dupN`` suffix back off
    each variable named in ``touched``, then rebuild the dataset so xarray's
    internal dims/sizes bookkeeping (which a direct ``.variable.dims``
    assignment does not refresh) reflects the restored state — otherwise the
    now-unused suffixed dim name lingers as a phantom entry in ``ds.dims``.
    """
    touched_present = [name for name in touched if name in ds.data_vars]
    if not touched_present:
        return ds
    for name in touched_present:
        dims = ds[name].variable.dims
        ds[name].variable.dims = tuple(str(d).split("__dup")[0] for d in dims)
    return xr.Dataset(dict(ds.data_vars), coords=dict(ds.coords), attrs=ds.attrs)


def _concat_datasets(
    datasets: List[xr.Dataset],
    paths: List[str],
    concat_dim: str,
    concat_coord: Optional[ConcatCoord],
    on_mismatch: str,
    attrs: str = "reconcile",
    coord_attrs: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Concatenate whole ``Dataset`` objects along ``concat_dim``, one per input file.

    This is the ``eoio.read_multi`` counterpart to the per-variable stacking
    ``StackCubes`` does within one dataset: same validation, attrs
    reconciliation, and concat primitives, applied one level up.

    Two concatenation cases:

    - **In-situ** (``concat_coord=None`` and ``concat_dim`` already a
      dimension on every dataset, e.g. a per-file ``time`` dimension):
      concatenated directly, no new dimension created.
    - **Multi-date raster stack** (``concat_dim`` not already present, or an
      explicit ``concat_coord`` given): a coordinate value is resolved per
      dataset (see :func:`_resolve_concat_coord_values`) and
      ``expand_dims({concat_dim: [value]})`` is applied before concatenating.

    :param datasets: Per-file datasets, in the same order as ``paths``.
    :param paths: Source path for each dataset (used in error/warning messages).
    :param concat_dim: Name of the dimension to concatenate along.
    :param concat_coord: See :func:`_resolve_concat_coord_values`.
    :param on_mismatch: ``"error"``, ``"skip"``, or ``"union"`` — see
        :func:`_validate_and_filter_datasets`. Also governs raster grid
        mismatches (a mismatched x/y shape is just another schema mismatch).

        Variables with a repeated dimension (e.g. a ``(measurement,
        measurement)`` error-correlation matrix) are transparently supported
        — see :func:`_split_repeated_dims` — even though ``xr.concat`` itself
        cannot handle them directly.
    :param attrs:
        How to combine dataset-level and per-variable attrs across files:

        - ``"reconcile"`` (default) — see :func:`_apply_attrs_reconciliation`:
          attrs identical across every file are kept; attrs that differ are
          kept as a per-file list attr (or promoted to a coordinate, see
          ``coord_attrs``) rather than silently dropped.
        - ``"first"`` — cheap legacy path: just keep the first file's attrs
          (``xr.concat``'s own default behaviour), discarding the rest.
    :param coord_attrs:
        Dataset-level attribute names to promote to a coordinate on
        ``concat_dim`` (one value per file, repeated across that file's
        ``concat_dim`` entries) instead of being kept as a list attr. Only
        used when ``attrs="reconcile"``.
    :returns: The concatenated dataset.
    """
    if on_mismatch not in ("error", "skip", "union"):
        raise ValueError(f"read_multi: on_mismatch must be one of 'error', 'skip', 'union'; got {on_mismatch!r}")
    if attrs not in ("reconcile", "first"):
        raise ValueError(f"read_multi: attrs must be 'reconcile' or 'first'; got {attrs!r}")

    datasets, paths = _validate_and_filter_datasets(datasets, paths, concat_dim, on_mismatch)

    # Split repeated dims (e.g. a (measurement, measurement) error-correlation
    # matrix) up front — expand_dims's internal transpose() rejects them just
    # as readily as xr.concat itself does.
    split_datasets = []
    touched_vars: set = set()
    for ds in datasets:
        split_ds, touched = _split_repeated_dims(ds)
        split_datasets.append(split_ds)
        touched_vars.update(touched)

    if concat_coord is None and all(concat_dim in ds.dims for ds in split_datasets):
        per_file_datasets = split_datasets
    else:
        coord_values = _resolve_concat_coord_values(split_datasets, paths, concat_dim, concat_coord)
        per_file_datasets = [
            ds if concat_dim in ds.dims else ds.expand_dims({concat_dim: [coord]})
            for ds, coord in zip(split_datasets, coord_values)
        ]

    # per_file_datasets is always List[xr.Dataset] here, so _obs_concat's
    # generic Union[DataArray, Dataset] return is always a Dataset in this
    # call site specifically.
    result = cast(xr.Dataset, _obs_concat(per_file_datasets, dim=concat_dim))

    if touched_vars:
        result = _restore_repeated_dims(result, list(touched_vars))

    if attrs == "reconcile":
        result = _apply_attrs_reconciliation(result, per_file_datasets, concat_dim, coord_attrs)

    return result


if __name__ == "__main__":
    pass
