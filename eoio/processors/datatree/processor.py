"""
eoio.processors.datatree.processor
------------------------------------

Reorganise a multi-grid Dataset into an xarray.DataTree.

Many EO products contain variables on multiple spatial grids (e.g. Sentinel-2
bands at 10 m, 20 m and 60 m). Storing all of these in a single Dataset leads
to multiple sets of coordinates and dimensions that complicate spatial
operations with rioxarray/rasterio, which expect a single grid.

This processor groups variables by grid and places each group into a separate
DataTree node. Dimension names such as ``x_10m``/``y_10m`` are simplified to
``x``/``y`` within each node (controlled by ``rename_dims``).

**Note:** This processor returns an ``xr.DataTree``, not an ``xr.Dataset``.
It must be the final step in a processing pipeline.

Example output tree::

    /
    ├── 10m
    │   ├── B02(y, x)
    │   ├── B03(y, x)
    │   └── viewing_zenith_angle(band_10m, y, x)
    ├── 20m
    │   ├── B05(y, x)
    │   └── B06(y, x)
    └── 60m
        └── B01(y, x)

User config example::

    processors = {
        "to_datatree": {},
    }

    # Use a specific metadata attribute to identify the grid:
    processors = {
        "to_datatree": {"grid_attr": "product_metadata.geometry_id"},
    }

    # Keep original dimension names within each node:
    processors = {
        "to_datatree": {"rename_dims": False},
    }

The reverse operation ``from_datatree`` reassembles a DataTree produced by
this processor back into a flat Dataset.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr
from processor_tools import BaseProcessor

from eoio.processors.registry import register_processor


@register_processor("to_datatree")
class ToDataTree(BaseProcessor):
    """
    Reorganise a multi-grid Dataset into an xarray.DataTree.

    Variables are grouped by grid identity (resolved from metadata or
    dimension names) and placed into separate DataTree nodes. Each node
    contains only variables on a common grid.

    **This processor returns an** ``xr.DataTree`` **not an** ``xr.Dataset``.
    It must be the final step in a processing pipeline.

    Processor parameters
    --------------------

    :param grid_attr:
        Dot-separated attribute path used to identify which grid a variable
        belongs to (e.g. ``"product_metadata.geometry_id"``). If ``None``
        (default), the grid is inferred from dimension name suffixes
        (``x_10m``/``y_10m`` → ``"10m"``).
    :param rename_dims:
        If ``True`` (default), simplify grid-suffixed dimension names to
        plain ``x``/``y`` within each node, since the grid is now encoded
        in the node path. Set to ``False`` to preserve original dimension
        names.
    :param rename_vars:
        If ``True`` (default), strip the grid suffix from variable and
        non-spatial coordinate names that carry it. For example,
        ``reflectance_10m`` → ``reflectance`` and ``band_10m`` → ``band``
        within the ``/10m`` node. Names without the suffix are left unchanged.
        Set to ``False`` to preserve original names.
    """

    _all_options = {
        "rename_dims": "Optional, Boolean, if True (default) simplify grid-suffixed dimension names to plain x/y within each node. Set to False to preserve original dimension names.",
        "rename_vars": "Optional, Boolean, if True (default) strip the grid suffix from variable and non-spatial coordinate names that carry it. For example, reflectance_10m -> reflectance and band_10m -> band within the /10m node. Names without the suffix are left unchanged. Set to False to preserve original names.",
        "grid_attr": " Optional, Dot-separated attribute path used to identify which grid a variable belongs to (e.g. 'product_metadata.geometry_id'). If None (default), the grid is inferred from dimension name suffixes (x_10m/y_10m -> '10m').",
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(context=context)
        params = params or {}
        self.grid_attr: Optional[str] = params.get("grid_attr", None)
        self.rename_dims: bool = params.get("rename_dims", True)
        self.rename_vars: bool = params.get("rename_vars", True)

    def run(self, ds: xr.Dataset) -> xr.DataTree:  # type: ignore[override]
        """
        Reorganise the dataset into a DataTree grouped by grid.

        :param ds: Input multi-grid dataset.
        :returns: DataTree with one node per grid.
        """
        steps: list = ds.attrs.get("eoio:processing_steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]
        steps.append(
            {
                "processor": "to_datatree",
                "grid_attr": self.grid_attr,
                "rename_dims": self.rename_dims,
                "rename_vars": self.rename_vars,
            }
        )
        ds.attrs["eoio:processing_steps"] = steps

        return build_datatree(
            ds,
            grid_attr=self.grid_attr,
            rename_dims=self.rename_dims,
            rename_vars=self.rename_vars,
        )


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def build_datatree(
    ds: xr.Dataset,
    *,
    grid_attr: Optional[str] = None,
    rename_dims: bool = True,
    rename_vars: bool = True,
) -> xr.DataTree:
    """
    Reorganise a multi-grid Dataset into an xarray.DataTree.

    Variables without a detectable grid are placed at the root node alongside
    the global dataset attributes.

    :param ds: Input dataset.
    :param grid_attr:
        Dot-separated attribute path used to identify the grid for each
        variable. Falls back to dimension name suffix detection if ``None``
        or if the attribute is absent on a variable.
    :param rename_dims:
        Rename grid-suffixed spatial dimensions (``x_10m`` → ``x``) within
        each node.
    :param rename_vars:
        Strip the grid suffix from variable and non-spatial coordinate names
        that carry it (``reflectance_10m`` → ``reflectance``).
    :returns: DataTree with one child node per grid.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for name, da in ds.data_vars.items():
        grid = _detect_grid(da, grid_attr)
        groups[grid].append(str(name))

    root_var_names = groups.pop("", [])
    root_ds = ds[root_var_names] if root_var_names else xr.Dataset()
    root_ds.attrs = dict(ds.attrs)

    tree_dict: Dict[str, xr.Dataset] = {"/": root_ds}
    for grid_id, var_names in sorted(groups.items()):
        node_ds = ds[var_names]
        node_ds.attrs = dict(node_ds.attrs)  # independent copy -- see _rescope_per_band_attrs
        node_ds = _rescope_per_band_attrs(ds, node_ds, var_names)
        if rename_dims:
            node_ds = _rename_grid_dims(node_ds, grid_id)
        if rename_vars:
            node_ds = _rename_grid_vars(node_ds, grid_id)
        tree_dict[f"/{grid_id}"] = node_ds

    return xr.DataTree.from_dict(tree_dict)


#: Dataset-level attrs that :func:`eoio.readers.metadata.BaseMetadataExtractor.extract_metadata`
#: sets once per product, as one list entry per originally-selected measurement band (see
#: its own docstring) -- i.e. scoped to the *whole, still-flat* dataset, not to any one
#: grid. ``node_ds = ds[var_names]`` above inherits ``ds.attrs`` wholesale (xarray's own
#: dataset-subsetting doesn't know these particular attrs are per-band), so left alone
#: every split node would carry the same whole-product list regardless of which bands it
#: actually holds. Maps each such dataset-level attr to the per-variable attr name that
#: correctly describes a single variable (the same ``product_metadata`` source
#: :func:`_detect_grid` already trusts for ``geometry_id``).
_PER_VARIABLE_RESCOPED_ATTRS: Dict[str, str] = {
    "spatial_resolution": "spatial_resolution",
    "geometry_ids": "geometry_id",
}


def _rescope_per_band_attrs(ds: xr.Dataset, node_ds: xr.Dataset, var_names: List[str]) -> xr.Dataset:
    """Recompute *node_ds*'s :py:data:`_PER_VARIABLE_RESCOPED_ATTRS` from *var_names*'
    own per-variable metadata on *ds*, instead of leaving the whole-product value it
    inherited via ``ds[var_names]``.

    A value found on every one of *var_names* collapses to a single scalar (the normal
    case -- one grid's worth of bands all share one resolution); several distinct values
    are kept as a sorted list (only possible with a custom ``grid_attr`` that doesn't
    correspond to spatial_resolution/geometry_id, since the real grid detection paths
    guarantee a uniform resolution within one node). If no variable in *var_names* carries
    the per-variable attr at all, the stale whole-product value is dropped rather than
    propagated -- a missing attr is honest; a wrong one is not.

    :param ds: the original, still-flat dataset (looked up by *var_names*' original names).
    :param node_ds: *ds*'s subset for this grid -- **mutated in place** (its ``attrs``
        must already be an independent dict, not shared with *ds* or another node).
    :param var_names: this node's own variable names, in *ds*.
    :return: *node_ds*.
    """
    for dataset_attr, variable_attr in _PER_VARIABLE_RESCOPED_ATTRS.items():
        if dataset_attr not in node_ds.attrs:
            continue
        values = []
        for name in var_names:
            da = ds[name]
            val = _get_nested_attr(da, f"product_metadata.{variable_attr}")
            if val is None:
                val = da.attrs.get(variable_attr)
            if val is not None:
                values.append(val)
        if not values:
            del node_ds.attrs[dataset_attr]
            continue
        unique = sorted({v for v in values}, key=str)
        node_ds.attrs[dataset_attr] = unique[0] if len(unique) == 1 else unique
    return node_ds


@register_processor("from_datatree")
class FromDataTree(BaseProcessor):
    _all_options = {}

    def __init__(self, context: Any | None = None, processor_path: str | None = None, **kwargs):
        super().__init__(context, processor_path, **kwargs)

    def run(self, data: xr.DataTree) -> xr.Dataset:  # type: ignore
        if not isinstance(data, xr.DataTree):
            raise TypeError("from_datatree requires a DataTree input")

        ds = from_datatree(data)

        steps = ds.attrs.get("eoio:processing_steps", [])
        steps.append({"processor": "from_datatree"})
        ds.attrs["eoio:processing_steps"] = steps

        return ds


def from_datatree(dt: xr.DataTree) -> xr.Dataset:
    """
    Reassemble a DataTree produced by :func:`build_datatree` into a flat Dataset.

    Restores grid-suffixed dimension names (``x`` → ``x_<grid>``) so the
    merged dataset can coexist in one flat namespace. Global attrs from the
    root are preserved.

    Note: variable name renaming (``rename_vars=True``) is not reversed —
    it is a lossy simplification. ``reflectance`` in ``/10m`` stays as
    ``reflectance`` after flattening.

    :param dt: DataTree to flatten.
    :returns: Merged Dataset with grid-suffixed names restored.
    """
    merged = dt.root.dataset.copy()

    for path, node in dt.subtree_with_keys:
        if path == "":
            continue
        grid_id = path.lstrip("/")
        node_ds = node.dataset

        # Restore spatial dims
        dim_rename: Dict[str, str] = {}
        if "x" in node_ds.dims:
            dim_rename["x"] = f"x_{grid_id}"
        if "y" in node_ds.dims:
            dim_rename["y"] = f"y_{grid_id}"
        if dim_rename:
            node_ds = node_ds.rename(dim_rename)

        for var_name, da in node_ds.data_vars.items():
            merged[str(var_name)] = da

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_grid(da: xr.DataArray, grid_attr: Optional[str] = None) -> str:
    """
    Return a grid identifier string for a DataArray.

    Resolution order:
    1. ``grid_attr`` parameter (dot-notation attribute path).
    2. ``product_metadata.geometry_id`` (eoio S2 convention).
    3. Dimension name suffix (``x_10m`` → ``"10m"``).
    4. Empty string (no grid — variable goes to root).
    """
    if grid_attr:
        val = _get_nested_attr(da, grid_attr)
        if val is not None:
            return str(val)

    pm = da.attrs.get("product_metadata")
    if isinstance(pm, dict):
        geom_id = pm.get("geometry_id")
        if geom_id is not None:
            return str(geom_id)

    suffix = _grid_suffix_from_dims(tuple(str(d) for d in da.dims))
    if suffix:
        return suffix

    return ""


def _grid_suffix_from_dims(dims: Tuple[str, ...]) -> Optional[str]:
    """Return the resolution suffix encoded in dimension names, or None."""
    for dim in dims:
        if dim.startswith("x_"):
            return dim[2:]
        if dim.startswith("y_"):
            return dim[2:]
    return None


def _rename_grid_vars(ds: xr.Dataset, grid_id: str) -> xr.Dataset:
    """
    Strip the grid suffix from variable and non-spatial coordinate names.

    Only names that end with ``_{grid_id}`` are renamed. Spatial dimensions
    (``x``, ``y``, ``x_<grid>``, ``y_<grid>``) are excluded since
    :func:`_rename_grid_dims` handles those.

    E.g. for ``grid_id="10m"``:
    ``reflectance_10m`` → ``reflectance``, ``band_10m`` → ``band``.
    """
    suffix = f"_{grid_id}"
    spatial = {"x", "y", f"x_{grid_id}", f"y_{grid_id}"}

    var_rename: Dict[str, str] = {
        str(name): str(name)[: -len(suffix)]
        for name in list(ds.data_vars) + list(ds.coords)
        if str(name).endswith(suffix) and str(name) not in spatial
    }
    return ds.rename(var_rename) if var_rename else ds


def _restore_grid_vars(ds: xr.Dataset, grid_id: str) -> xr.Dataset:
    """
    Re-append the grid suffix to variable and coordinate names that lack it.

    Called by :func:`from_datatree` to reverse :func:`_rename_grid_vars`.
    Only renames if adding the suffix produces a name that does not already
    exist, avoiding double-suffixing variables that were never renamed.
    """
    suffix = f"_{grid_id}"
    spatial = {"x", "y"}

    var_rename: Dict[str, str] = {}
    all_names = set(str(n) for n in list(ds.data_vars) + list(ds.coords))
    for name in list(ds.data_vars) + list(ds.coords):
        sname = str(name)
        if sname in spatial or sname.endswith(suffix):
            continue
        candidate = sname + suffix
        if candidate not in all_names:
            var_rename[sname] = candidate

    return ds.rename(var_rename) if var_rename else ds


def _rename_grid_dims(ds: xr.Dataset, grid_id: str) -> xr.Dataset:
    """
    Rename grid-suffixed spatial dimensions to plain ``x``/``y``.

    E.g. for ``grid_id="10m"``: ``x_10m`` → ``x``, ``y_10m`` → ``y``.
    Only renames if the suffixed names are present; leaves other dims unchanged.
    """
    rename_map: Dict[str, str] = {}
    if f"x_{grid_id}" in ds.dims:
        rename_map[f"x_{grid_id}"] = "x"
    if f"y_{grid_id}" in ds.dims:
        rename_map[f"y_{grid_id}"] = "y"
    return ds.rename(rename_map) if rename_map else ds


def _get_nested_attr(da: xr.DataArray, attr_path: str) -> Any:
    """Retrieve a variable attribute using dot-notation for nested dicts."""
    val: Any = da.attrs
    for key in attr_path.split("."):
        if not isinstance(val, dict):
            return None
        val = val.get(key)
        if val is None:
            return None
    return val


if __name__ == "__main__":
    pass
