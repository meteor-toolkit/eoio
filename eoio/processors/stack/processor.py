"""
eoio.processors.stack.processor
--------------------------------

Stack per-variable EO bands into cube-like DataArrays.

Many EO products are represented with one variable per band (e.g. B01, B02, …).
This processor groups variables that share the same physical quantity (``measurand``
attribute) and spatial grid (dimension names), then concatenates them along a new
stacking dimension to produce a single cube-like DataArray per group.

Variables without a ``measurand`` attribute are left unchanged.

Per-band ancillary variables (identified via CF ``ancillary_variables`` attributes)
are stacked automatically alongside their parent group, inheriting the parent's
stacking coordinates. This ensures e.g. per-band viewing angles share the same
wavelength axis as the reflectance cube they annotate.

Example output for a Sentinel-2 dataset::

    toa_reflectance_10m(stack_dim_10m, y_10m, x_10m)     # B02, B03, B04, B08
    viewing_zenith_angle_10m(stack_dim_10m, y_10m, x_10m) # VZA_B02, VZA_B03, …
    toa_reflectance_20m(stack_dim_20m, y_20m, x_20m)     # B05, B06, B07, …

The stacking dimension name includes the grid suffix so that cubes on different
grids use different dimension names and can coexist in the same Dataset.

User config example::

    processors = {
        "stack.cubes": {},
    }

    # Use wavelength values as the stacking coordinate (the original per-band
    # variable names, e.g. "B02", are kept too, as a "<stack_dim>_name" auxiliary
    # coordinate on the same dimension):
    processors = {
        "stack.cubes": {"dim_coord_attr": "central_wavelength"},
    }

    # Use a value nested inside a metadata dict:
    processors = {
        "stack.cubes": {"dim_coord_attr": "product_metadata.central_wavelength"},
    }

    # Use a custom stacking dimension name:
    processors = {
        "stack.cubes": {"stack_dim": "band"},
    }

    # Keep per-variable originals alongside the cubes:
    processors = {
        "stack.cubes": {"drop_originals": False},
    }
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import xarray as xr
from processor_tools import BaseProcessor

from eoio.processors.registry import register_processor
from eoio.processors.stack._concat import (
    _cube_attrs_and_coords,
    _get_nested_attr,
    _is_scalar_attr,  # noqa: F401 - re-exported for backwards-compatible imports
    _obs_concat,
)


@register_processor("stack")
class StackCubes(BaseProcessor):
    """
    Stack compatible per-variable EO bands into cube-like DataArrays.

    Variables are grouped by ``measurand`` attribute and spatial grid (inferred
    from dimension names). Each group with two or more variables is concatenated
    along a new stacking dimension.

    Per-band ancillary variables (those listed in constituent variables'
    ``ancillary_variables`` attributes, following CF conventions) are stacked
    automatically using the same coordinate values as their parent group. This
    allows e.g. per-band viewing angles to share a wavelength axis with the
    reflectance cube they annotate.

    By default, variables are sorted lexicographically by name and the stacking
    coordinate contains the original variable names. When ``dim_coord_attr`` is
    set, coordinate values are taken from the named attribute on each variable
    and the stacking order follows the attribute values (ascending); the original
    variable names are then kept too, as a same-dim ``<dim_name>_name`` auxiliary
    coordinate, so identity (e.g. ``"B02"``) is never lost in favour of the
    physical quantity (e.g. wavelength), or vice versa. If any variable in a
    group is missing the resolved ``dim_coord_attr`` value, the whole group falls
    back to variable names as the (sole) stacking coordinate -- see
    :py:func:`_sort_group`.

    Processor parameters
    --------------------

    :param stack_dim:
        Base name for the stacking dimension. The grid suffix is appended
        automatically so cubes on different grids get unique dimension names
        (e.g. ``"stack_dim"`` → ``"stack_dim_10m"``, ``"stack_dim_20m"``).
        Default: ``"stack_dim"``.
    :param dim_coord_attr:
        Attribute path (dot-separated) whose value is used as the stacking
        coordinate instead of the variable name. A simple key such as
        ``"central_wavelength"`` reads ``da.attrs["central_wavelength"]``;
        a dotted path such as ``"product_metadata.central_wavelength"``
        traverses nested dicts (tolerating a dict step that was already
        JSON-stringified, e.g. by a reader making it netCDF-safe -- it's parsed
        back before continuing). If any variable in a group is missing the
        resolved value, the group falls back to variable names as coordinates.
        When resolution succeeds, the original variable names are additionally
        kept as a ``<dim_name>_name`` auxiliary coordinate.
        Default: ``None`` (use variable names).
    :param drop_originals:
        If ``True`` (default), remove the original per-variable entries from the
        dataset after stacking. Set to ``False`` to keep them alongside the cube.
    :param stack_ancillaries:
        If ``True`` (default), automatically stack per-band ancillary variables
        (identified via CF ``ancillary_variables`` attributes) alongside their
        parent group using the parent's stacking coordinates.
    :param coord_attrs:
        List of per-band attribute names to promote to auxiliary coordinate
        variables on the stacking dimension. For example,
        ``["central_wavelength", "bandwidth"]`` makes those values accessible
        as labelled coordinates rather than list attrs. Attributes not listed
        here but that still differ across variables are kept as list attrs in
        stacking order. Default: ``None`` (no promotion).
    :param measurand_attr:
        Variable attribute used to identify the physical quantity. Default:
        ``"measurand"``. Change this if your reader uses a different attribute
        name (e.g. ``"standard_name"``).
    :param dim_attrs:
        Dictionary of attributes to set on the stacking dimension coordinate
        variable. Applied after any auto-detected attributes (e.g. units from
        a ``_unit`` sibling), so entries here take precedence. For example,
        ``{"long_name": "centre wavelength", "standard_name": "radiation_wavelength"}``.
        Default: ``None``.
    """

    _all_options = {
        "stack_dim": "Base name for the stacking dimension. The grid suffix is appended automatically so cubes on different grids get unique dimension names (e.g. 'stack_dim' -> 'stack_dim_10m', 'stack_dim_20m'). Default: 'stack_dim'.",
        "dim_coord_attr": "Attribute path (dot-separated) whose value is used as the stacking coordinate instead of the variable name. Default: None (use variable names).",
        "drop_originals": "If True (default), remove the original per-variable entries from the dataset after stacking. Set to False to keep them alongside the cube.",
        "stack_ancillaries": "If True (default), automatically stack per-band ancillary variables (identified via CF ancillary_variables attributes) alongside their parent group.",
        "coord_attrs": "List of per-band attribute names to promote to auxiliary coordinate variables on the stacking dimension. Default: None (no promotion).",
        "measurand_attr": "Variable attribute used to identify the physical quantity. Default: 'measurand'.",
        "dim_attrs": "Dictionary of attributes to set on the stacking dimension coordinate variable. Default: None.",
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(context=context)
        params = params or {}
        self.stack_dim: str = params.get("stack_dim", "stack_dim")
        self.dim_coord_attr: Optional[str] = params.get("dim_coord_attr", None)
        self.drop_originals: bool = params.get("drop_originals", True)
        self.stack_ancillaries: bool = params.get("stack_ancillaries", True)
        self.coord_attrs: Optional[List[str]] = params.get("coord_attrs", None)
        self.measurand_attr: str = params.get("measurand_attr", "measurand")
        self.dim_attrs: Dict[str, Any] = params.get("dim_attrs", None) or {}

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Stack compatible variables into cubes and return the updated dataset.

        :param ds: Input dataset.
        :returns: Dataset with cube variables added (and originals removed if
            ``drop_originals`` is ``True``).
        """
        all_ancillary_names = _collect_ancillary_names(ds) if self.stack_ancillaries else set()
        groups = self._collect_groups(ds, exclude=all_ancillary_names)
        vars_to_drop: List[str] = []

        for (measurand, dims), var_names in sorted(groups.items()):
            if len(var_names) < 2:
                continue

            ordered, coord_values = _sort_group(ds, var_names, self.dim_coord_attr)
            dim_name = _stack_dim_name(self.stack_dim, dims)
            cube = _obs_concat(
                [ds[name].expand_dims(dim={dim_name: [coord]}) for name, coord in zip(ordered, coord_values)],
                dim=dim_name,
            )

            ancillary_cube_names: List[str] = []
            if self.stack_ancillaries:
                anc_groups = _collect_per_band_ancillaries(ds, ordered)
                for anc_var_names in anc_groups:
                    anc_cube = _obs_concat(
                        [
                            ds[name].expand_dims(dim={dim_name: [coord]})
                            for name, coord in zip(anc_var_names, coord_values)
                        ],
                        dim=dim_name,
                    )
                    anc_measurand = ds[anc_var_names[0]].attrs.get(self.measurand_attr, "ancillary")
                    anc_out_name = _ancillary_output_var_name(anc_var_names, dims)
                    anc_attrs, anc_to_promote = _cube_attrs_and_coords(
                        ds, anc_var_names, anc_measurand, self.measurand_attr, self.coord_attrs
                    )
                    anc_cube.attrs = anc_attrs
                    for attr_name, values in anc_to_promote.items():
                        anc_cube = anc_cube.assign_coords({attr_name: (dim_name, values)})
                    ds[anc_out_name] = anc_cube
                    ancillary_cube_names.append(anc_out_name)
                    if self.drop_originals:
                        vars_to_drop.extend(anc_var_names)

            cube_attrs, to_promote = _cube_attrs_and_coords(
                ds, ordered, measurand, self.measurand_attr, self.coord_attrs
            )
            cube.attrs = cube_attrs
            for attr_name, values in to_promote.items():
                cube = cube.assign_coords({attr_name: (dim_name, values)})
            if self.dim_coord_attr:
                coord_units = _get_coord_units(ds[ordered[0]], self.dim_coord_attr)
                if coord_units is not None:
                    cube[dim_name].attrs["units"] = coord_units
            if self.dim_attrs:
                cube[dim_name].attrs.update(self.dim_attrs)
            if self.dim_coord_attr and ordered != coord_values:
                # dim_coord_attr resolved to something other than the variable names
                # themselves (e.g. a wavelength) -- keep the original per-variable names
                # too, as a same-dim auxiliary coordinate, so identity (e.g. "B02") isn't
                # silently lost in favour of the physical quantity, or vice versa. Skipped
                # when dim_coord_attr fell back to names (ordered == coord_values; see
                # _sort_group) since the two would be identical.
                name_coord = f"{dim_name}_name"
                cube = cube.assign_coords({name_coord: (dim_name, ordered)})
                cube[name_coord].attrs["long_name"] = "Original per-band variable name"
            if ancillary_cube_names:
                cube.attrs["ancillary_variables"] = " ".join(ancillary_cube_names)

            out_name = _output_var_name(measurand, dims)
            ds[out_name] = cube

            if self.drop_originals:
                vars_to_drop.extend(ordered)

        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        steps: list = ds.attrs.get("eoio:processing_steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]
        step: Dict[str, Any] = {"processor": "stack", "stack_dim": self.stack_dim}
        if self.dim_coord_attr:
            step["dim_coord_attr"] = self.dim_coord_attr
        steps.append(step)
        ds.attrs["eoio:processing_steps"] = steps

        return ds

    def _collect_groups(self, ds: xr.Dataset, exclude: Optional[Set[str]] = None) -> Dict[Tuple, List[str]]:
        """Group dataset variables by (measurand, dims), optionally excluding named variables."""
        exclude = exclude or set()
        groups: Dict[Tuple, List[str]] = defaultdict(list)
        for name, da in ds.data_vars.items():
            if str(name) in exclude:
                continue
            measurand = da.attrs.get(self.measurand_attr)
            if measurand is None:
                continue
            groups[(measurand, tuple(da.dims))].append(str(name))
        return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_ancillary_names(ds: xr.Dataset, ancillary_attr: str = "ancillary_variables") -> Set[str]:
    """Return the set of all variable names referenced as ancillary variables in the dataset."""
    names: Set[str] = set()
    for _, da in ds.data_vars.items():
        anc = da.attrs.get(ancillary_attr, [])
        if isinstance(anc, str):
            anc = anc.split()
        names.update(v for v in anc if v in ds)
    return names


def _collect_per_band_ancillaries(
    ds: xr.Dataset,
    ordered_var_names: List[str],
    ancillary_attr: str = "ancillary_variables",
) -> List[List[str]]:
    """
    Find sets of per-band ancillary variables corresponding to ``ordered_var_names``.

    A per-band ancillary is a variable listed in exactly one main variable's
    ``ancillary_variables`` attribute (i.e. not shared across all main variables).
    Ancillary variables are grouped by ``(measurand, dims)`` and only groups with
    exactly one ancillary per main variable are returned.

    Returns a list of lists, each inner list containing ancillary variable names
    in the same order as ``ordered_var_names``.
    """
    anc_lists: List[List[str]] = []
    for name in ordered_var_names:
        anc = ds[name].attrs.get(ancillary_attr, [])
        if isinstance(anc, str):
            anc = anc.split()
        anc_lists.append([v for v in anc if v in ds])

    if not all(anc_lists):
        return []

    shared = set(anc_lists[0]).intersection(*[set(a) for a in anc_lists[1:]])
    per_band_per_var = [[v for v in anc if v not in shared] for anc in anc_lists]

    if not all(per_band_per_var):
        return []

    # Group per-band ancillaries by (measurand, dims) across all main variables
    groups: Dict[Tuple, List[Tuple[int, str]]] = defaultdict(list)
    for i, var_ancs in enumerate(per_band_per_var):
        for anc_var in var_ancs:
            da = ds[anc_var]
            measurand = da.attrs.get("measurand", "ancillary")
            key = (measurand, tuple(da.dims))
            groups[key].append((i, anc_var))

    result: List[List[str]] = []
    for indexed_vars in groups.values():
        if len(indexed_vars) == len(ordered_var_names):
            result.append([v for _, v in sorted(indexed_vars)])

    return result


def _sort_group(
    ds: xr.Dataset,
    var_names: List[str],
    dim_coord_attr: Optional[str],
) -> Tuple[List[str], List[Any]]:
    """
    Return ``(ordered_names, coord_values)`` for a group.

    If ``dim_coord_attr`` is ``None``, variables are sorted lexicographically
    and the coordinate values are the variable names themselves.

    If ``dim_coord_attr`` is set, coordinate values are read from that
    attribute on each variable and the group is sorted by those values
    (ascending). If any variable is missing the attribute the group falls back
    to lexicographic sorting by variable name.
    """
    if dim_coord_attr is None:
        names = sorted(var_names)
        return names, names

    coord_values: List[Any] = []
    for name in var_names:
        val = _get_nested_attr(ds[name], dim_coord_attr)
        if val is None:
            warnings.warn(
                f"dim_coord_attr={dim_coord_attr!r} not found on variable {name!r}; "
                "falling back to variable names as stacking coordinates for this group."
            )
            names = sorted(var_names)
            return names, names
        coord_values.append(val)

    pairs = sorted(zip(coord_values, var_names))
    return [p[1] for p in pairs], [p[0] for p in pairs]


def _get_coord_units(da: xr.DataArray, coord_attr_path: str) -> Optional[str]:
    """
    Return the units for a coordinate attribute by appending ``_unit`` or
    ``_units`` to the last segment of ``coord_attr_path``.

    E.g. for ``"product_metadata.band_central_wavelength"``, tries
    ``"product_metadata.band_central_wavelength_unit"`` then
    ``"product_metadata.band_central_wavelength_units"``.

    Returns ``None`` if neither is found.
    """
    for suffix in ("_unit", "_units"):
        parts = coord_attr_path.rsplit(".", 1)
        if len(parts) == 2:
            candidate = f"{parts[0]}.{parts[1]}{suffix}"
        else:
            candidate = f"{parts[0]}{suffix}"
        val = _get_nested_attr(da, candidate)
        if val is not None:
            return str(val)
    return None


def _stack_dim_name(stack_dim: str, dims: Tuple[str, ...]) -> str:
    """
    Build a unique stacking dimension name for this group.

    Appends the grid suffix so that cubes on different grids use different
    dimension names and can coexist in the same Dataset without xarray
    attempting to align them. E.g. ``stack_dim="band"`` on a 10 m grid
    produces ``"band_10m"``; on a grid with no suffix produces ``"band"``.
    """
    suffix = _grid_suffix_from_dims(dims)
    return f"{stack_dim}_{suffix}" if suffix else stack_dim


def _grid_suffix_from_dims(dims: Tuple[str, ...]) -> Optional[str]:
    """
    Return the resolution suffix encoded in dimension names.

    E.g. ``('y_10m', 'x_10m')`` → ``'10m'``,
         ``('y_300m', 'x_300m')`` → ``'300m'``,
         ``('y', 'x')`` → ``None``.
    """
    for dim in dims:
        if dim.startswith("x_"):
            return dim[2:]
        if dim.startswith("y_"):
            return dim[2:]
    return None


def _output_var_name(measurand: str, dims: Tuple[str, ...]) -> str:
    """Build output variable name as ``{measurand}_{grid}`` or ``{measurand}``."""
    suffix = _grid_suffix_from_dims(dims)
    return f"{measurand}_{suffix}" if suffix else measurand


def _ancillary_output_var_name(anc_var_names: List[str], dims: Tuple[str, ...]) -> str:
    """
    Derive output variable name for a stacked ancillary group.

    Uses the common leading underscore-delimited name segments, so that
    band-specific suffixes are excluded. For example,
    ``["viewing_zenith_angle_B02", "viewing_zenith_angle_B03"]`` on a 10 m
    grid produces ``"viewing_zenith_angle_10m"``.
    """
    base = _common_segment_prefix(anc_var_names)
    suffix = _grid_suffix_from_dims(dims)
    return f"{base}_{suffix}" if suffix else base


def _common_segment_prefix(names: List[str]) -> str:
    """
    Return the longest common leading sequence of underscore-delimited segments.

    ``["viewing_zenith_angle_B02", "viewing_zenith_angle_B03"]``
    → ``"viewing_zenith_angle"``

    ``["abc", "xyz"]`` → ``"ancillary"``
    """
    if not names:
        return "ancillary"
    split = [name.split("_") for name in names]
    common: List[str] = []
    for segments in zip(*split):
        if len(set(segments)) == 1:
            common.append(segments[0])
        else:
            break
    return "_".join(common) or "ancillary"


if __name__ == "__main__":
    pass
