"""
eoio.processors.interpolate.processor
-------------------------------

Top-level interpolation processor.

This processor provides a single, stable user-facing interface (``interpolate``).

User config example
-------------------
processors = {
    "interpolate": {
      "coords": ["x_5000m", "y_5000m"],
      "target_grid": ["x_60m", "y_60m"],
      "method": "linear",
      },
    }
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from processor_tools import BaseProcessor
import xarray as xr
from eoio.processors.registry import register_processor
import numpy as np
from scipy.interpolate import griddata


@dataclass(frozen=True)
class InterpolateConfig:
    """
    Parameters for the interpolate processor.
    """

    coords: Sequence[str]
    interpolate_mode: str
    target_grid: Sequence
    data_vars: Optional[Sequence[str]] = None
    method: str = "linear"
    inplace: Optional[bool] = False
    on_missing: str = "error"  # "error" | "skip"


@register_processor("interpolate")
class Interpolate(BaseProcessor):
    """
    Interpolate measurement variables in a dataset to a new coordinate grid.

    Processor parameters
    --------------------

    The following parameters can be provided in the `params` dict:

    :param coords:
        List of coordinate names to interpolate along (e.g. ``["x", "y"]``).
    :param target_grid:
        List of target coordinate values to interpolate to.
        If str, must be the name of an existing coordinate in the dataset (e.g. ``"x_60m"``).
        If xr.DataArray, new coord name will be set to name of DataArray.
        If other (np.Array, list), coord name will be unchanged (e.g. ``"x_5000m"`` or ``"x_5000m_interp"`` depending on value of 'inplace').
    :param data_vars:
        List of variable names to interpolate. If omitted, all variables with any of the interpolation coordinates as a dimension will be interpolated.
    :param method:
        Interpolation method (e.g. ``"linear"``, ``"nearest"``, etc.).
        Must be supported by the underlying interpolation implementation (e.g. xarray.interp()).
    :param inplace:
        Bool, if False, new interpolated variables are added to the dataset with the suffix '_interp' (e.g. 'B02_interp'). If True, original variables are replaced by interpolated ones. Default is False.
    :param on_missing:
        Behaviour if required metadata for conversion is missing.
        Supported values are ``"error"`` (default, if omitted) or ``"skip"``.

    Notes
    -----
    - This processor is intended to run after reading.
    """

    _all_options = {
        "coords": "List of coordinate names to interpolate along (e.g. ['x', 'y']).",
        "target_grid": "List of target coordinate values to interpolate to. If str, must be the name of an existing coordinate in the dataset (e.g. 'x_60m'). If xr.DataArray, new coord name will be set to name of DataArray. If other (np.Array, list), coord name will be unchanged (e.g. 'x_5000m' or 'x_5000m_interp' depending on value of 'inplace').",
        "data_vars": "List of variable names to interpolate. If omitted, all variables with any of the interpolation coordinates as a dimension will be interpolated.",
        "method": "Interpolation method (e.g. 'linear', 'nearest', etc.). Must be supported by the underlying interpolation implementation (e.g. xarray.interp()).",
        "inplace": "Bool, if False, new interpolated variables are added to the dataset with the suffix '_interp' (e.g. 'B02_interp'). If True, original variables are replaced by interpolated ones. Default is False.",
        "on_missing": "Behaviour if required metadata for conversion is missing. Supported values are 'error' (default, if omitted) or 'skip'.",
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an interpolation processor.

        :param params:
            Processor parameters (see class docstring for details)
        :param context:
            Processing context provided by eoio (reader info, metadata view, logger, etc.).
        """

        super().__init__(context=context)
        self.interpolate_config = self._parse_params(params or {})

    def _parse_params(self, params: Dict[str, Any]) -> InterpolateConfig:
        """
        Validate and normalise processor parameters.

        :param params:
            Raw params dict from the processor spec.
        :return:
            Parsed InterpolateParams.
        :raises ValueError:
            If required parameters are missing or invalid.
        """

        # resolve "coords" param
        if "coords" not in params:
            raise ValueError("interpolate: missing required param 'coords' (e.g. ['x_5000m']).")

        coords = params["coords"]

        if any([("x" in c) or ("y" in c) for c in coords]):
            interpolate_mode = "xy"
        elif any([("latitude" in c) or ("longitude" in c) for c in coords]):
            interpolate_mode = "lat_lon"
        else:
            raise ValueError(
                "interpolate: unable to determine interpolation mode from 'coords' parameter. Expected coordinate names to contain 'x_'/'y_' for xy interpolation or 'latitude'/'longitude' for lat/lon interpolation."
            )

        # resolve "target_grid" param
        if "target_grid" not in params:
            raise ValueError("interpolate: missing required param 'target_grid' (e.g. ['x_60m']).")

        target_grid = params["target_grid"]

        # resolve "data_vars" param
        data_vars = params.get("data_vars", [])

        # resolve "method" param
        method = str(params.get("method", "linear")).lower()

        if len(coords) != len(target_grid):
            raise ValueError("interpolate: 'coords', 'target_grid' must have the same length.")
        # resolve inplace param
        inplace = bool(params.get("inplace", False))
        if inplace is True and any(isinstance(tg, list) and isinstance(tg[0], str) for tg in target_grid):
            raise ValueError(
                "interpolate: 'inplace' must be False when using list of coordinate names in 'target_grid'."
            )

        # resolve "on_missing" param
        on_missing = str(params.get("on_missing", "error")).lower()
        if on_missing not in {"error", "skip"}:
            raise ValueError("interpolate: 'on_missing' must be 'error' or 'skip'.")

        return InterpolateConfig(
            coords=coords,
            target_grid=target_grid,
            data_vars=data_vars,
            method=method,
            inplace=inplace,
            on_missing=on_missing,
            interpolate_mode=interpolate_mode,
        )

    def _format_target_grid(self, ds: xr.Dataset, target_grid: Sequence) -> Sequence:
        """
        Format the target grid to ensure it is in the correct format for interpolation.

        :param ds:
            Input dataset (used for context, e.g. to check coordinate types).
        :param target_grid:
            Target coordinate values to interpolate to.
        :return:
            Formatted target grid.
        """
        formatted_targets: list = []
        for target in target_grid:
            if isinstance(target, str):
                if target not in ds.coords:
                    raise ValueError(f"interpolate: target_grid '{target}' not found in dataset coordinates.")
                formatted_targets.append(ds.coords[target])
            elif isinstance(target, List) and isinstance(target[0], str):
                tar_list = []
                for tar in target:
                    if tar not in ds.coords:
                        raise ValueError(f"interpolate: target_grid '{tar}' not found in dataset coordinates.")
                    tar_list.append(ds.coords[tar])
                formatted_targets.append(tar_list)
            else:
                formatted_targets.append(target)

        return formatted_targets

    def _format_coords(self, ds: xr.Dataset, coords: Sequence) -> Sequence:
        """
        Format the coordinates to ensure they are in the correct format for interpolation.

        :param ds:
            Input dataset (used for context, e.g. to check available coordinates).
        :param coords:
            List of coordinate names to interpolate along (e.g. ``["x", "y"]``).
        """

        # Check all coords exist in the dataset
        for coord in coords:
            if coord not in ds.coords:
                raise ValueError(f"interpolate: coordinate '{coord}' not found in dataset.")

        # Format the coordinates
        return coords

    def _run_xy(self, ds: xr.Dataset, coords, target_coords) -> xr.Dataset:
        """
        Run interpolation on x/y coordinates.

        :param ds:
            Input dataset.
        :return:
            Output dataset with interpolated variables.
        """

        vars_with_coord = [name for name, da in ds.data_vars.items() if set(coords).intersection(da.dims)]
        # choose which variables to interpolate based on 'data_vars' config (if provided), otherwise use all variables with the interpolation coords as dimensions
        if self.interpolate_config.data_vars:
            vars_to_interpolate = list(set(vars_with_coord).intersection(self.interpolate_config.data_vars))
        else:
            vars_to_interpolate = vars_with_coord

        # if inplace is False, create new variables with '_interp' suffix and assign them to the dataset, and also create new coordinates with '_interp' suffix, then interpolate in place on the new variables.
        if not self.interpolate_config.inplace:
            # Check if we have list-of-coordinate-names targets (vs custom grids)
            has_list_of_coords = any(
                isinstance(tc, list) and all(isinstance(item, xr.DataArray) for item in tc) for tc in target_coords
            )

            if has_list_of_coords:
                # Add new placeholder coordinates for each target in target_coords
                placeholder_coords = {}  # Maps old coord name to list of placeholder names

                for coord, target_coord in zip(coords, target_coords):
                    if isinstance(target_coord, list) and all(isinstance(item, xr.DataArray) for item in target_coord):
                        # Create a placeholder for each target in the list
                        placeholders = [f"{coord}_interp_{i}" for i in range(len(target_coord))]
                        for placeholder in placeholders:
                            ds = ds.assign_coords({placeholder: ds[coord]})
                        placeholder_coords[coord] = placeholders
                    else:
                        # Single target or custom grid, use standard _interp suffix
                        ds = ds.assign_coords({coord + "_interp": ds[coord]})
                        placeholder_coords[coord] = [coord + "_interp"]

                # For each variable to interpolate, create new versions with swapped dims
                for var in vars_to_interpolate:
                    old_var = ds[var]
                    # Build complete dimension mapping for all placeholders at once
                    max_placeholders = max(len(p) for p in placeholder_coords.values())

                    for i in range(max_placeholders):
                        # Build full dim mapping for this interpolation step
                        dim_mapping = {}
                        for coord, placeholders in placeholder_coords.items():
                            if coord in old_var.dims and i < len(placeholders):
                                dim_mapping[coord] = placeholders[i]

                        if dim_mapping:  # Only create if there are dimensions to map
                            new_var = old_var.swap_dims(dim_mapping)
                            ds[f"{var}_interp_{i}"] = new_var

                # Update coords and vars_to_interpolate to use the new placeholders and interpolated vars
                new_coords: list = []
                new_vars_to_interpolate: list[str] = []
                for coord, placeholders in placeholder_coords.items():
                    new_coords.append(placeholders)
                for var in vars_to_interpolate:
                    for i in range(max(len(p) for p in placeholder_coords.values())):
                        new_vars_to_interpolate.append(f"{var}_interp_{i}")
                coords = new_coords
                vars_to_interpolate = new_vars_to_interpolate
            else:
                # Custom grids: use simpler logic with single _interp suffix
                inplace_coords = {coord + "_interp": ds[coord] for coord in coords}
                ds = ds.assign_coords(coords=inplace_coords)

                # For each variable to interpolate, create a new version with swapped dims
                for var in vars_to_interpolate:
                    old_var = ds[var]
                    # Swap dims: replace old coord names with new coord names
                    dim_mapping = {coord: coord + "_interp" for coord in coords if coord in old_var.dims}
                    new_var = old_var.swap_dims(dim_mapping)
                    # Add to dataset with _interp suffix
                    ds[var + "_interp"] = new_var

                coords = [coord + "_interp" for coord in coords]
                vars_to_interpolate = [var + "_interp" for var in vars_to_interpolate]

        # If inplace is True, interpolate on the original variables.

        # Build list of (coord, target_coord) pairs, expanding lists of coordinate names only
        interpolation_pairs = []
        using_indexed_placeholders = any(
            isinstance(tc, list) and all(isinstance(item, xr.DataArray) for item in tc) for tc in target_coords
        )

        for coord, target_coord in zip(coords, target_coords):
            if isinstance(target_coord, list) and all(isinstance(item, xr.DataArray) for item in target_coord):
                # List of coordinate names: expand into individual pairs
                for individual_coord, individual_target in zip(coord, target_coord):
                    interpolation_pairs.append((individual_coord, individual_target))
            else:
                # Single target (string or custom grid): add as-is
                interpolation_pairs.append((coord, target_coord))

        # perform interpolation
        for coord, target_coord in interpolation_pairs:
            interp_input = {coord: target_coord}

            # Extract index from coordinate name only if using indexed placeholders
            coord_index = None
            if using_indexed_placeholders and "_interp_" in coord:
                try:
                    coord_index = int(coord.split("_interp_")[-1])
                except (ValueError, IndexError):
                    pass

            if vars_to_interpolate == vars_with_coord:
                ds = ds.interp(coords=interp_input, method=self.interpolate_config.method)  # type: ignore[arg-type]
            else:
                for var in vars_to_interpolate:
                    # If using indexed placeholders, only interpolate variables with matching index
                    if coord_index is not None:
                        if "_interp_" not in var or not var.endswith(str(coord_index)):
                            continue
                    ds[var] = ds[var].interp(coords=interp_input, method=self.interpolate_config.method)  # type: ignore[arg-type]
        # rename new variables with resolution of new grid as suffix (e.g. "B02_interp_60m" instead of "B02_interp_0") - only if using coordinate-name lists
        if using_indexed_placeholders:
            for var in vars_to_interpolate:
                if "_interp_" in var:
                    # Extract the resolution from the target coordinate
                    target_resolution = None
                    for coord, target_coord in zip(coords, target_coords):
                        if isinstance(target_coord, list):
                            for individual_coord, individual_target in zip(coord, target_coord):
                                if (
                                    isinstance(individual_target, (xr.DataArray, str))
                                    and (
                                        individual_coord.split("_")[-2:][0] + "_" + individual_coord.split("_")[-2:][1]
                                    )
                                    in var
                                ):
                                    target_resolution = (
                                        individual_target.name
                                        if isinstance(individual_target, xr.DataArray)
                                        else individual_target
                                    )
                                    break
                        elif (
                            isinstance(target_coord, (xr.DataArray, str))
                            and (coord.split("_")[-2:][0] + "_" + coord.split("_")[-2:][1]) in var
                        ):
                            target_resolution = (
                                target_coord.name if isinstance(target_coord, xr.DataArray) else target_coord
                            )
                            break
                    if target_resolution:
                        new_var_name = f"{str(var).split('_interp', 1)[0]}_{str(target_resolution).split('_')[-1]}"
                        ds = ds.rename({var: new_var_name})

        # clean up dataset, removing placeholder coords
        for coord, target_coord in zip(coords, target_coords):
            if isinstance(target_coord, (xr.DataArray, str)) and coord not in ds.dims:
                ds = ds.reset_coords(names=coord, drop=True)

        return ds

    def _run_lat_lon(self, ds: xr.Dataset, coords, target_coords) -> xr.Dataset:
        """
        Run interpolation on latitude/longitude coordinates.

        :param ds:
            Input dataset.
        :return:
            Output dataset with interpolated variables.
        """
        vars_with_coord = [
            var
            for var in ds.data_vars
            if set([item for c in coords for item in ds[c].dims]).intersection(set(ds[var].dims))
        ]

        # choose which variables to interpolate based on 'data_vars' config (if provided), otherwise use all variables with the interpolation coords as dimensions
        if self.interpolate_config.data_vars:
            vars_to_interpolate = list(set(vars_with_coord).intersection(self.interpolate_config.data_vars))
        else:
            vars_to_interpolate = vars_with_coord

            # if inplace is False, create new variables with '_interp' suffix and assign them to the dataset, and also create new coordinates with '_interp' suffix, then interpolate in place on the new variables.
        if not self.interpolate_config.inplace:
            # Check if we have list-of-coordinate-names targets (vs custom grids)
            has_list_of_coords = any(
                isinstance(tc, list) and all(isinstance(item, xr.DataArray) for item in tc) for tc in target_coords
            )

            if has_list_of_coords:
                # Add new placeholder coordinates for each target in target_coords
                placeholder_coords = {}  # Maps old coord name to list of placeholder names

                for coord, target_coord in zip(coords, target_coords):
                    if isinstance(target_coord, list) and all(isinstance(item, xr.DataArray) for item in target_coord):
                        # Create a placeholder for each target in the list
                        placeholders = [f"{coord}_interp_{i}" for i in range(len(target_coord))]
                        for placeholder in placeholders:
                            ds = ds.assign_coords({placeholder: ds[coord]})
                        placeholder_coords[coord] = placeholders
                    else:
                        # Single target or custom grid, use standard _interp suffix
                        ds = ds.assign_coords({coord + "_interp": ds[coord]})
                        placeholder_coords[coord] = [coord + "_interp"]

                # For each variable to interpolate, create new versions with swapped dims
                for var in vars_to_interpolate:
                    old_var = ds[var]
                    # Build complete dimension mapping for all placeholders at once
                    max_placeholders = max(len(p) for p in placeholder_coords.values())

                    for i in range(max_placeholders):
                        # Build full dim mapping for this interpolation step
                        dim_mapping = {}
                        for coord, placeholders in placeholder_coords.items():
                            if coord in old_var.dims and i < len(placeholders):
                                dim_mapping[coord] = placeholders[i]

                        if dim_mapping:  # Only create if there are dimensions to map
                            new_var = old_var.swap_dims(dim_mapping)
                            ds[f"{var}_interp_{i}"] = new_var

                # Update coords and vars_to_interpolate to use the new placeholders and interpolated vars
                new_coords = []
                new_vars_to_interpolate = []
                for coord, placeholders in placeholder_coords.items():
                    new_coords.append(placeholders)
                for var in vars_to_interpolate:
                    for i in range(max(len(p) for p in placeholder_coords.values())):
                        new_vars_to_interpolate.append(f"{var}_interp_{i}")
                coords = new_coords
                vars_to_interpolate = new_vars_to_interpolate
            else:
                # Custom grids: use simpler logic with single _interp suffix
                inplace_coords = {coord + "_interp": ds[coord] for coord in coords}
                ds = ds.assign_coords(coords=inplace_coords)

                # For each variable to interpolate, create a new version with swapped dims
                for var in vars_to_interpolate:
                    old_var = ds[var]
                    # Swap dims: replace old coord names with new coord names
                    dim_mapping = {coord: coord + "_interp" for coord in coords if coord in old_var.dims}
                    new_var = old_var.swap_dims(dim_mapping)
                    # Add to dataset with _interp suffix
                    ds[var + "_interp"] = new_var

                coords = [coord + "_interp" for coord in coords]
                vars_to_interpolate = [var + "_interp" for var in vars_to_interpolate]

        # If inplace is True, interpolate on the original variables.

        # Build list of (coord, target_coord) pairs, expanding lists of coordinate names only
        interpolation_pairs = {}
        interpolation_groups = []  # To track which coords belong together for lat/lon interpolation
        using_indexed_placeholders = any(
            isinstance(tc, list) and all(isinstance(item, xr.DataArray) for item in tc) for tc in target_coords
        )

        for coord, target_coord in zip(coords, target_coords):
            if isinstance(target_coord, list) and all(isinstance(item, xr.DataArray) for item in target_coord):
                # List of coordinate names: expand into individual pairs
                for individual_coord, individual_target in zip(coord, target_coord):
                    interpolation_pairs[individual_coord] = individual_target
            else:
                # Single target (string or custom grid): add as-is
                interpolation_pairs[coord] = target_coord

        for coord, target_coord in interpolation_pairs.items():
            if "latitude" in coord:
                interpolation_groups.append(
                    (
                        [coord, coord.replace("latitude", "longitude")],
                        [target_coord, interpolation_pairs.get(coord.replace("latitude", "longitude"))],
                    )
                )

        # perform interpolation
        for cs, tcs in interpolation_groups:
            points = np.column_stack((ds[cs[1]].values.ravel(), ds[cs[0]].values.ravel()))
            lon_target = tcs[1].values
            lat_target = tcs[0].values
            for var in vars_to_interpolate:
                values = ds[var].values.ravel()
                target_resolution = tcs[0].name.split("_")[-1]

                mask = ~np.isnan(values)
                points_valid = points[mask]
                values_valid = values[mask]

                interp_data = griddata(
                    points_valid, values_valid, (lon_target, lat_target), method=self.interpolate_config.method
                )

                ds[var] = (("y_grid_" + target_resolution, "x_grid_" + target_resolution), interp_data)

                if not self.interpolate_config.inplace:
                    ds = ds.rename({var: f"{var.split('_interp', 1)[0]}_{target_resolution}"})

        # clean up dataset, removing placeholder coords
        for coord, target_coord in zip(coords, target_coords):
            if isinstance(target_coord, (xr.DataArray, str)) and coord not in ds.dims:
                ds = ds.reset_coords(names=coord, drop=True)

        return ds

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Run interpolation on the dataset.

        :param ds:
            Input dataset.
        :return:
            Output dataset with interpolated variables.
        """

        if not isinstance(ds, xr.Dataset):
            raise TypeError("interpolate: input must be an xarray.Dataset.")

        context: Mapping[str, Any] = self.context or {}

        coords = self._format_coords(ds, self.interpolate_config.coords)
        target_coords = self._format_target_grid(ds, self.interpolate_config.target_grid)
        if self.interpolate_config.interpolate_mode == "xy":
            ds = self._run_xy(ds, coords, target_coords)
        elif self.interpolate_config.interpolate_mode == "lat_lon":
            ds = self._run_lat_lon(ds, coords, target_coords)
        else:
            raise ValueError("interpolate: invalid 'interpolate_mode'. Expected 'xy' or 'lat_lon'.")
        # Record processing history
        ds = self._record_provenance(ds)

        return ds

    def _record_provenance(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Record a minimal provenance entry at processor level.

        :param ds:
            Output dataset.
        :return:
            Dataset (same object, attrs updated).
        """

        steps: list = ds.attrs.get("eoio:processing_steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]

        steps.append(
            {
                "processor": "interpolate",
                "interpolated_coords": self.interpolate_config.coords,
            }
        )
        ds.attrs["eoio:processing_steps"] = steps
        return ds


if __name__ == "__main__":
    pass
