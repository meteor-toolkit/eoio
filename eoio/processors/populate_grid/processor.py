"""
eoio.processors.populate_grid.processor
-------------------------------

Top-level populate_grid processor.

This processor provides a single, stable user-facing interface (``populate_grid``).

User config example
-------------------
processors= {
    "populate_grid": {
        "reference_grid": "path/to/reference_grid.nc",
        "fill_value": None,
        "tolerance": 1
      },
    }

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from processor_tools import BaseProcessor
import xarray as xr
from eoio.processors.registry import register_processor
import numpy as np


@dataclass(frozen=True)
class PopulateGridConfig:
    """
    Parameters for the populate_grid processor.
    """

    reference_grid: Union[str | xr.Dataset | None]
    dims_to_populate: Optional[str | list[str] | None]
    fill_value: Optional[Union[None | float | int]] = None
    tolerance: Optional[Union[float | None]] = 1


@register_processor("populate_grid")
class PopulateGrid(BaseProcessor):
    """
    Fill dataset with fill value to match shape of a given reference viewing geometry grid.

    Processor parameters
    --------------------

    The following parameters can be provided in the `params` dict:

    :param reference_grid: str or xarray.Dataset
        Reference grid used to determine where values should be filled.
        This may be a filepath or an explicit Dataset. The data_vars should be the variables to fill along
        and the coordinates of these variables should match those in the dataset to be processed.
    :param dims_to_populate: str or List of str, optional
        Variables of reference grid to populate.
        If None, all variables in reference grid will be used.
    :param fill_value: float, int, or None, optional
        Value to use for filled locations when interpolation is not requested.
        If None, the data will be filled with nans.
    :param tolerance: float or None, optional
        Value to use to search for matching viewing angles.
        If None, infinite range is used, the same idea as 'nearest' interpolation.

    Notes
    -----
    - This processor is intended to run after reading.
    """

    _all_options = {
        "reference_grid": "str/xarray.Dataset or filepath: Reference grid used to determine where values should be filled or interpolated.",
        "dims_to_populate": "str/list/None: Variables of reference grid to populate.",
        "fill_value": "float/int/None: Value to use for filled locations when interpolation is not requested. If None, NaN is used.",
        "tolerance": "float/None: Value to use to in search window for angle matches between dataset and reference grid.",
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an add_lat_lon processor.

        :param params:
            Processor parameters (see class docstring for details)
        :param context:
            Processing context provided by eoio (reader info, metadata view, logger, etc.).
        """

        super().__init__(context=context)
        self.populate_grid_config = self._parse_params(params or {})

    def _parse_params(self, params: Dict[str, Any]) -> PopulateGridConfig:
        """
        Validate and normalise processor parameters.

        :param params:
            Raw params dict from the processor spec.
        :return:
            Parsed PopulateGridParams.
        :raises ValueError:
            If required parameters are missing or invalid.
        """

        # resolve "reference_grid" param
        reference_grid = params.get("reference_grid", None)

        # resolve "dims_to_populate" param
        dims_to_populate = params.get("dims_to_populate", None)

        # resolve "fill_value" param
        fill_value = params.get("fill_value", None)

        # resolve "tolerance" param
        tolerance = params.get("tolerance", 1)

        return PopulateGridConfig(
            reference_grid=reference_grid, dims_to_populate=dims_to_populate, fill_value=fill_value, tolerance=tolerance
        )

    def _format_dims_to_populate(
        self, ds: xr.Dataset, reference_grid: xr.Dataset, dims_to_populate: Union[str | list[str] | None]
    ) -> list:
        """
        Format the dims_To_populate to ensure they are in the correct format for populate_grid processing.

        :param dims_to_populate:
            Variables of reference grid to populate.
        :param reference_grid:
            Reference grid for populate_grid processor.
        :param ds:
            Dataset to be processed.
        """
        if type(dims_to_populate) is str:
            dims = [dims_to_populate]
        elif type(dims_to_populate) is list and all([type(d) is str for d in dims_to_populate]):
            dims = dims_to_populate
        elif dims_to_populate is None:
            dims = [str(v) for v in reference_grid.data_vars]
        else:
            raise ValueError(
                "populate_grid: dims_to_populate is not a valid input, must be a str or list of str, or None."
            )

        if not all([(d in reference_grid.data_vars) for d in dims]):
            raise ValueError("populate_grid: all dims_to_populate are not present in reference_grid")
        if not all([(d in ds.data_vars) for d in dims]):
            raise ValueError("populate_grid: all dims_to_populate are not present in dataset")

        return dims

    def _format_reference_grid(self, reference_grid: Union[str | xr.Dataset | None]) -> xr.Dataset:
        """
        Format the reference grid to ensure they are in the correct format for populate_grid processing.

        :param reference_grid:
            Reference grid for populate_grid processor.
        """
        if type(reference_grid) is xr.Dataset:
            ref_grid = reference_grid
        elif type(reference_grid) is str:
            ref_grid = xr.open_dataset(reference_grid)
        else:
            raise ValueError(
                "populate_grid: reference grid is not a valid reference, must be xarray Dataset or DataArray, or a filepath to such objects"
            )

        return ref_grid

    def regrid_series(self, ds: xr.Dataset, reference_grid: xr.Dataset, dims_to_populate: list) -> xr.Dataset:
        ref_vars = [reference_grid[var] for var in dims_to_populate]
        ds_vars = [ds[var] for var in dims_to_populate]
        n_ref = len(ref_vars[0])

        ds_dim = [d.dims for d in ds_vars][0][0]
        # Find where existing series belong in the reference grid
        series_map = {}
        tol = self.populate_grid_config.tolerance
        for i in range(ds.sizes[ds_dim]):
            ds_vals = [d.values[i] for d in ds_vars]

            matches = np.where(
                np.all(
                    np.vstack([(np.abs(ref_vars[i] - ds_vals[i]) < tol).values for i in range(len(ds_vals))]), axis=0
                )
            )[0]

            if len(matches):
                series_map[i] = matches[0]

        new_vars = {}

        for name, da in ds.data_vars.items():
            if ds_dim not in da.dims:
                # Variables without series dimension copied unchanged
                new_vars[name] = da
                continue

            dims = da.dims
            shape = list(da.shape)

            series_axis = dims.index(ds_dim)
            shape[series_axis] = n_ref

            # Create output array
            out = np.full(shape, self.populate_grid_config.fill_value, dtype=da.dtype)

            # Copy existing data to correct output positions
            for old_idx, new_idx in series_map.items():
                src: list[slice | int] = [slice(None)] * da.ndim
                dst: list[slice | int] = [slice(None)] * da.ndim

                src[series_axis] = old_idx
                dst[series_axis] = new_idx

                out[tuple(dst)] = da.values[tuple(src)]

            new_vars[name] = xr.DataArray(
                out,
                dims=dims,
                attrs=da.attrs,
            )

        new_ds = xr.Dataset(
            new_vars,
            coords={d: ds[d] if d != ds_dim else np.arange(n_ref) for d in ds.dims},
            attrs=ds.attrs,
        )

        # Replace variables with reference grid
        for ref_var in ref_vars:
            new_ds[ref_var.name].values = ref_var.values

        return new_ds

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Run populate_grid processor on the dataset.

        :param ds:
            Input dataset.
        :return:
            Output dataset with filled variables.
        """

        if not isinstance(ds, xr.Dataset):
            raise TypeError("populate_grid: input must be an xarray.Dataset.")

        # get reference grid
        reference_grid = self._format_reference_grid(self.populate_grid_config.reference_grid)
        # get dims_to_populate
        dims_to_populate = self._format_dims_to_populate(ds, reference_grid, self.populate_grid_config.dims_to_populate)

        ds = self.regrid_series(ds, reference_grid, dims_to_populate)

        # Record processing history
        ds = self._record_provenance(ds, dims_to_populate)

        return ds

    def _record_provenance(
        self,
        ds: xr.Dataset,
        dims_to_populate: list,
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
                "processor": "populate_grid",
                "dims_populated": dims_to_populate,
            }
        )
        ds.attrs["eoio:processing_steps"] = steps
        return ds


if __name__ == "__main__":
    pass
