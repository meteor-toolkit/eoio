"""
eoio.processors.add_lat_lon.processor
-------------------------------

Top-level latitude/longitude processor.

This processor provides a single, stable user-facing interface (``add_lat_lon``).

User config example
-------------------
processors= {
    "add_lat_lon": {
        "geometry_id": ["10m", "60m"],
      },
    }

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from processor_tools import BaseProcessor
import xarray as xr
from eoio.processors.registry import register_processor
from eoio.deps import lazy_pyproj
import numpy as np


@dataclass(frozen=True)
class AddLatLonConfig:
    """
    Parameters for the add_lat_lon processor.
    """

    geometry_id: Optional[Sequence[str]] = None  # e.g. ["10m", "60m"]
    on_missing: str = "error"  # "error" | "skip"


@register_processor("add_lat_lon")
class AddLatLon(BaseProcessor):
    """
    Add latitude and longitude coordinates to a dataset.

    Processor parameters
    --------------------

    The following parameters can be provided in the `params` dict:

    :param geometry_id:
        Optional, List of geometry IDs to add (e.g. ``["10m", "60m"]``), if omitted lat/lon will be added for all grid resolutions found in the dataset.
    :param on_missing:
        Behaviour if required metadata for conversion is missing.
        Supported values are ``"error"`` (default, if omitted) or ``"skip"``.

    Notes
    -----
    - This processor is intended to run after reading.
    """

    _all_options = {
        "geometry_id": "Optional, List of geometry IDs to add (e.g. ['10m', '60m']), if omitted lat/lon will be added for all grid resolutions found in the dataset.",
        "on_missing": "Behaviour if required metadata for conversion is missing. Supported values are 'error' (default, if omitted) or 'skip'.",
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
        self.add_lat_lon_config = self._parse_params(params or {})

    def _parse_params(self, params: Dict[str, Any]) -> AddLatLonConfig:
        """
        Validate and normalise processor parameters.

        :param params:
            Raw params dict from the processor spec.
        :return:
            Parsed AddLatLonParams.
        :raises ValueError:
            If required parameters are missing or invalid.
        """

        # resolve "geometry_id" param
        geometry_id = params.get("geometry_id", None)

        # resolve "on_missing" param
        on_missing = str(params.get("on_missing", "error")).lower()
        if on_missing not in {"error", "skip"}:
            raise ValueError("interpolate: 'on_missing' must be 'error' or 'skip'.")

        return AddLatLonConfig(
            geometry_id=geometry_id,
            on_missing=on_missing,
        )

    def _format_geometry_id(self, ds: xr.Dataset, geometry_id: Optional[Sequence]) -> Sequence:
        """
        Format the geometry ids to ensure they are in the correct format for lat/lon processing.

        :param ds:
            Input dataset (used for context, e.g. to check available coordinates).
        :param geometry_id:
            List of geometry ids to process lat/lon for  (e.g. ``["10m", "60m"]``).
        """
        # available_geoms = list(set(ds.geometry_id))
        available_geoms = list(set([str(x).split("_")[-1] for x in ds.coords if "x_" in str(x) or "y_" in str(x)]))
        # Check all coords exist in the dataset
        if geometry_id is not None:
            for geom in geometry_id:
                if geom not in available_geoms:
                    raise ValueError(f"AddLatLon: geometry_id '{geom}' not found in dataset.")
        else:
            geometry_id = available_geoms

        return geometry_id

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Run add lat/lon on the dataset.

        :param ds:
            Input dataset.
        :return:
            Output dataset with lat/lon coords.
        """

        if not isinstance(ds, xr.Dataset):
            raise TypeError("add_lat_lon: input must be an xarray.Dataset.")

        # get geometry ids
        geometry_id = self._format_geometry_id(ds, self.add_lat_lon_config.geometry_id)

        # instantiate transformer
        pyproj = lazy_pyproj()
        crs_src = ds.rio.crs
        crs_dst = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)

        # add lat/lon as coords
        for geom in geometry_id:
            x, y = np.meshgrid(ds[f"x_{geom}"].values, ds[f"y_{geom}"].values)
            lons, lats = transformer.transform(x, y)

            lat_lon_dict = {
                f"latitude_{geom}": (
                    [f"y_{geom}", f"x_{geom}"],
                    lats,
                    {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north"},
                ),
                f"longitude_{geom}": (
                    [f"y_{geom}", f"x_{geom}"],
                    lons,
                    {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east"},
                ),
            }

            ds = ds.assign_coords(coords=lat_lon_dict)

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
                "processor": "add_lat_lon",
                "geometry_ids": self._format_geometry_id(ds, self.add_lat_lon_config.geometry_id),
            }
        )
        ds.attrs["eoio:processing_steps"] = steps
        return ds


if __name__ == "__main__":
    pass
