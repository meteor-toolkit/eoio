"""
eoio.readers.generic_netcdf.subset
=============================

Resolves user-defined subsets for GENERIC_NETCDF images.

Classes
-------
.. autosummary::
   :toctree: generated/

   GENERIC_NETCDFSubsetError
   GENERIC_NETCDFSubset

Functions
---------
.. autosummary::
   :toctree: generated/

   build_subset
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import xarray as xr
from eoio.deps import lazy_rasterio

from eoio.readers.subset.wavelength_subset import WavelengthSubsetResolver
from eoio.readers.subset.datetime_subset import DatetimeSubsetResolver
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset

# This module is intentionally “mostly light”.
# Heavy geo dependencies are imported only when ROI/subsetting is actually used.


class GENERIC_NETCDFSubsetError(ValueError):
    """
    Raised when a GENERIC_NETCDF subset definition error is invalid.
    """


@dataclass(frozen=True)
class GENERIC_NETCDFSubset:
    """
    Resolved subsetting info for GENERIC_NETCDF raster reads.

    Attributes
    ----------
    series_indices
        Indices for series dimension
    wavelength_indices
        Indices for wavelength dimension
    """

    wavelength_indices: Optional[np.ndarray]
    datetime_indices: Optional[np.ndarray]
    roi_subset: Optional[ResolvedROISubset]


def build_subset(ds: xr.Dataset, subset: dict) -> GENERIC_NETCDFSubset:
    """
    Build a GENERIC_NETCDFSubset from a dataset and subset definition.

    :param ds: The input dataset.
    :param subset: Subsetting definition.
    :returns: The resolved subsetting information.
    """

    wavelength_indices = None
    if "wavelength" in subset.keys() and subset["wavelength"] is not None:
        wav_subset = WavelengthSubsetResolver(ds["wavelength"], subset["wavelength"]).run()
        wavelength_indices = np.array(wav_subset.variable_indices) if len(wav_subset.variable_indices) > 0 else None


    # --- Datetime constraint: filter series by datetime range ---
    datetime_indices = None
    if "datetime" in ds.keys() and "datetime" in subset and subset["datetime"] is not None:
        # we might need this later for reading the netcdf version of the files
        datetime_subset = DatetimeSubsetResolver(ds["datetime"], subset["datetime"]).run()
        datetime_indices = datetime_subset.variable_indices

    # ROI
    roi_subset=None
    if "datetime" in ds.keys() and subset["roi"] is not None:
        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        try:
            rio = lazy_rasterio()
            image_crs = str(ds.rio.crs)

        except Exception:
            raise ValueError("Cannot resolve ROI subset: raster CRS not available.")
        
        roi_subset = ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    return GENERIC_NETCDFSubset(
        wavelength_indices=wavelength_indices,
        datetime_indices=datetime_indices,
        roi_subset=roi_subset
    )


if __name__ == "__main__":
    pass
