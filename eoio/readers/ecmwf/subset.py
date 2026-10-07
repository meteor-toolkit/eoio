"""
eoio.readers.ecmwf.subset
=========================

Resolves user-defined subsets for ECMWF (ERA5 / CAMS) datasets.

ROI and datetime subsetting live here rather than in
``eoio.readers.generic_netcdf.subset`` because they depend on knowing what the
product's coordinates mean: ``ECMWFReader.open_dataset`` renames ``valid_time``
to ``datetime`` and guarantees a CRS before calling in, and ``ECMWFReader``
declares ``roi``/``roi_crs``/``datetime`` in its ``default_subset`` so callers
can actually reach them. The generic netCDF reader can assume neither, so it
keeps a deliberately smaller subset vocabulary.

Classes
-------
.. autosummary::
   :toctree: generated/

   ECMWFSubset

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

from eoio.deps import lazy_rioxarray
from eoio.readers.subset.datetime_subset import DatetimeSubsetResolver
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset


@dataclass(frozen=True)
class ECMWFSubset:
    """
    Resolved subsetting info for ECMWF reads.

    Attributes
    ----------
    datetime_indices
        Indices along the ``datetime`` dimension, or ``None`` for no constraint
    roi_subset
        Resolved ROI in the image CRS, or ``None`` for no constraint
    """

    datetime_indices: Optional[np.ndarray]
    roi_subset: Optional[ResolvedROISubset]


def build_subset(ds: xr.Dataset, subset: dict) -> ECMWFSubset:
    """
    Build an ECMWFSubset from a dataset and subset definition.

    :param ds: The input dataset, with its time coordinate already named
        ``datetime`` and a CRS written.
    :param subset: Subsetting definition.
    :returns: The resolved subsetting information.
    """

    # --- Datetime constraint: filter along the time dimension ---
    datetime_indices = None
    if "datetime" in ds.keys() and subset.get("datetime") is not None:
        datetime_subset = DatetimeSubsetResolver(ds["datetime"], subset["datetime"]).run()
        datetime_indices = datetime_subset.variable_indices

    # --- ROI constraint ---
    roi_subset = None
    roi = subset.get("roi")
    if roi is not None:
        roi_crs = subset.get("roi_crs")
        try:
            # rioxarray (not rasterio) is what registers the .rio accessor used below
            lazy_rioxarray()
            image_crs = str(ds.rio.crs)
        except Exception:
            raise ValueError("Cannot resolve ROI subset: raster CRS not available.")

        roi_subset = ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    return ECMWFSubset(datetime_indices=datetime_indices, roi_subset=roi_subset)
