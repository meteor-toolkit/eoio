"""Subset builder for EMIT. Returns a small dict describing requested subset."""

from __future__ import annotations
from typing import Dict, Optional, Sequence
import xarray as xr
from dataclasses import dataclass
import numpy as np

from eoio.readers.subset.roi_subset import ResolvedROISubset, ROISubsetResolver
from eoio.readers.subset.wavelength_subset import WavelengthSubsetResolver


@dataclass(frozen=True)
class EMITSubset:
    """
    Resolved subsetting info for EMIT subset reads.

    Attributes
    ----------
    series_indices
        Indices for series dimension
    wavelength_indices
        Indices for wavelength dimension
    """

    x_indices: Optional[np.ndarray] = None
    y_indices: Optional[np.ndarray] = None
    wavelength_indices: Optional[Sequence] = None
    roi: Optional[ResolvedROISubset] = None


def build_subset(ds: xr.Dataset, subset: Dict) -> "EMITSubset":
    """
    Build a simple subset descriptor consumed by image_io and angles modules.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset from which to extract subset information.
    subset : dict
        Subset dictionary specifying region of interest and CRS.

    Returns
    -------
    dict
        Minimal structure containing ROI information for downstream modules.

    Notes
    -----
    This intentionally returns a minimal structure: {roi, roi_crs, sample_path}
    """
    wavelength_indices = None
    roi_subset = None

    if "roi" in subset.keys() and subset["roi"] is not None:
        roi = subset["roi"]
        roi_crs = subset["roi_crs"]

        lat = ds["latitude"].values
        lon = ds["longitude"].values

        min_lat = float(lat.min())
        max_lat = float(lat.max())
        min_lon = float(lon.min())
        max_lon = float(lon.max())

        image_bounds = (min_lon, min_lat, max_lon, max_lat)

        roi_subset = ROISubsetResolver(roi, roi_crs, "EPSG:4326", image_bounds).run()

    if "wavelength" in subset.keys() and subset["wavelength"] is not None:
        wav_subset = WavelengthSubsetResolver(ds["wavelength"], subset["wavelength"]).run()
        wavelength_indices = wav_subset.variable_indices

    return EMITSubset(
        roi=roi_subset,
        wavelength_indices=wavelength_indices,
    )
