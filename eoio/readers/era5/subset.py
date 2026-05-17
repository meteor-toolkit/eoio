"""Subset builder for EMIT. Returns a small dict describing requested subset."""

from __future__ import annotations
from typing import Any, Dict
from eoio.readers.subset.roi_subset import ResolvedROISubset, ROISubsetResolver
import xarray as xr


def build_subset(ds: xr.Dataset, subset: Dict) -> Dict[str, Any]:
    """
    Build a simple subset descriptor consumed by image_io and angles modules.
    This intentionally returns a minimal structure: {roi, roi_crs, sample_path}
    """

    if subset["roi"] is not None:
        roi = subset["roi"]
        roi_crs = subset["roi_crs"]

        lat = ds["latitude"].values
        lon = ds["longitude"].values

        min_lat = float(lat.min())
        max_lat = float(lat.max())
        min_lon = float(lon.min())
        max_lon = float(lon.max())

        image_bounds = (min_lon, min_lat, max_lon, max_lat)

        roi_subset: ResolvedROISubset = ROISubsetResolver(roi, roi_crs, "EPSG:4326", image_bounds).run()
        return {"roi": roi_subset}
    return {}
