"""
eoio.readers.flex.subset
========================

Subset resolution for FLEX products.

Classes
-------
.. autosummary::
   :toctree: generated/

   FLEXSubset

Functions
---------
.. autosummary::
   :toctree: generated/

   build_subset
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional
import numpy as np
import xarray as xr

from eoio.readers.flex.grid import FlexGrid
from eoio.readers.subset.roi_subset import ResolvedROISubset, ROISubsetResolver
from eoio.readers.subset.wavelength_subset import WavelengthSubsetResolver

__all__ = ["FLEXSubset", "build_subset"]


@dataclass(frozen=True)
class FLEXSubset:
    """
    Resolved subsetting information for a FLEX read.

    :param x_indices: column indices to keep, or ``None`` for all.
    :param y_indices: row indices to keep, or ``None`` for all.
    :param wavelength_indices: indices to keep for each wavelength dimension, e.g.
        ``{"wavelength_olci": array([...])}``. The same wavelength criteria are applied to
        every spectral grid in the product, each on its own wavelengths.
    :param roi: resolved ROI (in the product CRS), or ``None``.
    """

    x_indices: Optional[np.ndarray] = None
    y_indices: Optional[np.ndarray] = None
    wavelength_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    roi: Optional[ResolvedROISubset] = None


def build_subset(grid: FlexGrid, wavelengths: Mapping[str, xr.DataArray], subset: Mapping) -> FLEXSubset:
    """
    Resolve a user subset request against the product grid and wavelength coordinates.

    :param grid: product grid.
    :param wavelengths: wavelength coordinate (1-D array-like with ``.size``) for each
        wavelength dimension.
    :param subset: subset request with optional ``roi``, ``roi_crs`` and ``wavelength`` keys.
    :returns: resolved subset.
    """
    roi_subset = None
    x_idx = y_idx = None

    if subset.get("roi") is not None:
        roi_subset = ROISubsetResolver(subset["roi"], subset["roi_crs"], grid.epsg, grid.bounds).run()
        xmin, ymin, xmax, ymax = roi_subset.clip_box  # type: ignore[misc]
        x, y = grid.x, grid.y
        # keep every pixel that overlaps the ROI bounding box
        x_idx = np.where((x + grid.dx / 2 >= xmin) & (x - grid.dx / 2 <= xmax))[0]
        y_idx = np.where((y + grid.dy / 2 >= ymin) & (y - grid.dy / 2 <= ymax))[0]

    wavelength_indices: Dict[str, np.ndarray] = {}
    if subset.get("wavelength") is not None:
        for dim, wvl in wavelengths.items():
            resolved = WavelengthSubsetResolver(wvl, subset["wavelength"]).run()
            wavelength_indices[dim] = np.asarray(resolved.variable_indices, dtype=int)

    return FLEXSubset(x_indices=x_idx, y_indices=y_idx, wavelength_indices=wavelength_indices, roi=roi_subset)


if __name__ == "__main__":
    pass
