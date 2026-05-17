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

from eoio.readers.subset.wavelength_subset import WavelengthSubsetResolver

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

    # series_indices: Optional[np.ndarray]


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

    return GENERIC_NETCDFSubset(
        # series_indices=indices,
        wavelength_indices=wavelength_indices,
    )


if __name__ == "__main__":
    pass
