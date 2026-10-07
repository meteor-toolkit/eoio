"""
eoio.readers.generic_netcdf.subset
=============================

Resolves user-defined subsets for GENERIC_NETCDF images.

Deliberately narrow: an arbitrary netCDF cannot be assumed to name its time
coordinate ``datetime`` or its spectral coordinate ``wavelength``, nor to carry
a CRS, so this module resolves only what can be checked against the file itself.
Readers that *do* know what their coordinates mean own their own subset module
(see ``eoio.readers.ecmwf.subset``). ``NetCDFReader`` correspondingly declares no
``default_subset``, so it advertises and accepts no subsetting parameters.

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
    wavelength_indices
        Indices for wavelength dimension
    """

    wavelength_indices: Optional[np.ndarray]


def build_subset(ds: xr.Dataset, subset: dict) -> GENERIC_NETCDFSubset:
    """
    Build a GENERIC_NETCDFSubset from a dataset and subset definition.

    :param ds: The input dataset.
    :param subset: Subsetting definition.
    :returns: The resolved subsetting information.
    """

    wavelength_indices = None
    if "wavelength" in ds.dims and subset.get("wavelength") is not None:
        wav_subset = WavelengthSubsetResolver(ds["wavelength"], subset["wavelength"]).run()
        wavelength_indices = np.array(wav_subset.variable_indices) if len(wav_subset.variable_indices) > 0 else None

    return GENERIC_NETCDFSubset(wavelength_indices=wavelength_indices)


if __name__ == "__main__":
    pass
