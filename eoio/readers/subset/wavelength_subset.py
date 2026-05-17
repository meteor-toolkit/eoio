"""
Future notes:

        indices_wav never none, if data is empty [] the indices_way []. but not None
        ds["wavelength"].size returns scalar value
        np.arange() create evenly sample index array with step = 1
        np.arrange(3) -> [0, 1, 2]

        ds["wavelength"].size returns tuple(2, 7) # do a check it is a scalar raise NotImplemented
        np.arrange(2, 7) -> [2, 3, 4, 5, 6]

        Is ds['wavelength] always 1D? # to make sense it should be either already idx list or it should be number of the wavelenghts
Create for 2D
"""

from __future__ import annotations
from typing import Any, Dict

from eoio.readers.subset.base_subset import BaseSubsetResolver

__all__ = ["WavelengthSubsetResolver"]


# -----------------------------------------------------------------------------------
class WavelengthSubsetResolver(BaseSubsetResolver):
    """
    Resolves wavelength indices for subsetting raster data.

    :param data:
        Wavelength data (e.g., xarray DataArray).
        Data is expected to be an xarray-like object with .values, .size, and .dims.
    :param wavelength_params:
        Dictionary specifying wavelength selection criteria:
        * {'min': 200}
        * {'max': 600}
        * {'nearest': 450, 'tolerance': 20}
        * {'min': 200, 'max': 600}
    :returns WavelengthSubsetResolver:
        An instance ready to compute wavelength subsets.
    """

    def __init__(
        self,
        data: Any,
        variable_params: Dict,
    ):
        super().__init__(data, variable_params)


if __name__ == "__main__":
    pass
