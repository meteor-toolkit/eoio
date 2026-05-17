"""eoio.processors.base - base processor class"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import xarray as xr

__author__ = [
    "Mattea Goalen <mattea.goalen@npl.co.uk>",
]
__all__ = ["BaseProcessor"]


class BaseProcessor(ABC):
    """
    Base Processor

    :param process_params: definition of desired processing parameters, by default None
    :param subset_info: definition of subsetting parameters passed through to the reader object, by default None
    """

    name: str
    _default_process_params: Optional[Union[Dict[str, Any], list]]

    def __init__(
        self,
        ds: xr.Dataset,
        process_params: Optional[Dict[str, Any]] = None,
        subset_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise satellite processor

        (subset_info doesn't need to be checked as values are checked when used to read in the product)
        """
        self._all_options = None
        self.subset_info = subset_info
        self.process_params: Optional[Union[Dict[str, Any], Union[list, dict]]] = {}

        if isinstance(process_params, dict):
            process_specific_params = [v for k, v in process_params.items() if k == self.name][0]
        else:
            process_specific_params = None

        if isinstance(process_specific_params, dict):
            self._set_keys(process_specific_params)
        elif process_specific_params is True:
            self.process_params = self._default_process_params
        elif process_specific_params is False or process_specific_params is None:
            raise ValueError(
                f"'{process_specific_params}' is not a valid processing parameter, please selected from {self.all_options}"
            )
        else:
            self.process_params = process_specific_params

        self.ds = ds

    def _set_keys(self, process_params: Dict[str, Any]) -> None:
        """
        Set class attributes from input keys and default values

        :param process_params: definition of desired processing parameters for data
        """
        if not isinstance(self._default_process_params, dict):
            raise ValueError(
                f"""Processor selected: '{self.name}' does not accept dictionary processing input.
        Please choose from '{self._default_process_params}'"""
            )

        for k, v in process_params.items():
            if k not in self._default_process_params.keys():
                raise ValueError("'{}' is not a valid reading parameter".format(k))

        self.process_params = {
            k: (v if k not in process_params.keys() else process_params[k])
            for (k, v) in self._default_process_params.items()
        }

    @property
    @abstractmethod
    def all_options(self):
        """
        Return dictionary or list of available processing parameter
        """

    @abstractmethod
    def process_dataset(self) -> xr.Dataset:
        """
        Process data and metadata and return an xarray.Dataset
        """


if __name__ == "__main__":
    pass
