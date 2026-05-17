"""eoio.readers.base - base classes for eoio product readers

Design goals
------------
- Keep import-time dependencies light (optional backends are imported lazily).
- Provide a small, explicit reader contract suitable for thin orchestrator readers.
- Use validated configuration dictionaries for subsetting and read-time options.

Reader contract
---------------
Concrete readers must implement:
- `open_dataset() -> xarray.Dataset`
- `extract_metadata() -> tuple[dict, dict]` (if metadata is supported)
- `get_extension() -> str`

Typical usage
-------------
    reader = SomeReader(path, vars_sel={...}, subset={...}, read_params={...})
    ds = reader.open()

`open()` is the public entrypoint and handles optional metadata attachment.
"""

from __future__ import annotations
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List, TYPE_CHECKING
import os

import xarray as xr

if TYPE_CHECKING:
    import rasterio  # noqa: F401
    import rioxarray  # noqa: F401


# -------------------------
# Configuration
# -------------------------


MetadataLevel = Union[None, bool, str]


@dataclass(frozen=True)
class ReaderConfig:
    """Normalised, validated configuration for a reader instance."""

    vars_sel: Dict[str, Any]
    subset: Any  # Optional[Dict] or Optional[ResolvedROISubset] depending on reader
    read_params: Dict[str, Any]


# -------------------------
# Base reader
# -------------------------


class BaseReader(ABC):
    """Base class for EOIO readers.

    This class is intentionally small. It owns:
    - path validation
    - config merge/validation
    - a single public entrypoint: `open()`

    Concrete readers implement `open_dataset()` and optionally `extract_metadata()`.
    """

    # Default values for user inputs, for all available parameters
    # Concrete readers should override these

    default_vars_sel = {"meas": "all", "mask": None, "aux": None}

    default_subset: Dict[str, Any] = {}
    all_subset: Dict[str, Any] = {}

    default_read_params: Dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": False,
    }
    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": "True/False",
    }

    # All available options for user inputs, for all available parameters
    # Concrete readers should override these

    meas_def: Dict[str, List[str]] = {
        "all": [],
    }

    aux_def: Dict[str, List[str]] = {"all": [], "basic": []}

    mask_def: Dict[str, List[str]] = {"all": []}

    uncertainty_vars: List[str] = []

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not os.path.exists(path):
            raise ValueError(f"'{path}' cannot be found")
        self.path = Path(path)

        self.config = ReaderConfig(
            vars_sel=self._merge_and_validate(vars_sel, self.default_vars_sel, kind="variable selection"),
            subset=self._merge_and_validate(subset, self.default_subset, kind="subsetting"),
            read_params=self._merge_and_validate(read_params, self.default_read_params, kind="reading"),
        )

        self.resolved_config = ReaderConfig(
            vars_sel=self.resolved_vars_sel(),
            subset=self.resolve_subset(self.config.subset),
            read_params=self.config.read_params,
        )

        self._all_options: Optional[dict] = None

    # ---- public API ----
    def open(self) -> xr.Dataset:
        """Public entrypoint: open the product and return an xarray.Dataset."""
        ds = self.open_dataset()

        return ds

    @abstractmethod
    def open_dataset(self) -> xr.Dataset:
        """Read the product and return an xarray.Dataset."""

    def resolved_vars_sel(self) -> dict[str, List[str]]:
        """List selected variables to read based on provider user values or defaults."""

        resolved_vars_sel = {
            "meas": self.list_selected_meas(),
            "aux": self.list_selected_aux(),
            "mask": self.list_selected_mask(),
        }

        return resolved_vars_sel

    def list_selected_meas(self) -> List[str]:
        """List measurement variables to read based on provider user values or defaults."""
        if self.config.vars_sel["meas"] is None:
            meas_vars = []

        elif isinstance(self.config.vars_sel["meas"], str):
            if self.config.vars_sel["meas"] in self.meas_def.keys():
                meas_vars = self.meas_def[self.config.vars_sel["meas"]]

            elif self.config.vars_sel["meas"] in self.meas_def["all"]:
                meas_vars = [self.config.vars_sel["meas"]]

            else:
                raise ValueError("Unknown meas requested: " + str(self.config.vars_sel["meas"]))

        else:
            meas_vars = self.config.vars_sel["meas"]

        return meas_vars

    def list_selected_aux(self) -> List[str]:
        """List auxiliary variables to read based on provider user values or defaults."""
        if self.config.vars_sel["aux"] is None:
            aux = []

        elif isinstance(self.config.vars_sel["aux"], str):
            if self.config.vars_sel["aux"] in self.aux_def.keys():
                aux = self.aux_def[self.config.vars_sel["aux"]]
            elif self.config.vars_sel["aux"] in self.aux_def["all"]:
                aux = [self.config.vars_sel["aux"]]

            else:
                raise ValueError("Unknown aux requested: " + str(self.config.vars_sel["aux"]))

        else:
            aux = self.config.vars_sel["aux"]

        return aux

    def list_selected_mask(self) -> List[str]:
        """List mask variables to read based on provider user values or defaults."""
        if self.config.vars_sel["mask"] is None:
            mask = []

        elif isinstance(self.config.vars_sel["mask"], str):
            if self.config.vars_sel["mask"].lower() in self.mask_def.keys():
                mask = self.mask_def[self.config.vars_sel["mask"].lower()]

            elif self.config.vars_sel["mask"] in self.mask_def["all"]:
                mask = [self.config.vars_sel["mask"]]

            else:
                raise ValueError("Unknown mask requested: " + str(self.config.vars_sel["mask"]))

        else:
            mask = self.config.vars_sel["mask"]

        return mask

    def resolve_subset(self, subset: Optional[Dict[str, Any]]) -> Any:
        """
        Resolve the subset variables to read based on provider user values or defaults.

        Subclasses should implement their own version of this method.
        """

        return subset

    def list_uncertainty_vars(self) -> List[str]:
        """List uncertainty variables to read based on defaults."""
        return self.uncertainty_vars

    def list_include_vars(self) -> List[str]:
        meas_vars = self.list_selected_meas()
        aux_vars = self.list_selected_aux()
        mask = self.list_selected_mask()
        include_vars = meas_vars + aux_vars + mask
        if self.config.read_params["include_uncertainties"]:
            include_vars += self.list_uncertainty_vars()

        return include_vars

    # ---- config helpers ----

    @staticmethod
    def _merge_and_validate(
        provided: Optional[Dict[str, Any]],
        defaults: Dict[str, Any],
        *,
        kind: str,
    ) -> Dict[str, Any]:
        """
        Merge provided dict onto defaults and validate keys.
        This is intentionally strict: unknown keys raise ValueError.

        :param provided: Provided keys
        :param defaults: Default keys
        :param kind: Provided dictionary name (only used for clarifying errors)
        :return: Merged dictionary
        """
        provided = provided or {}
        unknown = [k for k in provided.keys() if k not in defaults]
        if unknown:
            raise ValueError(f"Unknown {kind} parameter(s): {unknown}. Allowed: {sorted(defaults.keys())}")

        merged = dict(defaults)
        merged.update(provided)
        return merged

    @staticmethod
    @abstractmethod
    def get_extension() -> str:
        pass

    @property
    def all_options(self) -> Dict[str, Any]:
        """
        Return dictionary of available subsetting parameters.

        :returns: Dictionary of available subsetting parameters.
        """

        if self._all_options is None:
            self._all_options = {}
            self._all_options["subset"] = self.all_subset.copy()
            self._all_options["read_params"] = self.all_read_params.copy()
            self._all_options["vars_sel"] = {
                "aux": list(self.aux_def.keys()) + self.aux_def.get("all", []),
                "meas": list(self.meas_def.keys()) + self.meas_def.get("all", []),
                "mask": list(self.mask_def.keys()) + self.mask_def.get("all", []),
            }
        return self._all_options


class BaseRasterReader(BaseReader):
    """
    Base Reader class for raster imagery
    """

    # Default values for user inputs, for subset parameters

    default_subset = {
        "roi": None,
        "roi_crs": 4326,
        "angle": None,
        "wavelength": None,
    }
    all_subset = {
        "roi": "TBD",  # TODO list accepted formats
        "roi_crs": "str/int",
        "angle": "min/max/nearest/tolerance",
        "wavelength": "min/max/nearest/tolerance",
    }
    # inheriting default_read_params from BaseReader
    # inheriting all_read_params from BaseReader

    @abstractmethod
    def get_extension(self):
        pass


class BaseInSituReader(BaseReader):
    """
    Base Reader class for In Situ data
    """

    # Default values for user inputs, for subset parameters
    # Concrete readers should override these

    default_subset = {"wavelength": None, "datetime": None, "angle": None}

    all_subset = {
        "wavelength": "min/max/nearest/tolerance",
        "datetime": "min/max/nearest/tolerance_days/tolerance_hours/tolerance_minutes",
        "time_of_day_utc": "min/max/nearest/tolerance_hours/tolerance_minutes",
        "angle": "min/max/nearest/tolerance",
    }

    # inheriting default_read_params from BaseReader
    # inheriting all_read_params from BaseReader

    @staticmethod
    @abstractmethod
    def get_extension():
        pass


if __name__ == "__main__":
    pass
