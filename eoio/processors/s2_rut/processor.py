"""
eoio.processors.s2_rut.processor
-------------------------------

Top-level S2 RUT processor.

This processor provides a single, stable user-facing interface (``s2_rut``).

User config example
-------------------
processors = {
    "s2_rut": {
      "data_vars": ["B02"],
      },
    }
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from processor_tools import BaseProcessor
import xarray as xr
from eoio.processors.registry import register_processor


@dataclass(frozen=True)
class S2RUTConfig:
    """
    Parameters for the S2 RUT processor.
    """

    data_vars: Optional[Sequence[str]] = None
    group_unc: Optional[bool] = True
    subset_unc: Optional[Sequence[str]] = None
    on_missing: str = "error"  # "error" | "skip"


@register_processor("s2_rut")
class S2Rut(BaseProcessor):
    """
    Run Sentinel-2 Radiometric Uncertainty Tool (RUT) on a dataset.

    Processor parameters
    --------------------

    The following parameters can be provided in the `params` dict:

    :param data_vars:
        List of variable names to run RUT for. If omitted, run for all measurement variables (bands) in the dataset.
    :param group_unc:
        Whether to group uncertainty contributors in to components - random, systematic. Default is True.
    :param subset_unc:
        List of uncertainty variables to include in the output dataset. If omitted, include all uncertainty variables.
    :param on_missing:
        Behaviour if required metadata for conversion is missing.
        Supported values are ``"error"`` (default, if omitted) or ``"skip"``.

    Notes
    -----
    - This processor is intended to run after reading.
    """

    _all_options = {
        "data_vars": "List of variable names to run RUT for. If omitted, run for all measurement variables (bands) in the dataset.",
        "group_unc": "Whether to group uncertainty contributors in to components - random, systematic. Default is True.",
        "subset_unc": "List of uncertainty variables to include in the output dataset. If omitted, include all uncertainty variables.",
        "on_missing": "Behaviour if required metadata for conversion is missing. Supported values are 'error' (default, if omitted) or 'skip'.",
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an S2 RUT processor.

        :param params:
            Processor parameters (see class docstring for details)
        :param context:
            Processing context provided by eoio (reader info, metadata view, logger, etc.).
        """

        super().__init__(context=context)
        self.rut_config = self._parse_params(params or {})

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Run uncertainty tool (S2 RUT) on the dataset.

        :param ds:
            Input dataset.
        :return:
            Output obsarray dataset with uncertainty variables.
        """

        if not isinstance(ds, xr.Dataset):
            raise TypeError("s2_rut: input must be an xarray.Dataset.")

        # run RUT
        from s2_rut_python.interface import S2RUTTool  # lazy import — optional dependency

        rut = S2RUTTool()
        ds = rut.run(
            ds,
            data_vars=self.rut_config.data_vars,
            group_unc=self.rut_config.group_unc,
        )

        # Record processing history
        ds = self._record_provenance(ds)

        return ds

    def _record_provenance(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Record a minimal provenance entry at processor level.

        :param ds:
            Output dataset.
        :return:
            Dataset (same object, attrs updated).
        """

        steps: list = ds.attrs.get("eoio:processing_steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]

        steps.append(
            {
                "processor": "s2_rut",
                "rut_config": self.rut_config,
            }
        )
        ds.attrs["eoio:processing_steps"] = steps
        return ds

    def _parse_params(self, params: Dict[str, Any]) -> S2RUTConfig:
        """
        Parse and validate processor parameters.

        :param params:
            Processor parameters (see class docstring for details)
        :return:
            Parsed and validated processor configuration.
        """

        # resolve "data_vars" param
        data_vars = params.get("data_vars", None)

        # resolve "group_unc" param
        group_unc = bool(params.get("group_unc", True))

        # resolve "subset_unc" param
        subset_unc = params.get("subset_unc", None)

        # resolve "on_missing" param
        on_missing = str(params.get("on_missing", "error")).lower()
        if on_missing not in {"error", "skip"}:
            raise ValueError("s2_rut: 'on_missing' must be 'error' or 'skip'.")

        return S2RUTConfig(
            data_vars=data_vars,
            group_unc=group_unc,
            subset_unc=subset_unc,
            on_missing=on_missing,
        )


if __name__ == "__main__":
    pass
