"""
eoio.processors.resample_wavelength.processor
---------------------------------------------

Top-level wavelength resampling processor.

Resamples spectral variables, each on its own wavelength dimension (``wavelength_olci``,
``wavelength_floris``, ... as produced by the FLEX reader), onto one common wavelength grid so
they can be combined or compared.

User config example
-------------------
processors= {
    "resample_wavelength": {
        "target_wavelength": {"min": 500, "max": 780, "step": 1},
        "data_vars": ["floris_apparent_reflectance", "olci_apparent_reflectance"],
        "method": "linear",
      },
    }

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from processor_tools import BaseProcessor
import numpy as np
import xarray as xr
from eoio.processors.registry import register_processor

#: prefix identifying a source wavelength dimension
WAVELENGTH_DIM_PREFIX = "wavelength_"
_SUFFIX = "_resampled"
_UNC_SUFFIX = "_uncertainty"


@dataclass(frozen=True)
class ResampleWavelengthConfig:
    """
    Parameters for the resample_wavelength processor.
    """

    target_wavelength: np.ndarray
    data_vars: Optional[Sequence[str]] = None
    method: str = "linear"
    dim_name: str = "wavelength"
    inplace: bool = False
    drop_original: bool = False
    on_missing: str = "error"  # "error" | "skip"


@register_processor("resample_wavelength")
class ResampleWavelength(BaseProcessor):
    """
    Resample spectral variables onto a common wavelength grid.

    Each selected variable is interpolated along its own ``wavelength_*`` dimension onto the target
    wavelengths, giving variables that share one ``wavelength`` dimension. Target wavelengths
    outside the range of a variable's source grid are set to NaN (no extrapolation).

    Obsarray uncertainty components (``<var>_uncertainty``) of resampled variables are resampled
    with them and re-registered against the resampled variable, keeping their error-correlation
    definition with the wavelength dimension renamed.

    Processor parameters
    --------------------

    The following parameters can be provided in the `params` dict:

    :param target_wavelength:
        Required. Target wavelengths in nm, either a list/array, or a dict ``{"min", "max", "step"}``
        (``max`` inclusive).
    :param data_vars:
        Optional list of variables to resample. If omitted, all variables with a ``wavelength_*``
        dimension are resampled.
    :param method:
        Interpolation method (``"linear"``, ``"nearest"``, ``"cubic"``, ...), as for ``xarray.interp``.
        Default ``"linear"``.
    :param dim_name:
        Name of the new common wavelength dimension. Default ``"wavelength"``.
    :param inplace:
        If False (default) resampled variables are added with the suffix ``_resampled``
        (e.g. ``floris_apparent_reflectance_resampled``). If True they replace the originals.
    :param drop_original:
        If True, drop the originals (and their now-unneeded wavelength coordinates) after
        resampling. No-op when ``inplace`` is True. Default False.
    :param on_missing:
        Behaviour if a requested variable is missing or has no wavelength dimension:
        ``"error"`` (default) or ``"skip"``.

    Notes
    -----
    - This processor is intended to run after reading.
    """

    _all_options = {
        "target_wavelength": "List/array of target wavelengths in nm, or dict {'min', 'max', 'step'} (max inclusive).",
        "data_vars": "List of variable names to resample. If omitted, all variables with a 'wavelength_*' dimension are resampled.",
        "method": "Interpolation method (e.g. 'linear', 'nearest', 'cubic'). Default 'linear'.",
        "dim_name": "Name of the new common wavelength dimension. Default 'wavelength'.",
        "inplace": "Bool, if False (default) resampled variables are added with the suffix '_resampled'. If True they replace the originals.",
        "drop_original": "Bool, if True drop the originals after resampling (no-op when inplace=True). Default False.",
        "on_missing": "Behaviour if a requested variable is missing or has no wavelength dimension. Supported values are 'error' (default, if omitted) or 'skip'.",
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a resample_wavelength processor.

        :param params:
            Processor parameters (see class docstring for details)
        :param context:
            Processing context provided by eoio (reader info, metadata view, logger, etc.).
        """

        super().__init__(context=context)
        self.config = self._parse_params(params or {})

    def _parse_params(self, params: Dict[str, Any]) -> ResampleWavelengthConfig:
        """
        Validate and normalise processor parameters.

        :param params:
            Raw params dict from the processor spec.
        :return:
            Parsed ResampleWavelengthConfig.
        :raises ValueError:
            If required parameters are missing or invalid.
        """
        if "target_wavelength" not in params:
            raise ValueError("resample_wavelength: missing required param 'target_wavelength'.")
        target = params["target_wavelength"]
        if isinstance(target, dict):
            try:
                lo, hi, step = float(target["min"]), float(target["max"]), float(target["step"])
            except KeyError as e:
                raise ValueError(f"resample_wavelength: 'target_wavelength' dict needs min, max and step ({e}).")
            if step <= 0 or hi < lo:
                raise ValueError("resample_wavelength: 'target_wavelength' needs step > 0 and max >= min.")
            target = lo + step * np.arange(int(np.floor((hi - lo) / step + 1e-9)) + 1)
        target = np.asarray(target, dtype=float)
        if target.ndim != 1 or target.size == 0:
            raise ValueError("resample_wavelength: 'target_wavelength' must be a non-empty 1-D sequence.")

        on_missing = str(params.get("on_missing", "error")).lower()
        if on_missing not in {"error", "skip"}:
            raise ValueError("resample_wavelength: 'on_missing' must be 'error' or 'skip'.")

        return ResampleWavelengthConfig(
            target_wavelength=target,
            data_vars=params.get("data_vars"),
            method=str(params.get("method", "linear")).lower(),
            dim_name=str(params.get("dim_name", "wavelength")),
            inplace=bool(params.get("inplace", False)),
            drop_original=bool(params.get("drop_original", False)),
            on_missing=on_missing,
        )

    @staticmethod
    def _source_dim(da: xr.DataArray) -> Optional[str]:
        dims = [str(d) for d in da.dims if str(d).startswith(WAVELENGTH_DIM_PREFIX)]
        return dims[0] if dims else None

    def _select_vars(self, ds: xr.Dataset) -> List[str]:
        """Names of the base variables to resample (uncertainty components follow their base)."""
        requested = self.config.data_vars
        if requested is None:
            return [
                str(v) for v, da in ds.data_vars.items() if self._source_dim(da) and not str(v).endswith(_UNC_SUFFIX)
            ]
        selected = []
        for v in requested:
            if v not in ds.data_vars or self._source_dim(ds[v]) is None:
                if self.config.on_missing == "error":
                    raise ValueError(f"resample_wavelength: variable '{v}' not found or has no wavelength dimension.")
                continue
            selected.append(v)
        return selected

    def _resample(self, da: xr.DataArray, src_dim: str) -> xr.DataArray:
        cfg = self.config
        out = da.interp({src_dim: cfg.target_wavelength}, method=cfg.method)  # type: ignore[arg-type]
        out = out.rename({src_dim: cfg.dim_name})
        out = out.transpose(cfg.dim_name, ...)
        if da.dtype.kind == "f":
            out = out.astype(da.dtype)  # interp promotes to float64
        out.attrs = dict(da.attrs)
        return out

    @staticmethod
    def _err_corr_from_attrs(attrs: Dict[str, Any], old_dim: str, new_dim: str) -> List[Dict[str, Any]]:
        """Rebuild an obsarray ``err_corr`` list from encoded attrs, renaming the wavelength dim."""
        out = []
        i = 1
        while f"err_corr_{i}_dim" in attrs:
            dims = attrs[f"err_corr_{i}_dim"]
            dims = [dims] if isinstance(dims, str) else list(dims)
            out.append(
                {
                    "dim": [new_dim if d == old_dim else d for d in dims],
                    "form": attrs[f"err_corr_{i}_form"],
                    "params": list(attrs.get(f"err_corr_{i}_params", [])),
                    "units": list(attrs.get(f"err_corr_{i}_units", [])),
                }
            )
            i += 1
        return out

    @staticmethod
    def _strip_unc_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in attrs.items() if k != "unc_comps" and not k.startswith("err_corr_")}

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Run the processor on the dataset.

        :param ds:
            Input dataset.
        :return:
            Dataset with spectral variables resampled onto the common wavelength grid.
        """
        if not isinstance(ds, xr.Dataset):
            raise TypeError("resample_wavelength: input must be an xarray.Dataset.")

        cfg = self.config
        names = self._select_vars(ds)
        suffix = "" if cfg.inplace else _SUFFIX
        if cfg.dim_name in ds.dims and not cfg.inplace:
            raise ValueError(f"resample_wavelength: dimension '{cfg.dim_name}' already exists in the dataset.")

        resampled: Dict[str, xr.DataArray] = {}
        uncertainties: Dict[str, Dict[str, tuple]] = {}
        replaced: List[str] = []
        src_dims = set()
        for name in names:
            src_dim = self._source_dim(ds[name])
            assert src_dim is not None
            src_dims.add(src_dim)
            new_name = f"{name}{suffix}"
            resampled[new_name] = self._resample(ds[name], src_dim)
            replaced.append(name)
            for unc_name in self._unc_components(ds, name):
                unc = ds[unc_name]
                new_unc = self._resample(unc, src_dim)
                err_corr = self._err_corr_from_attrs(unc.attrs, src_dim, cfg.dim_name)
                new_unc.attrs = self._strip_unc_attrs(unc.attrs)
                uncertainties.setdefault(new_name, {})[f"{unc_name}{suffix}"] = (new_unc, err_corr)
                replaced.append(unc_name)

        out = ds.drop_vars(replaced) if (cfg.inplace or cfg.drop_original) else ds.copy()
        for name, da in resampled.items():
            attrs = self._strip_unc_attrs(da.attrs)
            out[name] = da.drop_vars([c for c in da.coords if c not in out.coords and c != cfg.dim_name])
            out[name].attrs = attrs
        out = out.assign_coords({cfg.dim_name: cfg.target_wavelength})
        out[cfg.dim_name].attrs = {"units": "nm", "long_name": "Wavelength", "standard_name": "radiation_wavelength"}
        for base, comps in uncertainties.items():
            for unc_name, (da, err_corr) in comps.items():
                out.unc[base][unc_name] = (da.dims, da.values, {**da.attrs, "err_corr": err_corr})

        if cfg.drop_original or cfg.inplace:
            still_used = {str(d) for v in out.data_vars.values() for d in v.dims}
            out = out.drop_vars([d for d in src_dims if d in out.coords and d not in still_used])

        return self._record_provenance(out, names)

    @staticmethod
    def _unc_components(ds: xr.Dataset, name: str) -> List[str]:
        comps = ds[name].attrs.get("unc_comps", [])
        comps = [comps] if isinstance(comps, str) else list(comps)
        return [c for c in comps if c in ds.data_vars]

    def _record_provenance(self, ds: xr.Dataset, names: List[str]) -> xr.Dataset:
        """
        Record a minimal provenance entry at processor level.

        :param ds:
            Output dataset.
        :param names:
            Variables that were resampled.
        :return:
            Dataset (same object, attrs updated).
        """
        steps: list = ds.attrs.get("eoio:processing_steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]
        steps.append(
            {
                "processor": "resample_wavelength",
                "vars": names,
                "method": self.config.method,
                "n_wavelengths": int(self.config.target_wavelength.size),
            }
        )
        ds.attrs["eoio:processing_steps"] = steps
        return ds


if __name__ == "__main__":
    pass
