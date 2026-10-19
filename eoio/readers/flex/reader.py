"""
eoio.readers.flex.reader
========================

Reader for FLEX L2 FLXSYN products (NetCDF4).

The product is a synergy of FLORIS, OLCI and SLSTR data on a common 300 m UTM grid. Variables are
exposed under the following ``vars_sel`` selections:

``meas``
    All science variables: OLCI/SLSTR/FLORIS apparent reflectance, FLORIS real reflectance, SIF
    (spectrum, integrated, peaks, O2 bands), and the vegetation products. Shortcuts: ``"all"``,
    ``"olci"``, ``"slstr"``, ``"floris"``, ``"fluorescence"``, ``"vegetation"``.
``aux``
    Retrieved atmospheric state (water vapour, AOT, Angstrom exponent, asymmetry parameter, cloud
    probability, LST, PAR), at-surface direct/diffuse irradiance, elevation and view geometry.
    Shortcuts: ``"all"``, ``"atmosphere"``, ``"irradiance"``, ``"geometry"``.
``mask``
    ``cirrus_cloud_mask`` (the only mask in the product).

Every dataset also carries the ``x``/``y`` (projected, from the product's ``GeoTransform``) and
2-D ``latitude``/``longitude`` coordinates; the CRS comes from the file (``crs`` attribute), as the
UTM zone differs between products.

Wavelength information is kept as in the product: each spectral grid has its own dimension
(``wavelength_olci``, ``wavelength_floris``, ...). Use the ``resample_wavelength`` processor to
put spectra on a common grid. FLORIS wavelengths are stored per across-track sample but are
treated as a single vector, as only the first sample is populated in the products seen so far; this
is provisional until real products show whether a spectral smile (per-row wavelengths) must be
handled (it is expected to be corrected in L1).

Values outside the product's stated ``valid_min``/``valid_max`` are not altered (e.g.
``cloud_probability`` below 0 and ``total_integrated_sif`` far above ``valid_max`` in the test
products). These are considered product issues to be raised with the producers, not corrected in
the reader.

The split of variables between ``meas``, ``aux`` and ``mask`` follows the product's science/auxiliary
division. Placement of ``cloud_probability`` (``aux``) relative to cloud masks is to be revisited as
part of a wider consistency discussion across readers.

The files carry no acquisition time, so ``product_datetime`` is taken from the start time in the
filename. Whether a start/stop range is needed is still to be confirmed.

Memory: by default the selected (subsetted) variables are loaded into memory. With
``read_params={"use_chunks": True}`` (optionally ``"chunks": {"y": ..., "x": ...}``, default
64 x 64) they stay lazy, dask-backed arrays and are only read when computed, so a full product
can be opened and then reduced without holding it all in memory. The file stays open while the
dataset is in use.

Uncertainties (``include_uncertainties``) are registered through obsarray as ``ds.unc``
components. The product documents no error-correlation model, so a placeholder is assumed (random
across ``y``/``x``, systematic across wavelength); see :py:func:`eoio.readers.flex.data_io.err_corr_for`.
The spectral error correlation is computed and passed to the L2B retrieval but is not stored per
product (too expensive), so it cannot be read from the files; typical matrices per land-cover type
may replace the placeholder in future.

Classes
-------
.. autosummary::
   :toctree: generated/

   FLEXL2Reader
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import xarray as xr

from eoio.readers.base import BaseReader
from eoio.readers.flex.data_io import DEFAULT_CHUNKS, read_dataset, read_grid, read_wavelengths, register_uncertainties
from eoio.readers.flex.layout import FlexLayout
from eoio.readers.flex.metadata import FLEXMetadataExtractor
from eoio.readers.flex.subset import build_subset
from eoio.readers.flex.variables import (
    AUX_GROUPS,
    MEAS_GROUPS,
    UNCERTAINTY_SUFFIX,
    VARIABLES,
    variables_for,
)


class FLEXL2Reader(BaseReader):
    """Reader for FLEX L2 FLXSYN products."""

    default_vars_sel = {
        "meas": "all",
        "aux": None,
        "mask": None,
    }

    default_subset = {
        "roi": None,
        "roi_crs": 4326,
        "wavelength": None,
    }
    all_subset: Dict[str, Any] = {
        "roi": "Primary/"
        "shapely geometry/"
        "bounding box tuple (xmin, ymin, xmax, ymax)/"
        "GeoJSON-like ``dict`` with a ``type`` key/"
        "list of ``[x, y]`` coordinate pairs defining a polygon/"
        "``((x, y), half_width_m)`` defining a square box around a point",
        "roi_crs": "Any EPSG Code",
        "wavelength": ["min", "max", "nearest", "tolerance"],
    }

    default_read_params = {
        "save_extracted": False,
        "metadata_level": "all",
        "include_uncertainties": True,
        "use_chunks": False,
        "chunks": None,
    }
    all_read_params = {
        **BaseReader.all_read_params,
        "use_chunks": "True/False: keep variables lazy (dask-backed) instead of loading them into memory",
        "chunks": "None/{'x':X,'y':Y}: dask chunk sizes (default 64 x 64 when use_chunks is True)",
    }

    meas_def = {"all": variables_for("meas"), **MEAS_GROUPS}
    aux_def = {"all": variables_for("aux"), **AUX_GROUPS}
    mask_def = {"all": variables_for("mask")}

    uncertainty_vars = [f"{n}{UNCERTAINTY_SUFFIX}" for n, s in VARIABLES.items() if s.has_uncertainty]

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param path: path to the FLEX product file.
        :param vars_sel: variable selection (``meas``, ``aux``, ``mask``).
        :param subset: subset request (``roi``, ``roi_crs``, ``wavelength``).
        :param read_params: read options (``metadata_level``, ``include_uncertainties``,
            ``use_chunks``, ``chunks``). ``save_extracted`` is accepted for consistency with other
            readers but has no effect: FLEX products are plain NetCDF files, never extracted.
        """
        self.layout = FlexLayout(path)
        super().__init__(path, vars_sel=vars_sel, subset=subset, read_params=read_params)

    def list_uncertainty_vars(self) -> List[str]:
        """Uncertainty variables of the selected meas/aux variables only."""
        selected = self.list_selected_meas() + self.list_selected_aux()
        return [f"{v}{UNCERTAINTY_SUFFIX}" for v in selected if v in VARIABLES and VARIABLES[v].has_uncertainty]

    def _chunks(self) -> Optional[Dict[str, int]]:
        """Dask chunk sizes if lazy reading was requested, otherwise ``None`` (load into memory)."""
        rp = self.config.read_params
        if rp.get("chunks"):
            return dict(rp["chunks"])
        return dict(DEFAULT_CHUNKS) if rp.get("use_chunks") else None

    def open_dataset(self) -> xr.Dataset:
        """
        Open the FLEX product as an ``xarray.Dataset`` according to the request parameters.

        :returns: the opened and subsetted dataset.
        """
        path = str(self.path)
        grid = read_grid(path)
        wavelengths = read_wavelengths(path)
        subset = build_subset(grid, wavelengths, self.config.subset)

        ds = read_dataset(
            path=path,
            include_vars=self.list_include_vars(),
            grid=grid,
            wavelengths=wavelengths,
            subset=subset,
            chunks=self._chunks(),
        )
        ds = register_uncertainties(ds)

        mtd_level = self.config.read_params.get("metadata_level", None)
        if mtd_level is True:
            mtd_level = "all"

        if mtd_level == "original" or mtd_level is None:
            return ds

        with xr.open_dataset(path) as root:
            ds.attrs.update(root.attrs)
        meta_ex = FLEXMetadataExtractor(self, ds, self.layout, grid)
        ds = meta_ex.clear_metadata(ds)
        if mtd_level in ("all", "basic"):
            ds = meta_ex.attach_metadata(ds, level=mtd_level)
        return ds

    @staticmethod
    def get_extension() -> str:
        """Return the file extension of FLEX products."""
        return ".nc"


if __name__ == "__main__":
    pass
