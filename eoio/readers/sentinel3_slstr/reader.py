"""eoio.readers.sentinel3_slstr.reader - Sentinel-3 SLSTR L1 reader adapter.

Thin adapter that mirrors the sentinel3_olci style: it provides a small
`BaseRasterReader`-compatible class which orchestrates the package-level
helpers (`layout`, `metadata`, `data_io`, `auxiliary`, `masks`, `conventions`).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import xarray as xr

from eoio.readers.base import BaseRasterReader
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset
from .layout import S3SLSTRLayout
from .metadata.extractor import S3SLSTRMetadataExtractor
from .data_io import read_bands_into_dataset, read_lat_lon_coordinates
from .auxiliary import add_aux
from .masks import add_masks
from .conventions import apply_conventions

__all__ = ["SLSTRL1Reader"]


class SLSTRL1Reader(BaseRasterReader):
    """Sentinel-3 SLSTR L1 reader.

    This adapter follows the ``eoio.readers.*.reader`` style: it accepts the
    standard BaseReader constructor arguments and exposes ``open()`` which
    returns an xarray.Dataset.

    :cvar default_read_params: Default read parameters for the reader.
    :cvar all_read_params: Documentation strings for supported read params.
    """

    default_read_params: Dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",
        "include_uncertainties": False,
        "use_chunks": False,
        "chunks": None,
    }

    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",
        "include_uncertainties": "True/False",
        "use_chunks": "True/False",
        "chunks": "None/{'x':X,'y':Y}",
    }

    MASK_OPTIONS = ["bayes", "cloud", "confidence", "pointing", "exception"]
    AUX_OPTIONS = [
        "cartesian",
        "indices",
        "met",
        "time",
        "viscal",
        "observation_geometry",
        "orphan",
    ]
    MEAS_VARS = [
        "S1_radiance_an",
        "S1_radiance_ao",
        "S2_radiance_an",
        "S2_radiance_ao",
        "S3_radiance_an",
        "S3_radiance_ao",
        "S4_radiance_an",
        "S4_radiance_ao",
        "S4_radiance_bn",
        "S4_radiance_bo",
        "S5_radiance_an",
        "S5_radiance_ao",
        "S5_radiance_bn",
        "S5_radiance_bo",
        "S6_radiance_an",
        "S6_radiance_ao",
        "S6_radiance_bn",
        "S6_radiance_bo",
        "S7_BT_in",
        "S7_BT_io",
        "S8_BT_in",
        "S8_BT_io",
        "S9_BT_in",
        "S9_BT_io",
        "F1_BT_fn",
        "F1_BT_fo",
        "F2_BT_in",
        "F2_BT_io",
    ]

    grid_res = {
        "an": 500,
        "bn": 500,
        "ao": 500,
        "bo": 500,
        "in": 1000,
        "io": 1000,
        "fn": 1000,
        "fo": 1000,
        "tx": 16000,
    }

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = Path(path)

        # layout and metadata extractor
        self.layout = S3SLSTRLayout(str(self.path))
        self.mtd = S3SLSTRMetadataExtractor(self)

        # set grids
        self.grids: List[str] = []
        self.clip_boxes: Dict[str, Any] = {}

        # Initialise the variable def dictionaries
        self.meas_def = {
            "all": self.layout.default_meas(),
        }
        self.mask_def = {"all": list(self.MASK_OPTIONS)}
        self.aux_def = {"all": list(self.AUX_OPTIONS)}

        super().__init__(path, vars_sel, subset, read_params)

    def resolve_subset(self, subset: Optional[Dict[str, Any]]) -> Optional[ResolvedROISubset]:
        """
        Resolve the subset of the SLSTR product.

        :param subset: Subset of the SLSTR product.
        :returns: Resolved ROI subset.
        """
        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")
        image_crs = self.mtd.product_metadata.get("crs_code", None)

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS not available.")

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        """Open the Sentinel-3 SLSTR product and return an xarray.Dataset.

        The method resolves configuration, reads coordinate and image data,
        attaches metadata and optionally includes auxiliary data and masks.

        :returns: An xarray Dataset representing the opened SLSTR product.
        """
        # Unpack resolved config
        meas = self.resolved_config.vars_sel.get("meas", None)
        roi_subset = self.resolved_config.subset
        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level", None)

        if not hasattr(self, "grids") or self.grids is None:
            self.grids = []
        if not hasattr(self, "clip_boxes") or self.clip_boxes is None:
            self.clip_boxes = {}

        # validate/expand `meas` when provided as a string
        if isinstance(meas, str):
            if meas == "all":
                meas = self.meas_def.get("all", [])
            else:
                raise ValueError(f"Unknown meas selection: {meas}")

        # derive product grids from selected bands and auxiliary variables
        for b in meas or []:
            g = b[-2:]
            if g not in self.grids:
                self.grids.append(g)
        _aux_sel = self.resolved_config.vars_sel.get("aux") or []
        if "met" in _aux_sel or "observation_geometry" in _aux_sel:
            self.grids.append("tx")

        # initialise dataset
        ds = xr.Dataset()

        # Read lat/lon coordinates and compute clip boxes for ROI subsetting
        ds, self.clip_boxes = read_lat_lon_coordinates(
            ds=ds,
            layout=self.layout,
            grids=self.grids,
            subset=roi_subset,
            config=self.resolved_config,
            use_chunks=bool(rp.get("use_chunks", False)),
            chunks=rp.get("chunks"),
        )

        # Read image data into dataset
        if meas:
            ds = read_bands_into_dataset(
                ds=ds,
                layout=self.layout,
                meas=meas,
                subset=roi_subset,
                clip_boxes=self.clip_boxes,
                mtd=self.mtd,
                use_chunks=bool(rp.get("use_chunks", False)),
                chunks=rp.get("chunks"),
            )

        # Attach metadata attributes if requested
        if mtd_level is True:
            mtd_level = "all"
        if isinstance(mtd_level, str) and mtd_level.lower() in ("all", "basic"):
            ds = self.mtd.attach_metadata(ds, level=mtd_level.lower())

        # Add auxiliary and masks if requested
        if self.resolved_config.vars_sel["aux"]:
            ds = add_aux(
                ds=ds,
                layout=self.layout,
                grids=self.grids,
                subset=roi_subset,
                clip_boxes=self.clip_boxes,
                config=self.resolved_config,
                use_chunks=bool(rp.get("use_chunks", False)),
                chunks=rp.get("chunks"),
            )

        if self.resolved_config.vars_sel["mask"]:
            ds = add_masks(
                ds=ds,
                layout=self.layout,
                grids=self.grids,
                clip_boxes=self.clip_boxes,
                config=self.resolved_config,
                use_chunks=bool(rp.get("use_chunks", False)),
                chunks=rp.get("chunks"),
            )

        # Apply conventions (noop by default)
        ds = apply_conventions(ds, layout=self.layout, config=self.resolved_config)

        return ds

    @classmethod
    def get_extension(cls) -> str:
        """Return the extension of the extracted Sentinel-3 SLSTR product.

        :returns: File extension for extracted SEN3 products."""
        return ".SEN3"


if __name__ == "__main__":
    pass
