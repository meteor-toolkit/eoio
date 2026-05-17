"""eoio.readers.emit.reader - EMIT data reader implementation."""

from __future__ import annotations
from typing import Any, Dict, Optional
import xarray as xr
from pathlib import Path

from eoio.readers.emit.layout import EmitLayout
from eoio.readers.emit.subset import build_subset
from eoio.readers.emit.data_io import read_dataset, attach_coords_from_groups
from eoio.readers.emit.aux_data import add_aux
from eoio.readers.emit.metadata import EMITMetadataExtractor
from eoio.readers.generic_netcdf.reader import NetCDFReader

AUX_OPTIONS = [
    "viewing_zenith_angle",
    "viewing_azimuth_angle",
    "solar_zenith_angle",
    "solar_azimuth_angle",
    "elev",
    "Slope",
    "Aspect",
    "Path length",
    "Solar phase",
    "Cosine(i)",
    "UTC Time",
    "Earth-sun distance",
]


class EMITL1BReader(NetCDFReader):
    default_vars_sel = {
        "meas": "all",  # e.g. []
        "mask": None,  # e.g. ["cloud"]
        "aux": None,  # e.g. ["viewing_zenith_angle"]
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

    default_read = {"save_extracted": False, "metadata_level": "all"}
    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",  # None | False disables metadata; True/'basic'/'full' etc enables
    }

    meas_def = {
        "all": ["radiance"],
    }

    aux_def = {
        "all": AUX_OPTIONS,
    }

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the EMITL1BReader.

        Parameters
        ----------
        path : Path or str
            Path to the EMIT product file.
        vars_sel : dict, optional
            Variable selection dictionary.
        subset : dict, optional
            Subset dictionary specifying region of interest.
        read_params : dict, optional
            Additional read parameters.
        """
        self.layout = EmitLayout(str(path))
        super().__init__(
            path=self.layout.path,
            vars_sel=vars_sel,
            subset=subset,
            read_params=read_params,
        )

        self.ds_src = attach_coords_from_groups(self.ds_src, self.layout.path)

    def open_dataset(self) -> xr.Dataset:
        """
        Open the EMIT dataset as an xarray.Dataset according to the request parameters.

        Returns
        -------
        xr.Dataset
            The opened EMIT dataset as an xarray.Dataset.
        """

        # list measurement variables to include
        meas_vars = self.list_selected_meas()

        subset = build_subset(
            ds=self.ds_src,
            subset=self.config.subset,
        )

        ds = xr.Dataset()

        if meas_vars:
            ds = read_dataset(
                ds=ds,
                src=self.ds_src,
                meas_vars=meas_vars,
                subset=subset,
            )

        if self.config.vars_sel["aux"]:
            ds = add_aux(
                ds=ds,
                src=self.ds_src,
                layout=self.layout,
                subset=subset,
                config=self.config,
            )

        if (
            self.config.read_params.get("metadata_level", None) == "original"
            or self.config.read_params.get("metadata_level", None) is None
        ):
            return ds
        else:
            meta_ex = EMITMetadataExtractor(self, ds, self.layout)
            # ds=meta_ex.clear_metadata(ds) #TODO Is this needed for EMIT?
            if self.config.read_params.get("metadata_level", None) in ("all", "basic"):
                ds = meta_ex.attach_metadata(ds, level=self.config.read_params["metadata_level"])

        return ds
