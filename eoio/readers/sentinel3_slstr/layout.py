"""eoio.readers.sentinel3_slstr.layout - Sentinel-3 SLSTR product layout helpers.

This module mirrors the style of `sentinel3_olci.layout` and provides a small
helper class that knows where to find SLSTR-specific files inside an extracted
SEN3 product directory.
"""

from glob import glob
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from eoio.readers.sentinel3_slstr.utils import GRID_RES_MAP


class S3SLSTRLayoutError(ValueError):
    """Raised when a SEN3 folder is missing expected files/structure."""


@dataclass(frozen=True)
class S3SLSTRLayout:
    """Representation of a Sentinel-3 SLSTR SEN3 product layout.

    The class provides helpers to discover radiance/BT files, auxiliary
    files (geodetic, cartesian, indices, meteo, time, viscal, geometry)
    and the product manifest.

    :param path: Filesystem path to the SEN3 directory (string or Path-like).
    """

    path: str

    def __post_init__(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise S3SLSTRLayoutError(f"SEN3 path not found: {self.path}")
        if not p.is_dir():
            raise S3SLSTRLayoutError(f"SEN3 path is not a directory: {self.path}")

    @property
    def proc_version(self) -> int:
        """Parse processing version from the SEN3 directory name.

        This attempts to extract a numeric token from the directory name;
        callers should handle unexpected formats.
        :returns: Processing version as an integer, or 0 if it cannot be parsed.
        """
        name = os.path.split(self.path)[-1]
        # try to find a trailing numeric token (fallback to 0)
        try:
            return int([t for t in name.split(".") if t.isdigit()][-1])
        except Exception:
            # best-effort: take last 3 digits if present
            digits = "".join([c for c in name if c.isdigit()])
            return int(digits[-3:]) if len(digits) >= 3 else 0

    def meas_paths(self, bands: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """Return mapping of band name to radiance file path.

        If ``bands`` is ``None``, all available radiance or BT files under
        the SEN3 directory are discovered and returned. When ``bands`` is a
        sequence, only the requested band names are considered.

        :param bands: Optional list of band tokens to filter by.
        :returns: Mapping from band token (e.g. 'S1_an') to absolute file path.
        """

        if bands is None:
            regex = re.compile(r".*(BT|radiance).*")
            nc_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if regex.match(f)]
            band_names = [Path(f).stem for f in nc_files]
            return {b: f for b, f in zip(band_names, nc_files)}
        else:
            paths = {}
            for b in bands:
                files = glob(os.path.join(self.path, f"{b}.nc"))
                if files:
                    paths[b] = files[0]
            return paths

    def default_meas(self) -> list:
        """Return default list of measurement variable stems available in the product.
        This is used as a fallback if no measurements are explicitly selected in the reader config.
        :returns: List of default measurement variable stems (e.g. ['S1_an', 'S2_an', ...]).
        """
        return list(self.meas_paths().keys())

    def requested_uncertainty_paths(self, bands: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """Return requested uncertainty paths for the given bands.

        For now return an empty mapping; but can be implemented if in-product uncertainties become avaialable in future.

        :returns: Mapping from band token (e.g. 'S1_an') to uncertainty file path.
        """
        return {}

    def get_grid(self, band: str) -> Optional[str]:
        """Extract grid token from a band name (e.g. 'S1_an' -> 'an').
        :param band: Band tokens to extract the grid ID from

        :returns: Mapping from band token (e.g. 'S1_an') to grid (e.g. 'an').
        """
        parts = band.split("_")
        return parts[2] if len(parts) > 2 else None

    def get_grid_res(self, grid: str) -> Optional[int]:
        """Return nominal spatial resolution for a given grid token.
        :param grid: Grid ID

        :returns: Nominal spatial resolution in meters (e.g. 500), or None if grid is unrecognized.
        """
        return GRID_RES_MAP.get(grid)

    def mask_to_var(self, mask: str) -> Optional[str]:
        """Map a requested mask name to the corresponding variable name to allow mapping of mask to a meas_path.

        :param mask: Requested mask name (e.g. 'S1_exception_an').
        :returns: Corresponding variable name (e.g. "S1_radiance_an"), or None if
            the mask is not recognized.
        """
        radiance_vars = ["S1", "S2", "S3", "S4", "S5", "S6"]
        BT_vars = ["S7", "S8", "S9", "F1", "F2"]

        band = mask.split("_")[0]
        grid = mask.split("_")[-1]
        if band in radiance_vars:
            var = f"{band}_radiance_{grid}"
        elif band in BT_vars:
            var = f"{band}_BT_{grid}"
        else:
            var = None

        return var

    def geodetic_path(self, grid: str) -> Optional[str]:
        """Return path to geodetic file for a given grid, or None if not present.
        :param grid: Grid ID (e.g. 'an').
        :returns: Path to geodetic file, or None if not present.
        """
        path = os.path.join(self.path, f"geodetic_{grid}.nc")
        return path if os.path.exists(path) else None

    def cartesian_path(self, grid: str) -> Optional[str]:
        """Return path to cartesian file for a given grid, or None if not present.
        :param grid: Grid ID (e.g. 'an').
        :returns: Path to cartesian file, or None if not present.
        """
        path = os.path.join(self.path, f"cartesian_{grid}.nc")
        return path if os.path.exists(path) else None

    def flags_path(self, grid: str) -> Optional[str]:
        """Return path to flags file for a given grid, or None if not present.
        :param grid: Grid ID (e.g. 'an').
        :returns: Path to flags file, or None if not present.
        """
        path = os.path.join(self.path, f"flags_{grid}.nc")
        return path if os.path.exists(path) else None

    def indices_path(self, grid: str) -> Optional[str]:
        """Return path to indices file for a given grid, or None if not present.
        :param grid: Grid ID (e.g. 'an').
        :returns: Path to indices file, or None if not present.
        """
        path = os.path.join(self.path, f"indices_{grid}.nc")
        return path if os.path.exists(path) else None

    def met_path(self) -> Optional[str]:
        """
        Return path to meteo file
        :returns: Path to meteo file, or None if not present.
        """
        path = os.path.join(self.path, "met_tx.nc")
        return path if os.path.exists(path) else None

    def time_path(self, grid: str) -> Optional[str]:
        """
        Return path to time data file
        :param grid: Grid ID
        :returns: Path to time file, or None if not present
        """
        path = os.path.join(self.path, f"time_{grid}.nc")
        return path if os.path.exists(path) else None

    def viscal_path(self) -> Optional[str]:
        """
        Return path to viscal data file
        :returns: Path to viscal file, or None if not present
        """
        path = os.path.join(self.path, "viscal.nc")
        return path if os.path.exists(path) else None

    def geometry_tn_path(self) -> Optional[str]:
        """
        Return path to nadir geometry file
        :returns: Path to nadir geometry file, or None if not present
        """
        path = os.path.join(self.path, "geometry_tn.nc")
        return path if os.path.exists(path) else None

    def geometry_to_path(self) -> Optional[str]:
        """
        Return path to oblique geometry file
        :returns: Path to oblique geometry file, or None if not present
        """
        path = os.path.join(self.path, "geometry_to.nc")
        return path if os.path.exists(path) else None

    def manifest_path(self) -> Optional[Path]:
        """
        Return path to metadata manifest file (xfdumanifest.xml)
        :returns: Path to metadata manifest file, or None if not present
        """
        cands = sorted(Path(self.path).glob("xfdumanifest.xml"))
        return cands[0] if cands else None
