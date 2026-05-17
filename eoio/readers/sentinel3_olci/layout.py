"""eoio.readers.sentinel3_olci.layout - Sentinel-3 OLCI product file structure and path logic."""

import os
import glob
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


class S3LayoutError(ValueError):
    """Raised when a SEN3 folder is missing expected files/structure."""


@dataclass(frozen=True)
class S3OLCILayout:
    """Representation of a Sentinel-3 OLCI SEN3 product layout.

    This class encapsulates filesystem layout logic for a SEN3 directory
    (extracted product). It provides helpers to discover radiance and
    uncertainty files, optional auxiliary files (geo coordinates, tie
    geometries, removed pixels) and basic product metadata locations.

    :param path: Filesystem path to the SEN3 directory (string or Path-like).
    :type path: str | pathlib.Path
    """

    path: str

    def __post_init__(self) -> None:
        """Validate SEN3 path."""

        p = Path(self.path)
        if not p.exists():
            raise S3LayoutError(f"SEN3 path not found: {self.path}")
        if not p.is_dir():
            raise S3LayoutError(f"SEN3 path is not a directory: {self.path}")

    @property
    def proc_version(self) -> int:
        """Parse and return the processing version embedded in the SEN3
        directory name.

        The implementation extracts a numeric token from the final part
        of the directory name and converts it to an integer. This is used
        by higher-level code to decide behaviour that depends on
        processing baseline (e.g. uncertainty handling).

        :returns: Processing version parsed from the SEN3 directory name.
        """

        return int(os.path.split(self.path)[-1].split(".")[0][-3:])

    def radiance_paths(self, bands=None):
        """Return mapping of band name to radiance file path.

        If ``bands`` is ``None``, all available '*_radiance.nc' files under
        the SEN3 directory are discovered and returned. When ``bands`` is a
        sequence, only the requested band names are considered.

        :param bands: Optional list of band tokens to filter by.
        :returns: Mapping from band token (e.g. 'Oa01') to absolute file path.
        """
        # If bands is None, find all available bands
        if bands is None:
            nc_files = glob.glob(os.path.join(self.path, "*_radiance.nc"))
            band_names = [os.path.split(f)[-1].split("_")[0] for f in nc_files]
            return {b: f for b, f in zip(band_names, nc_files)}
        else:
            paths = {}
            for b in bands:
                files = glob.glob(os.path.join(self.path, f"{b}_radiance.nc"))
                if files:
                    paths[b] = files[0]
            return paths

    def requested_uncertainty_paths(self, bands):
        """Return uncertainty file paths for requested bands.

        Scans for files matching '*_unc.nc' and filters by the supplied
        band list.

        :param bands: List of band tokens to filter by.
        :returns: Mapping of band token to uncertainty file path for those
            bands present in the SEN3 directory.
        """
        all_paths = {}
        nc_files = glob.glob(os.path.join(self.path, "*_unc.nc"))
        for f in nc_files:
            band_name = os.path.split(f)[-1].split("_")[0]
            all_paths[band_name] = f
        return {b: all_paths[b] for b in bands if b in all_paths}

    def requested_aux_paths(self, aux_names):
        """Return paths for requested auxiliary data.

        Scans for expected auxiliary files (geo_coordinates.nc, tie_geometries.nc,
        removed_pixels.nc, tie_meteo.nc) and returns paths for those that exist
        and are requested.

        :param aux_names: Iterable of auxiliary names to look up.
        :returns: Mapping of requested auxiliary names to their file paths.
        """
        aux_paths = {}
        if "geo_coordinates" in aux_names:
            geo_path = self.geo_coordinates_path()
            if geo_path is not None:
                aux_paths["geo_coordinates"] = geo_path
        if "tie_geometries" in aux_names:
            tie_geo_path = self.tie_geometries_path()
            if tie_geo_path is not None:
                aux_paths["tie_geometries"] = tie_geo_path
        if "removed_pixels" in aux_names:
            removed_path = self.removed_pixels_path()
            if removed_path is not None:
                aux_paths["removed_pixels"] = removed_path
        if "tie_meteo" in aux_names:
            tie_meteo_path = self.tie_meteo_path()
            if tie_meteo_path is not None:
                aux_paths["tie_meteo"] = tie_meteo_path
        return aux_paths

    def default_meas(self) -> List[str]:
        """Return the default set of measurement variables for this product.

        Currently this returns the set of available band tokens discovered
        from filenames in the SEN3 directory.

        :returns: List of default measurement variable tokens.
        """
        return list(self.radiance_paths().keys())

    def geo_coordinates_path(self):
        """Return the path to ``geo_coordinates.nc`` if it exists, else ``None``.

        :returns: Path to the geo coordinates file or ``None`` when missing.
        """
        path = os.path.join(self.path, "geo_coordinates.nc")
        if os.path.exists(path):
            return path
        return None

    def quality_flags_path(self):
        """Return the path to ``qualityFlags.nc`` if it exists, else ``None``.

        :returns: Path to the quality flags file or ``None`` when missing.
        """
        path = os.path.join(self.path, "qualityFlags.nc")
        if os.path.exists(path):
            return path
        return None

    def tie_geo_coordinates_path(self):
        """Return the path to the tie point geo coordinates file, or ``None``.

        Tie point files are optional; callers should handle ``None``.

        :returns: Path to the tie point geo coordinates file or ``None``.
        """
        path = os.path.join(self.path, "tie_geo_coordinates.nc")
        if os.path.exists(path):
            return path
        return None

    def tie_geometries_path(self):
        """Return the path to the tie geometries file, or ``None``.

        :returns: Path to the tie geometries file or ``None``.
        """
        path = os.path.join(self.path, "tie_geometries.nc")
        if os.path.exists(path):
            return path
        return None

    def removed_pixels_path(self):
        """Return the path to a removed-pixels mask file, or ``None``.

        :returns: Path to the removed-pixels mask file or ``None``.
        """
        path = os.path.join(self.path, "removed_pixels.nc")
        if os.path.exists(path):
            return path
        return None

    def instrument_data_path(self):
        """Return the path to instrument data file, or ``None`` if absent.

        :returns: Path to the instrument data file or ``None``.
        """
        path = os.path.join(self.path, "instrument_data.nc")
        if os.path.exists(path):
            return path
        return None

    def tie_meteo_path(self):
        """Return the path to tie meteo file, or ``None`` if absent.

        :returns: Path to the tie meteo file or ``None``.
        """
        path = os.path.join(self.path, "tie_meteo.nc")
        if os.path.exists(path):
            return path
        return None

    def manifest_path(self) -> Optional[Path]:
        """Locate product-level metadata at the SEN3 root.

        Example candidates include ``xfdumanifest.xml``.

        :returns: Path to the product metadata XML file if present, else ``None``.
        """
        cands = sorted(Path(self.path).glob("xfdumanifest.xml"))

        return cands[0] if cands else None
