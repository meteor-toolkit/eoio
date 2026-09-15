"""eoio.readers.landsat.layout - Helper for Landsat product layout parsing."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import glob
import os
import warnings


class LandsatLayoutError(ValueError):
    """Raised when a Landsat layout is missing expected files/structure."""


@dataclass(frozen=True)
class LandsatLayout:
    """
    Layout helper for Landsat products (e.g. Landsat 8 OLI/TIRS L1/L2).

    Responsible for filesystem/path logic. No heavy dependencies.
    """

    path: str

    def __post_init__(self) -> None:
        """Validate the product path."""

        p = Path(self.path)
        if not p.exists():
            raise LandsatLayoutError(f"Landsat path not found: {self.path}")
        if not p.is_dir():
            raise LandsatLayoutError(f"Landsat path is not a directory: {self.path}")

    @property
    def product_dir(self) -> Path:
        return Path(self.path)

    def default_meas_vars(self) -> List[str]:
        """
        Return the default measurement variables for Landsat to read if user does not specify.

        :return: List of default measurement variables inferred from filenames.
        """
        return list(self.available_band_tokens())

    def product_metadata_xml(self) -> Optional[Path]:
        """
        Return the product metadata XML file for the Landsat product.

        :return: Path to the product metadata XML file, if it exists.
        """
        candidates = glob.glob(os.path.join(self.product_dir, "*MTL.xml"))
        if not candidates:
            raise LandsatLayoutError(f"No product metadata XML file found in: {self.product_dir}")

        return Path(candidates[0])

    def product_metadata_json(self) -> Optional[Path]:
        """
        Return the product metadata JSON file for the Landsat product (STAC preferred), or
        ``None`` if the product doesn't have one.

        Missing rather than raising: every value this file supplies (per-band GSD/central
        wavelength, EPSG, proj:shape/proj:transform) has an equivalent source elsewhere --
        MTL.xml's own GRID_CELL_SIZE_* fields, fixed OLI/TIRS instrument-spec band centres,
        and the product's own GeoTIFF band files' raster metadata respectively -- see
        :py:mod:`eoio.readers.landsat.metadata.ls_mtd_fallback`, which
        :py:class:`eoio.readers.landsat.metadata.extractor.LSMetadataExtractor` falls back to
        when this returns ``None``. In practice this file is usually genuinely present in a
        USGS Collection 2 Level-1 delivery; its absence more often means an interrupted local
        extraction left some archive members missing (see
        :py:func:`eoio.utils.read_utils._extraction_is_complete`, which now detects and heals
        that) than a real product-format difference.

        :return: Path to the product metadata JSON file, or ``None`` if not found.
        """
        # Match both uppercase (*STAC.json, as on Windows) and lowercase (*stac.json,
        # as shipped by USGS) to stay portable across case-sensitive filesystems.
        candidates = glob.glob(os.path.join(self.product_dir, "*STAC.json")) or glob.glob(
            os.path.join(self.product_dir, "*stac.json")
        )
        if not candidates:
            warnings.warn(
                f"No STAC product metadata JSON file found in: {self.product_dir} -- falling back to "
                "MTL.xml + raster metadata for band GSD/wavelength and projection info."
            )
            return None
        return Path(candidates[0])

    def available_band_tokens(self) -> Set[str]:
        """
        Return the unique set of available band tokens based on the product files, e.g.: {'B1', 'B2', 'B3', ...'B11'}

        :return: Set of available band tokens.
        """
        return set(self.tif_band_files().keys())

    def tif_band_files(self, meas_vars=None) -> dict:
        """
        Return a mapping of band name to TIFF file path for the requested bands.

        Uses exact stem-suffix matching (``stem.endswith("_<band>")``), so ``B1``
        never accidentally matches ``B10`` or ``B11``.

        :return: Dict of {band_name: file_path}.
        """
        files = glob.glob(os.path.join(self.product_dir, "*B*.TIF"))
        if meas_vars is None:
            meas_vars = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
        filtered_files = {}
        for f in files:
            stem = Path(f).stem
            for band in meas_vars:
                if stem.endswith(f"_{band}"):
                    filtered_files[band] = f
                    break
        if not filtered_files:
            raise LandsatLayoutError(f"No TIFF band files found for meas_vars {meas_vars} in: {self.product_dir}")
        return filtered_files

    def get_angle_files(self) -> Dict[str, str]:
        """
        Return a dictionary of angle file paths for illumination and viewing angles.

        Matches both uppercase (.TIF) and lowercase (.tif) extensions so the method
        works on case-sensitive filesystems (e.g. OneDrive on macOS/Linux).

        :return: Dict of angle file paths.
        :raises LandsatLayoutError: If any expected angle file is missing.
        """
        angle_files = {}
        required_files = {
            "viewing_zenith_angle": "_VZA",
            "solar_zenith_angle": "_SZA",
            "viewing_azimuth_angle": "_VAA",
            "solar_azimuth_angle": "_SAA",
        }

        for key, suffix in required_files.items():
            candidates = glob.glob(os.path.join(self.product_dir, f"*{suffix}.TIF")) or glob.glob(
                os.path.join(self.product_dir, f"*{suffix}.tif")
            )
            if not candidates:
                raise LandsatLayoutError(f"Missing expected angle file matching '*{suffix}.TIF' in: {self.product_dir}")
            angle_files[key] = candidates[0]

        return angle_files


if __name__ == "__main__":
    pass
