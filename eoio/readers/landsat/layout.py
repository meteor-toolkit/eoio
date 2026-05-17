"""eoio.readers.landsat.layout - Helper for Landsat product layout parsing."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import glob
import re
import os


class LandsatLayoutError(ValueError):
    """Raised when a Landsat layout is missing expected files/structure."""


_BAND_TOKEN_RE = re.compile(r"\bB(?P<band>(?:[1-9]))\b")  # e.g. B2, B8


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
        Return the product metadata JSON file for the Landsat product (STAC preferred).

        :return: Path to the product metadata JSON file, if it exists.
        """
        # Match both uppercase (*STAC.json, as on Windows) and lowercase (*stac.json,
        # as shipped by USGS) to stay portable across case-sensitive filesystems.
        candidates = glob.glob(os.path.join(self.product_dir, "*STAC.json")) or glob.glob(
            os.path.join(self.product_dir, "*stac.json")
        )
        if not candidates:
            raise LandsatLayoutError(f"No STAC product metadata JSON file found in: {self.product_dir}")
        return Path(candidates[0])

    def available_band_tokens(self) -> Set[str]:
        """
        Return the unique set of available band tokens based on the product files, e.g.: {'B1', 'B2', 'B3', ...'B9'}

        :return: Set of available band tokens.
        """

        tif_files = self.tif_band_files()
        return {match.group("band") for f in tif_files if (match := _BAND_TOKEN_RE.search(Path(f).stem))}

    def tif_band_files(self, meas_vars=None) -> dict:
        """
        Return a list of all TIFF files containing band data in the product directory.

        :return: List of TIFF band file paths.
        """
        files = glob.glob(os.path.join(self.product_dir, "*B*.TIF"))
        if meas_vars is None:
            meas_vars = [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B9",
                "B10",
                "B11",
            ]
        filtered_files = {}
        for f in files:
            stem = Path(f).stem
            for band in meas_vars:
                if band in stem:
                    filtered_files[band] = f
                    break
        if not filtered_files:
            raise LandsatLayoutError(f"No TIFF band files found for meas_vars {meas_vars} in: {self.product_dir}")
        return filtered_files

    def get_angle_files(self) -> Dict[str, str]:
        """
        Return a dictionary of angle file paths for illumination and viewing angles.

        :return: Dict of angle file paths.
        :raises LandsatLayoutError: If any expected angle file is missing.
        """
        angle_files = {}
        required_files = {
            "viewing_zenith_angle": "*_VZA.tif",
            "solar_zenith_angle": "*_SZA.tif",
            "viewing_azimuth_angle": "*_VAA.tif",
            "solar_azimuth_angle": "*_SAA.tif",
        }

        for key, pattern in required_files.items():
            candidates = glob.glob(os.path.join(self.product_dir, pattern))
            if not candidates:
                raise LandsatLayoutError(f"Missing expected angle file matching '{pattern}' in: {self.product_dir}")
            angle_files[key] = candidates[0]

        return angle_files


if __name__ == "__main__":
    pass
