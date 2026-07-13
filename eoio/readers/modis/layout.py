"""eoio.readers.modis.layout - helper for MODIS layout parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import os
import re


class MODISLayoutError(ValueError):
    """Raised when a MODIS layout is missing expected files/structure."""


DEFAULT_IMG_RES_M: Dict[str, int] = {
    "Band 1": 250,
    "Band 2": 250,
    "Band 3": 500,
    "Band 4": 500,
    "Band 5": 500,
    "Band 6": 500,
    "Band 7": 500,
    "Band 8": 1000,
    "Band 9": 1000,
    "Band 10": 1000,
    "Band 11": 1000,
    "Band 12": 1000,
    "Band 13": 1000,
    "Band 14": 1000,
    "Band 15": 1000,
    "Band 16": 1000,
    "Band 17": 1000,
    "Band 18": 1000,
    "Band 19": 1000,
    "Band 20": 1000,
    "Band 21": 1000,
    "Band 22": 1000,
    "Band 23": 1000,
    "Band 24": 1000,
    "Band 25": 1000,
    "Band 26": 1000,
    "Band 27": 1000,
    "Band_28": 1000,
    "Band_29": 1000,
    "Band_30": 1000,
    "Band_31": 1000,
    "Band_32": 1000,
    "Band_33": 1000,
    "Band_34": 1000,
    "Band_35": 1000,
    "Band_36": 1000,
}


@dataclass(frozen=True)
class MODISLayout:
    """
    MODIS layout helper for MODIS products.

    Solely responsible for filesystem/path logic. No heavy dependencies.
    """

    path: str
    preferred_resolution: Optional[int] = None
    geolocation_dir: Optional[Path | str] = None

    def __post_init__(self) -> None:
        """
        Validate MODIS path.
        """
        p = Path(self.path)
        if not p.exists():
            raise MODISLayoutError(f"MODIS path not found: {self.path}")

    @property
    def proc_version(self) -> int:
        """
        Return processing version number as an int.

        :return:
            Processing version.
        """
        return os.path.split(self.path)[-1].split(".")[3]

    @property
    def modis_dir(self) -> Path:
        """
        Return directory where data is stored.

        :return:
            MODIS directory path.
        """
        return Path(self.path).parent

    @property
    def modis_platform(self) -> str:
        """
        Return platform.

        :return:
            MODIS platform.
        """
        return os.path.split(self.path)[-1][:3]

    @property
    def collection(self) -> str:
        """
        Return collection.

        :return:
            MODIS collection name.
        """
        return os.path.split(self.path)[-1].split(".")[0]

    @property
    def processing_level(self) -> str:
        """
        Best-effort processing level inference from filename / metadata presence.

        :return:
            Processing level of file from filename.
        """
        product_name = self.collection

        if "MOD02" in product_name or "MYD02" in product_name:
            return "L1B"
        elif "MOD09" in product_name or "MYD09" in product_name:
            return "L2"
        else:
            raise MODISLayoutError(f"Unsupported product for reader: {product_name}")

    def geolocation_path(self) -> Path:
        """
        Return a path to corresponding MOD03 geolocation product.

        :return:
            Path to the MOD03 product.
        :raises MODISLayoutError:
            If MOD03/MYD03 is missing or the requested product cannot be found.
        """
        product_name = self.collection
        if product_name == "MOD021KM" and self.proc_version == "7":
            return Path(self.path)

        if self.geolocation_dir is None:
            aux_dir = str(self.modis_dir).replace("D09", "D03")
        else:
            aux_dir = self.geolocation_dir

        aux_dir = Path(aux_dir)

        aux_name = f"{self.modis_platform}03." + ".".join(os.path.split(self.path)[-1].split(".")[1:3])
        cands = sorted(aux_dir.glob(f"{aux_name}*"))
        if cands:
            return cands[0]

        deep = sorted(aux_dir.rglob(f"{aux_name}*"))
        if deep:
            return deep[0]

        raise MODISLayoutError(f"MOD03/MYD03 product '{aux_name}' not found under {aux_dir}")

    @property
    def file_res_key(self) -> str:
        """
        Return file resolution key from filename.

        :return:
            File resolution key (e.g. '1', 'H', 'Q').
        """
        preferred_res_dict = {"1": 1000, "H": 500, "Q": 250}

        if self.processing_level == "L1B":
            return os.path.split(self.path)[-1][5]
        elif self.processing_level == "L2":
            return next((key for key, res in preferred_res_dict.items() if res == self.preferred_resolution), "1")
        else:
            raise MODISLayoutError(f"Unsupported processing level '{self.processing_level}' for file '{self.path}'")

    def default_meas_vars(self) -> List[str]:
        """
        Return default measurement variables to read.

        :return:
            Default measurement variable list.
        """
        file_res_dict = {"1": 36, "H": 7, "Q": 2}

        file_res_key = self.file_res_key
        num_bands = file_res_dict.get(file_res_key)
        if num_bands is None:
            raise MODISLayoutError(f"Unsupported file resolution key '{file_res_key}' in filename '{self.path}'")

        bands = [f"Band {i}" for i in range(1, num_bands + 1)]
        return bands


if __name__ == "__main__":
    pass
