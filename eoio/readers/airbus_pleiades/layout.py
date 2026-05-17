"""eoio.readers.airbus_pleiades.layout -  helper for Airbus Pleiades file layout parsing."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import glob
import os


class PleiadesLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class PleiadesLayout:
    """
    Layout helper for Airbus Pleiades MS_ORT (L1) products.

    Solely responsible for filesystem/path logic. No heavy dependencies.

    """

    _default_img_str = "IMG*.TIF"
    _default_metadata_str = "DIM_*.XML"

    product_dir: Path

    def __post_init__(self):
        p = Path(self.product_dir)
        if not p.exists():
            raise PleiadesLayoutError(f"Pleiades product path not found: {self.product_dir}")
        if not p.is_dir():
            raise PleiadesLayoutError(f"Pleiades product path is not a dir: {self.product_dir}")

    @property
    def tif_img_pardir_path(self) -> Path:
        return Path(self.product_dir)

    def image_file(self) -> str:
        # find file with IMG*.TIF in path
        file = glob.glob(os.path.join(self.product_dir, f"{self._default_img_str}"))
        if not file:
            raise PleiadesLayoutError(f"No image TIFs found in {self.product_dir}")
        return file[0]

    def metadata_file(self) -> Optional[Path]:
        # find file with DIM_*.XML in path
        xml_file = glob.glob(os.path.join(self.product_dir, f"{self._default_metadata_str}"))
        if not xml_file:
            raise PleiadesLayoutError(f"No metadata XML file found in {self.product_dir}")
        return Path(xml_file[0])

    def aux_dir(self) -> Path:
        """
        Return MASKS directory path.
        """
        aux_dir = Path(self.product_dir) / "MASKS"
        return aux_dir

    def aux_data_dir(self, aux_name: Optional[str] = None) -> Optional[Path]:
        """
        Return AUX_DATA directory if present.
        """
        if aux_name is None:
            return None
        aux_data = self.aux_dir() / aux_name
        if aux_data.exists():
            return self.aux_dir() / aux_name
        else:
            return None

    def aux_path(self, aux_name: str) -> Optional[Path]:
        # angle data contained in metadata xml file
        # mask data contained in MASKS dir
        aux_dir = self.aux_data_dir(aux_name=aux_name + ".GML")

        return aux_dir
