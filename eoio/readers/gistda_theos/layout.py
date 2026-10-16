"""eoio.readers.gistda_theos.layout - helper for GISTDA THEOS file layout parsing."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


class GISTDATHEOSLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class GISTDATHEOSLayout:
    """Filesystem/layout helper for GISTDA THEOS DIMAP products.

    Expects an already-extracted product directory, same as every other
    ``get_extension() == ""`` reader here (e.g. Landsat): a zip/tar archive is
    extracted upstream by :py:func:`eoio.utils.read_utils.setup_file`, the common
    entry point every reader is constructed through (:py:func:`eoio.interface.read`,
    ``product_options``, ...), before the reader -- and so this layout -- ever sees
    the path. Callers that build a reader directly instead (bypassing that common
    entry point) are responsible for extracting first.

    *product_dir* may or may not directly hold METADATA.DIM/IMAGERY.TIF -- see
    :py:attr:`dimap_dir`.
    """

    product_dir: Path

    def __post_init__(self):
        p = Path(self.product_dir)
        if not p.exists():
            raise GISTDATHEOSLayoutError(f"THEOS product path not found: {self.product_dir}")
        if not p.is_dir():
            raise GISTDATHEOSLayoutError(f"THEOS product path is not a dir: {self.product_dir}")

    @cached_property
    def dimap_dir(self) -> Path:
        """The directory directly holding METADATA.DIM/IMAGERY.TIF.

        A THEOS delivery zip's top level holds loose wrapper files (README/PDF/
        LOGO) alongside one folder actually named for the product id -- extracting
        such a zip (see :py:func:`eoio.utils.read_utils.extract_zipped_file`) wraps
        all of that, DIMAP folder included, in one container directory named for
        the outer (order-derived) zip filename. So ``product_dir`` (that container)
        does not directly hold METADATA.DIM/IMAGERY.TIF -- its one subdirectory
        that does, does.

        Also accepts ``product_dir`` already *being* that DIMAP folder directly
        (METADATA.DIM present at its own top level) -- e.g. an archive laid out by
        hand with the wrapper files discarded, not derived from extracting a real
        delivery zip.

        :raises GISTDATHEOSLayoutError: neither ``product_dir`` itself nor exactly
            one of its immediate subdirectories holds a METADATA.DIM.
        """
        p = Path(self.product_dir)
        if (p / "METADATA.DIM").exists():
            return p
        candidates = [d for d in p.iterdir() if d.is_dir() and (d / "METADATA.DIM").exists()]
        if len(candidates) == 1:
            return candidates[0]
        raise GISTDATHEOSLayoutError(
            f"No unique DIMAP folder (containing a METADATA.DIM) found in or directly under "
            f"{self.product_dir} (found {len(candidates)} candidate subfolder(s): {candidates})"
        )

    @property
    def tif_img_pardir_path(self) -> Path:
        return self.dimap_dir

    def image_file(self) -> str:
        files = glob.glob(os.path.join(self.dimap_dir, "IMAGERY.TIF"))
        if not files:
            raise GISTDATHEOSLayoutError(f"No image TIFF found in {self.dimap_dir}")
        return files[0]

    def metadata_file(self) -> Path | None:
        xml_files = glob.glob(os.path.join(self.dimap_dir, "METADATA.DIM"))
        if not xml_files:
            raise GISTDATHEOSLayoutError(f"No metadata DIM file found in {self.dimap_dir}")
        return Path(xml_files[0])

    def aux_dir(self) -> Path:
        return self.dimap_dir / "AUX_DATA"

    def aux_data_dir(self, aux_name: str | None = None) -> Path | None:
        if aux_name is None:
            return None
        aux_path = self.aux_dir() / aux_name
        return aux_path if aux_path.exists() else None

    def aux_path(self, aux_name: str) -> Path | None:
        return self.aux_data_dir(aux_name=aux_name)
