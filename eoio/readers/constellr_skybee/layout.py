"""eoio.readers.constellr_skybee.layout - helper for constellr Skybee file layout parsing."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

#: ground sampling distance (m) of each VNIR band, B<n> being the product's ``vnir<nn>`` files
MEAS_VAR_RES: dict[str, int] = {
    "B1": 60,
    "B2": 10,
    "B3": 10,
    "B4": 10,
    "B5": 20,
    "B6": 20,
    "B7": 20,
    "B8": 10,
    "B9": 20,
    "B10": 60,
}


class ConstellrSkybeeLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class ConstellrSkybeeLayout:
    """Filesystem/layout helper for constellr Skybee products."""

    product_dir: Path

    def __post_init__(self):
        p = Path(self.product_dir)
        if not p.exists():
            raise ConstellrSkybeeLayoutError(f"Skybee product path not found: {self.product_dir}")
        if not p.is_dir():
            raise ConstellrSkybeeLayoutError(f"Skybee product path is not a dir: {self.product_dir}")

    @property
    def tif_img_pardir_path(self) -> Path:
        return Path(self.product_dir)

    def image_files(self) -> list[Path]:
        tiff_files = [
            tiff_file
            for tiff_file in Path(self.product_dir).glob("*.tiff")
            if "qa" not in tiff_file.name and "composite" not in tiff_file.name
        ]
        if not tiff_files:
            raise ConstellrSkybeeLayoutError(f"No tiff files found in: {self.product_dir}")
        return sorted(tiff_files)

    def image_qa_files(self) -> list[Path]:
        # TODO use qa tiff files later for masking
        qa_tiff_files = [tiff_file for tiff_file in Path(self.product_dir).glob("*.tiff") if "qa" in tiff_file.name]

        if not qa_tiff_files:
            raise ConstellrSkybeeLayoutError(f"No QA tiff files found in: {self.product_dir}")
        return sorted(qa_tiff_files)

    def qa_file(self, band: str) -> Path:
        """Path of the QA raster (blackfill/saturated/untested layers) of measurement band *band*, e.g. B4."""
        token = f"vnir{int(band[1:]):02d}_qa"
        for f in self.image_qa_files():
            if Path(f).stem.lower().endswith(token):
                return f
        raise ConstellrSkybeeLayoutError(f"No QA tiff found for band {band} in: {self.product_dir}")

    def image_file(self) -> Path:
        files = self.image_files()
        for path in files:
            stem = Path(path).stem.lower()
            if re.search(r"(?:^|[_-])vnir\d+(?:[_.-]|$)", stem):
                return path
        return files[0]

    def tif_band_files(self, meas_vars: list[str] | None = None) -> dict[str, Path]:
        files = self.image_files()

        filtered_files: dict[str, Path] = {}
        if meas_vars:
            for band in meas_vars:
                # B2 -> vnir02, B10 -> vnir10
                token = f"vnir{int(band[1:]):02d}"
                for f in files:
                    if Path(f).stem.lower().endswith(token):
                        filtered_files[band] = f
                        break
        else:
            raise ConstellrSkybeeLayoutError(
                f"No TIFF band files found for meas_vars {meas_vars} in: {self.product_dir}"
            )

        if not filtered_files:
            raise ConstellrSkybeeLayoutError(
                f"No TIFF band files found for meas_vars {meas_vars} in: {self.product_dir}"
            )
        return filtered_files

    def metadata_file(self) -> Path | None:
        """The product's ``metadata.json`` (any other ``*.json`` only if that is absent), or ``None``."""
        product = Path(self.product_dir)
        if (product / "metadata.json").is_file():
            return product / "metadata.json"
        json_files = sorted(product.glob("*.json"))
        if json_files:
            return json_files[0]
        warnings.warn(f"No metadata json found in {self.product_dir}; metadata will be empty.")
        return None

    # def default_meas_vars(self) -> list[str]:
    #     vnir_naming = list(self.tif_band_files(self.).keys())
    #     return vnir_naming  # TODO check if needed here

    # def available_band_tokens(self) -> set[str]:
    #     vnir_naming = set(self.tif_band_files().keys())
    #     return vnir_naming  # TODO check if needed here
