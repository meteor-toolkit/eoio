"""eoio.readers.landsat.metadata.ls_mtd_json - reader for Landsat JSON metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json


class LSL1ProdJSONReader:
    """
    Landsat Collection-2 Level-1 JSON metadata reader.

    Supports STAC item JSON (preferred) and MTL JSON (limited).
    """

    def __init__(self, json_path: Path | str):
        """
        Initialise the JSON reader.

        :param json_path: Path to STAC item JSON or MTL JSON.
        """
        self.path = Path(json_path)
        with self.path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Identify STAC item vs. MTL JSON payload
        self._is_stac = self.data.get("type") == "Feature" or "stac_version" in self.data

        # Flatten MTL JSON wrapper if present
        if not self._is_stac and "LANDSAT_METADATA_FILE" in self.data and len(self.data) == 1:
            self.data = self.data["LANDSAT_METADATA_FILE"]

        # Lazy caches
        self._band_wavelengths: dict[str, float] | None = None
        self._band_gsds: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _iter_eo_bands(self) -> Iterable[tuple[str, dict]]:
        """
        Yield (band_name, band_dict) for STAC eo:bands entries.

        :returns: Iterable of (name, band metadata dict).
        """
        if not self._is_stac:
            return
        assets = self.data.get("assets", {}) or {}
        for asset in assets.values():
            for band in asset.get("eo:bands", []) or []:
                name = band.get("name")
                if name:
                    yield name, band

    def _build_band_cache(self) -> None:
        """
        Build caches for band central wavelengths and GSDs.

        STAC `center_wavelength` is typically in micrometers; convert to nm
        when values are < 100 (heuristic).
        """
        wavelengths: dict[str, float] = {}
        gsds: dict[str, float] = {}

        for name, band in self._iter_eo_bands() or []:
            cw = band.get("center_wavelength")
            if cw is not None:
                value = float(cw)
                # STAC center_wavelength is usually in micrometers, convert to nm
                if value < 100:
                    value *= 1000.0
                wavelengths[name] = value

            gsd = band.get("gsd")
            if gsd is not None:
                gsds[name] = float(gsd)

        self._band_wavelengths = wavelengths
        self._band_gsds = gsds

    # ------------------------------------------------------------------
    # Band-level metadata
    # ------------------------------------------------------------------
    def find_all_band_central_wavelengths(self) -> dict[str, float]:
        """
        Return a mapping of band name to central wavelength (nm).

        :returns: Dict of {band: wavelength_nm}.
        """
        if self._band_wavelengths is None:
            self._build_band_cache()
        return dict(self._band_wavelengths or {})

    def find_band_central_wavelength(self, band: str, default: float | None = None) -> float:
        """
        Return the central wavelength (nm) for a specific band.

        :param band: Band identifier (e.g., "B2", "B10").
        :param default: Value to return if band is not found.
        :returns: Central wavelength in nm.
        :raises KeyError: If band is missing and no default is provided.
        """
        if self._band_wavelengths is None:
            self._build_band_cache()
        if self._band_wavelengths and band in self._band_wavelengths:
            return self._band_wavelengths[band]
        if default is not None:
            return default
        raise KeyError(f"Band not found in JSON metadata: {band}")

    def find_all_band_gsds(self) -> dict[str, float]:
        """
        Return a mapping of band name to GSD (meters).

        :returns: Dict of {band: gsd_m}.
        """
        if self._band_gsds is None:
            self._build_band_cache()
        return dict(self._band_gsds or {})

    # ------------------------------------------------------------------
    # Projection metadata
    # ------------------------------------------------------------------
    def find_epsg(self, default: int | None = None) -> int:
        """
        Return the EPSG code of the projection from JSON metadata.

        :param default: Value to return if EPSG not found.
        :returns: EPSG code as integer.
        :raises ValueError: If EPSG is missing and no default is provided.
        """
        epsg = None
        if self._is_stac:
            epsg = (self.data.get("properties") or {}).get("proj:epsg")

        if epsg is None:
            if default is not None:
                return default
            raise ValueError("proj:epsg not found in JSON metadata")

        return int(epsg)

    def find_proj_shape(self, default=None):
        """
        Return STAC proj:shape if present.

        :param default: Value to return if missing.
        :returns: Shape list [rows, cols] or default.
        """
        if self._is_stac:
            return (self.data.get("properties") or {}).get("proj:shape", default)
        return default

    def find_proj_transform(self, default=None):
        """
        Return STAC proj:transform if present.

        :param default: Value to return if missing.
        :returns: Affine transform list or default.
        """
        if self._is_stac:
            return (self.data.get("properties") or {}).get("proj:transform", default)
        return default


if __name__ == "__main__":
    pass
