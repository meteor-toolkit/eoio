"""eoio.readers.landsat.metadata.ls_mtd_fallback - MTL.xml + raster-derived fallback for
projection/band metadata, used only when a product has no STAC/MTL JSON sidecar (see
:py:meth:`eoio.readers.landsat.layout.LandsatLayout.product_metadata_json`) -- most often an
interrupted local extraction (see :py:mod:`eoio.utils.read_utils`'s extraction-completeness
check, which now detects and heals exactly that) rather than a genuinely stripped-down
delivery, but handled here either way rather than failing outright.

Exposes the same small interface :py:class:`~eoio.readers.landsat.metadata.ls_mtd_json.
LSL1ProdJSONReader` does -- only the methods :py:class:`~eoio.readers.landsat.metadata.
extractor.LSMetadataExtractor` actually calls: ``find_all_band_central_wavelengths``,
``find_all_band_gsds``, ``find_epsg``, ``find_proj_shape``, ``find_proj_transform`` -- so
``LSMetadataExtractor`` can use either interchangeably without caring which one it got.
"""

from __future__ import annotations

from typing import Dict, Optional

import rasterio

from eoio.readers.landsat.layout import LandsatLayout
from eoio.readers.landsat.metadata.ls_mtd_xml import LSL1ProdXMLReader

#: Nominal OLI/TIRS band centre wavelengths (nm) -- identical for Landsat 8 and 9 (same
#: instrument design), per USGS's published Landsat 8-9 OLI/TIRS band designations. The STAC
#: json's own "center_wavelength" is this same fixed instrument spec, not a per-scene
#: measurement, so hardcoding it here as a fallback loses no real information.
NOMINAL_BAND_WAVELENGTHS_NM: Dict[str, float] = {
    "B1": 443.0,
    "B2": 482.0,
    "B3": 561.5,
    "B4": 654.6,
    "B5": 864.7,
    "B6": 1608.9,
    "B7": 2200.7,
    "B8": 589.5,
    "B9": 1373.4,
    "B10": 10895.0,
    "B11": 12005.0,
}

#: Which MTL.xml GRID_CELL_SIZE_* field each band's GSD comes from.
_REFLECTIVE_BANDS = {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"}
_PANCHROMATIC_BANDS = {"B8"}
_THERMAL_BANDS = {"B10", "B11"}


class LSMTDFallbackReader:
    """
    Stand-in for :py:class:`~eoio.readers.landsat.metadata.ls_mtd_json.LSL1ProdJSONReader`
    when a product has no STAC/MTL JSON sidecar.

    Band GSD comes from MTL.xml's own ``GRID_CELL_SIZE_REFLECTIVE/PANCHROMATIC/THERMAL``
    fields; EPSG/``proj:shape``/``proj:transform`` come from one of the product's own GeoTIFF
    band files, read directly via rasterio -- arguably more authoritative than the STAC
    sidecar's own copy of the same information anyway, since it's the raster's own embedded
    georeferencing, not a separately-maintained record of it. Every band in one Landsat
    product shares the same CRS (only the pixel grid/resolution differs between the
    reflective/panchromatic/thermal band groups), so any single band file is representative
    enough for the product-level EPSG this supplies.
    """

    def __init__(self, layout: LandsatLayout, xml_reader: LSL1ProdXMLReader):
        """
        :param layout: the product's own :py:class:`LandsatLayout`, for locating a
            representative band GeoTIFF.
        :param xml_reader: the product's own, already-initialised MTL.xml reader.
        """
        self.layout = layout
        self.xml_reader = xml_reader
        self._raster_meta: Optional[dict] = None

    def _raster_metadata(self) -> dict:
        """EPSG/shape/transform read once (then cached) from one representative band file."""
        if self._raster_meta is None:
            band_files = self.layout.tif_band_files()
            representative_path = next(iter(band_files.values()))
            with rasterio.open(representative_path) as src:
                self._raster_meta = {
                    "epsg": src.crs.to_epsg() if src.crs is not None else None,
                    "shape": list(src.shape),
                    "transform": list(src.transform)[:6],
                }
        return self._raster_meta

    def find_all_band_central_wavelengths(self) -> Dict[str, float]:
        """
        :returns: {band: wavelength_nm} -- see :py:data:`NOMINAL_BAND_WAVELENGTHS_NM`.
        """
        return dict(NOMINAL_BAND_WAVELENGTHS_NM)

    def find_all_band_gsds(self) -> Dict[str, float]:
        """
        :returns: {band: gsd_m}, from MTL.xml's GRID_CELL_SIZE_* fields.
        """
        gsds: Dict[str, float] = {}
        reflective = self.xml_reader.find_value("grid_cell_size_reflective", default=None)
        panchromatic = self.xml_reader.find_value("grid_cell_size_panchromatic", default=None)
        thermal = self.xml_reader.find_value("grid_cell_size_thermal", default=None)
        for band in _REFLECTIVE_BANDS:
            if reflective is not None:
                gsds[band] = float(reflective)
        for band in _PANCHROMATIC_BANDS:
            if panchromatic is not None:
                gsds[band] = float(panchromatic)
        for band in _THERMAL_BANDS:
            if thermal is not None:
                gsds[band] = float(thermal)
        return gsds

    def find_epsg(self, default: Optional[int] = None) -> Optional[int]:
        """
        :param default: value to return if the representative band file has no CRS.
        :returns: EPSG code, read from a representative band GeoTIFF's own CRS.
        :raises ValueError: if unavailable and no default is given -- matching
            ``LSL1ProdJSONReader.find_epsg``'s own contract.
        """
        epsg = self._raster_metadata()["epsg"]
        if epsg is None:
            if default is not None:
                return default
            raise ValueError("No CRS found on the product's own GeoTIFF band files (and no STAC json to fall back to).")
        return epsg

    def find_proj_shape(self, default=None):
        """
        :param default: value to return if unavailable.
        :returns: ``[rows, cols]``, read from a representative band GeoTIFF.
        """
        shape = self._raster_metadata()["shape"]
        return shape if shape else default

    def find_proj_transform(self, default=None):
        """
        :param default: value to return if unavailable.
        :returns: affine transform as a 6-element list, read from a representative band
            GeoTIFF.
        """
        transform = self._raster_metadata()["transform"]
        return transform if transform else default


if __name__ == "__main__":
    pass
