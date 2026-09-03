"""High-level metadata extractor for Sentinel-3 SLSTR products.

This extractor wraps the SLSTR XML reader implemented in
``s3_slstr_mtd.S3SLSTRXMLReader`` and exposes the same public interface
used by the rest of the readers framework (``BaseMetadataExtractor``).
"""

from __future__ import annotations
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.sentinel3_slstr.metadata.s3_slstr_mtd import S3SLSTRXMLReader
from eoio.readers.footprint_utils import normalize_footprint


class S3SLSTRMetadataExtractor(BaseMetadataExtractor):
    """High-level metadata extractor for Sentinel-3 SLSTR products.

    The extractor caches spectral information from the product manifest and
    exposes product- and variable-level metadata through the
    ``BaseMetadataExtractor`` interface.
    """

    def __init__(self, reader):
        super().__init__(reader)
        self.layout = reader.layout
        try:
            manifest_path = self.layout.manifest_path()
        except Exception:
            manifest_path = None

        if manifest_path:
            try:
                self.xml_reader = S3SLSTRXMLReader(manifest_path)
            except Exception:
                self.xml_reader = None
        else:
            self.xml_reader = None

        if self.xml_reader:
            try:
                self._central_wavelengths = self.xml_reader.find_all_band_central_wavelengths()
            except Exception:
                self._central_wavelengths = {}

            try:
                self._bandwidths = self.xml_reader.find_all_band_bandwidths()
            except Exception:
                self._bandwidths = {}

            try:
                self._crs = self.xml_reader.find_crs_code()
            except Exception:
                self._crs = None
        else:
            self._central_wavelengths = {}
            self._bandwidths = {}
            self._crs = None

    def get_basic_metadata(self) -> dict:
        if not self.xml_reader:
            return {}

        try:
            img = self.xml_reader.find_image_size()
            rows = img.get("rows")
            cols = img.get("columns")
            acquisition_start_datetime = self.xml_reader.find_acquisition_start_datetime()

            basic = {
                "collection_name": self.xml_reader.find_product_type(),
                "product_name": self.xml_reader.find_product_name(),
                "platform": self.xml_reader.find_spacecraft_name(),
                "instrument": "SLSTR",
                "processing_level": "L1",
                "processing_version": self.xml_reader.find_processing_baseline(),
                "product_bounds": self.xml_reader.find_bounds().wkt,
                "product_datetime": acquisition_start_datetime,
                "image_size": f"{rows}x{cols}" if rows and cols else None,
                "eoio:reader": "sentinel3_slstr",
                "footprint": normalize_footprint(
                    geometry_input=self.xml_reader.find_bounds(),
                    crs_input=self._crs,
                ),
            }
            return basic
        except Exception:
            return {}

    def get_product_metadata(self) -> dict:
        if not self.xml_reader:
            return {}
        md = {}
        # Extract fields individually so that a missing optional field
        # (e.g. stop time) does not cause the whole extraction to fail.
        try:
            md["orbit_number"] = self.xml_reader.find_orbit_number()
        except Exception:
            md["orbit_number"] = None

        try:
            md["relative_orbit_number"] = self.xml_reader.find_relative_orbit_number()
        except Exception:
            md["relative_orbit_number"] = None

        try:
            md["orbit_direction"] = self.xml_reader.find_orbit_direction()
        except Exception:
            md["orbit_direction"] = None

        try:
            md["cycle_number"] = self.xml_reader.find_cycle_number()
        except Exception:
            md["cycle_number"] = None

        try:
            md["acquisition_start_time"] = self.xml_reader.find_acquisition_start_datetime()
        except Exception:
            md["acquisition_start_time"] = None

        try:
            md["acquisition_stop_time"] = self.xml_reader.find_acquisition_stop_datetime()
        except Exception:
            md["acquisition_stop_time"] = None

        try:
            md["measurement_accuracy"] = (
                self.xml_reader.find_value("measurement_accuracy") if hasattr(self.xml_reader, "find_value") else None
            )
        except Exception:
            md["measurement_accuracy"] = None

        md["crs_code"] = self._crs

        return md

    def get_variable_basic_metadata(self, var: str) -> dict:
        # derive basic units/description from variable name
        if "radiance" in var.lower():
            units = "W/m2/sr/nm"
            measurand = "radiance"
        elif "bt" in var.lower():
            units = "K"
            measurand = "brightness_temperature"
        else:
            units = ""
            measurand = ""

        return {
            "units": units,
            "long_name": var,
            "standard_name": var,
            "measurand": measurand,
        }

    def get_variable_product_metadata(self, var: str) -> dict:
        # Expect variable names starting with band token like 'S1', 'F1', etc.
        band_token = var.split("_")[0][:2]
        m = {"band_name": band_token}

        if band_token in self._central_wavelengths:
            m["band_central_wavelength"] = self._central_wavelengths[band_token]
            m["band_central_wavelength_units"] = "nm"

        if band_token in self._bandwidths:
            m["band_bandwidth"] = self._bandwidths[band_token]
            m["band_bandwidth_units"] = "nm"

        # spatial resolution may be inferred from layout when available
        try:
            grid = self.layout.get_grid(var)
            if grid:
                m["spatial_resolution"] = self.layout.get_grid_res(grid)
                m["spatial_resolution_units"] = "m"
        except Exception:
            pass

        return m


if __name__ == "__main__":
    pass
