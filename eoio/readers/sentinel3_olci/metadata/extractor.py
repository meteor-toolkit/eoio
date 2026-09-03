"""High-level metadata extractor for Sentinel-3 OLCI products.

This module provides the ``S3OLCIMetadataExtractor`` adapter which
wraps lower-level XML parsing provided by
``eoio.readers.sentinel3_olci.metadata.s3_olci_mtd.S3OLCIXMLReader``.

The extractor exposes a small, stable interface used by the reader code
to obtain product-wide metadata (``get_basic_metadata``,
``get_product_metadata``) and variable-specific metadata
(``get_variable_basic_metadata``, ``get_variable_product_metadata``).

Typical usage::

    reader = OLCIL1Reader(path)
    mtd = S3OLCIMetadataExtractor(reader)
    basic = mtd.get_basic_metadata()
    band_mtd = mtd.get_variable_product_metadata('Oa02')

The implementation caches spectral information (central wavelengths and
bandwidths) when available in the product manifest to avoid repeated XML
parsing.
"""

from __future__ import annotations
import json
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.sentinel3_olci.metadata.s3_olci_mtd import S3OLCIXMLReader
from eoio.readers.sentinel3_olci.metadata.var_names import AUX_OPTIONS, MASK_OPTIONS
from eoio.readers.footprint_utils import normalize_footprint

# Mapping from OLCI band names to band indices
OLCI_BAND_IDS = {f"Oa{i:02d}": i for i in range(1, 22)}


class S3OLCIMetadataExtractor(BaseMetadataExtractor):
    """
    High-level metadata extractor for Sentinel-3 OLCI products.

    This class orchestrates metadata extraction from Sentinel-3 OLCI product-
    level metadata (xfdumanifest.xml), and exposes the results as structured
    Python dictionaries suitable for downstream use (e.g., MetEOR tools).

    It delegates low-level XML parsing to :class:`S3OLCIXMLReader`.
    """

    def __init__(self, reader):
        """
        Initialise the Sentinel-3 OLCI metadata extractor.

        :param reader:
            The EOIO Sentinel-3 OLCI reader instance providing layout and configuration.
        """
        super().__init__(reader)

        # Define attributes
        self.subset = reader.resolved_config.subset if hasattr(reader, "resolved_config") else None
        self.layout = reader.layout

        # Initialise XML file reader for xfdumanifest.xml
        try:
            manifest_path = self.layout.manifest_path()
        except (AttributeError, TypeError):
            manifest_path = None

        if manifest_path:
            self.xml_reader = S3OLCIXMLReader(manifest_path)
        else:
            self.xml_reader = None

        # Cache metadata for efficiency
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
        """
        Return high-level, product-wide metadata.

        This metadata is intended for discovery, cataloguing, and general
        product description, and is shared across all variables in the dataset.

        :returns:
            Dictionary containing basic Sentinel-3 OLCI product metadata.
        """

        if not self.xml_reader:
            return {}

        try:
            image_size = self.xml_reader.find_image_size()
            rows = image_size.get("rows", 0)
            cols = image_size.get("columns", 0)

            basic_md = {
                "collection_name": self.xml_reader.find_product_type(),
                "product_name": self.xml_reader.find_product_name(),
                "platform": self.xml_reader.find_spacecraft_name(),
                "instrument": "OLCI",
                "processing_level": "L1",
                "processing_version": self.xml_reader.find_processing_baseline(),
                "spatial_resolution": 300,  # OLCI full resolution
                "spatial_resolution_units": "m",
                "geometry_ids": "300m",
                "product_bounds": self.xml_reader.find_bounds().wkt,
                "product_datetime": self.xml_reader.find_acquisition_start_datetime(),
                "description": "TBD",
                "image_size": f"{rows}x{cols}" if rows and cols else None,
                "institution": "European Space Agency (ESA)",
                "keywords": [
                    "Sentinel-3",
                    "OLCI",
                    "satellite",
                    "remote sensing",
                    "multispectral",
                    "earth observation",
                    "level 1",
                    "visible",
                ],
                "source": "Sentinel-3 OLCI Level 1 Satellite Product",
                "platform_type": "satellite",
                "footprint": normalize_footprint(
                    geometry_input=self.xml_reader.find_bounds(),
                    crs_input=self._crs,
                ),
            }
            if self.subset:
                # json.dumps keeps this parseable JSON and netCDF-attribute-safe (a raw
                # dict isn't), matching the eoio:subset convention used elsewhere.
                basic_md["eoio:subset"] = json.dumps(self.subset, default=str)
            basic_md["eoio:reader"] = "sentinel3_olci"

            return basic_md
        except Exception as e:
            print(f"Error extracting basic metadata: {e}")
            return {}

    def get_product_metadata(self) -> dict:
        """
        Return detailed product-level metadata.

        This metadata captures acquisition context, orbit information,
        and other product-specific identifiers.

        :returns:
            Dictionary containing Sentinel-3 OLCI product metadata.
        """

        if not self.xml_reader:
            return {}

        try:
            prod_md = {
                "orbit_number": self.xml_reader.find_orbit_number(),
                "relative_orbit_number": self.xml_reader.find_relative_orbit_number(),
                "orbit_direction": self.xml_reader.find_orbit_direction(),
                "cycle_number": self.xml_reader.find_cycle_number(),
                "acquisition_start_time": self.xml_reader.find_acquisition_start_datetime(),
                "acquisition_stop_time": self.xml_reader.find_acquisition_stop_datetime(),
                "measurement_accuracy": self.xml_reader.find_measurement_accuracy(),
                "crs_code": self.xml_reader.find_crs_code(),
            }
            return prod_md
        except Exception as e:
            print(f"Error extracting product metadata: {e}")
            return {}

    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Return basic descriptive metadata for a dataset variable.

        This method is intended for lightweight metadata such as units,
        names, or CF-style descriptors.

        :param var:
            Variable name for which metadata should be extracted.
        :returns:
            Dictionary containing basic variable metadata.
        """
        # Check if this is an OLCI band variable
        if var in OLCI_BAND_IDS:
            band_id = OLCI_BAND_IDS[var]
            return {
                "units": "W/m²/sr/nm",
                "description": f"Band Oa{band_id:02d} radiance",
                "long_name": f"Top of atmosphere radiance in band {band_id}",
                "standard_name": "toa_radiance",
                "measurand": "radiance",
            }

        return {"units": "", "long_name": "", "standard_name": ""}

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Return detailed metadata for a specific Sentinel-3 OLCI band variable.

        This includes spectral characteristics and band-specific information.

        :param var:
            Variable name corresponding to an OLCI band (e.g., ``"Oa02"``).
        :returns:
            Dictionary containing variable-specific metadata.
        :raises KeyError:
            If ``var`` is not a recognised OLCI band.
        """

        if var not in OLCI_BAND_IDS:
            if var in AUX_OPTIONS or var in MASK_OPTIONS:
                return {}
            elif var in self.reader.uncertainty_vars:
                return {}
            else:
                raise KeyError(f"Unknown OLCI variable: {var}")

        band_id = OLCI_BAND_IDS[var]

        variable_mtd = {
            "band_id": band_id,
            "band_name": var,
            "spatial_resolution": 300,
            "spatial_resolution_units": "m",
            "geometry_id": "300m",
            "ancillary_variables": "",
        }

        # Add spectral information if available
        if band_id in self._central_wavelengths:
            variable_mtd["band_central_wavelength"] = self._central_wavelengths[band_id]
            variable_mtd["band_central_wavelength_units"] = "nm"

        if band_id in self._bandwidths:
            variable_mtd["band_bandwidth"] = self._bandwidths[band_id]
            variable_mtd["band_bandwidth_units"] = "nm"

        return variable_mtd


def get_aux_product_metadata(var: str) -> dict:
    """
    Return metadata for auxiliary variables.

    :param var:
        Variable name corresponding to an auxiliary variable.
    """
    return {}


if __name__ == "__main__":
    pass
