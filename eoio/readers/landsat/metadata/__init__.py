"""
eoio.readers.landsat.metadata
==============================

Landsat metadata readers and extractors for Collection 2 Level-1 data.

This module provides two main classes for working with Landsat metadata:

1. LSL1ProdXMLReader: Low-level XML metadata reader
   - Reads Landsat Collection 2 Level-1 MTL XML files
   - Provides semantic accessors for temporal, radiometric, projection, and band metadata
   - Handles graceful degradation when optional sections are missing

2. LSL1ProdJSONReader: Low-level JSON metadata reader
   - Reads Landsat Collection 2 Level-1 MTL JSON files
   - Provides semantic accessors for temporal, radiometric, projection, and band metadata
   - Handles graceful degradation when optional sections are missing

3. LSMetadataExtractor: High-level metadata extractor
   - Transforms low-level metadata into structured datasets
   - Integrates with the xarray data model
   - Matches the Sentinel-2 reader architecture pattern

Example usage:

    from eoio.readers.landsat.metadata import LSL1ProdXMLReader, LSMetadataExtractor
    from pathlib import Path

    # Load and parse metadata
    mtl_file = Path("LC08_L1TP_040033_20200306_20200822_02_T1_MTL.xml")
    reader = LSL1ProdXMLReader(mtl_file)

    # Access basic metadata
    print(reader.find_product_id())
    print(reader.find_acquisition_datetime())

    # Get radiometric calibration coefficients
    rescaling = reader.get_radiometric_rescaling()
    thermal_constants = reader.get_thermal_constants()

    # Get bounds geometry
    bounds_polygon = reader.find_bounds()

"""

from eoio.readers.landsat.metadata.ls_mtd_xml import LSL1ProdXMLReader
from eoio.readers.landsat.metadata.ls_mtd_json import LSL1ProdJSONReader
from eoio.readers.landsat.metadata.extractor import LSMetadataExtractor

__all__ = [
    "LSL1ProdXMLReader",
    "LSL1ProdJSONReader",
    "LSMetadataExtractor",
]
