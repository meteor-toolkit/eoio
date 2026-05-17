"""XML metadata reader for Sentinel-3 SLSTR product manifests.

Provides :class:`S3SLSTRXMLReader`, a thin semantic layer on top of
the generic :class:`eoio.readers.xml.XMLReader` exposing commonly used
fields from `xfdumanifest.xml` for SLSTR SEN3 products.
"""

from __future__ import annotations
import datetime as dt
from typing import Optional
from eoio.deps import lazy_shapely
from eoio.readers.xml import XMLReader


# Paths used to extract values from the SLSTR manifest
S3_SLSTR_MTD_METADATA_PATHS: dict[str, str] = {
    "acquisition_start_time": (
        ".//{http://www.esa.int/safe/sentinel/1.1}acquisitionPeriod/{http://www.esa.int/safe/sentinel/1.1}startTime"
    ),
    "acquisition_stop_time": (
        ".//{http://www.esa.int/safe/sentinel/1.1}acquisitionPeriod/{http://www.esa.int/safe/sentinel/1.1}stopTime"
    ),
    "product_name": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/1.0}generalProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}productName"
    ),
    "product_type": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/1.0}generalProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}productType"
    ),
    "processing_baseline": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/1.0}generalProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}processingBaseline"
    ),
    "spacecraft_family_name": (
        ".//{http://www.esa.int/safe/sentinel/1.1}platform/{http://www.esa.int/safe/sentinel/1.1}familyName"
    ),
    "spacecraft_number": (
        ".//{http://www.esa.int/safe/sentinel/1.1}platform/{http://www.esa.int/safe/sentinel/1.1}number"
    ),
    "orbit_number": (
        ".//{http://www.esa.int/safe/sentinel/1.1}orbitReference/"
        "{http://www.esa.int/safe/sentinel/1.1}orbitNumber[@type='start']"
    ),
    "relative_orbit_number": (
        ".//{http://www.esa.int/safe/sentinel/1.1}orbitReference/"
        "{http://www.esa.int/safe/sentinel/1.1}relativeOrbitNumber[@type='start']"
    ),
    "cycle_number": (
        ".//{http://www.esa.int/safe/sentinel/1.1}orbitReference/{http://www.esa.int/safe/sentinel/1.1}cycleNumber"
    ),
    "bounds": (
        ".//{http://www.esa.int/safe/sentinel/1.1}frameSet/"
        "{http://www.esa.int/safe/sentinel/1.1}footPrint/"
        "{http://www.opengis.net/gml}posList"
    ),
    "srs_name": (
        ".//{http://www.esa.int/safe/sentinel/1.1}frameSet/{http://www.esa.int/safe/sentinel/1.1}footPrint[@srsName]"
    ),
    # SLSTR-specific
    "measurement_accuracy": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}slstrProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}measurementAccuracy"
    ),
    # image size use nadir grid 1 km by default
    "nadir_image_rows": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}slstrProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}nadirImageSize[@grid='1 km']/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}rows"
    ),
    "nadir_image_columns": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}slstrProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}nadirImageSize[@grid='1 km']/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}columns"
    ),
    # band descriptions container
    "band_descriptions": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}slstrProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}bandDescriptions"
    ),
}


class S3SLSTRXMLReader(XMLReader):
    """Sentinel-3 SLSTR metadata reader for ``xfdumanifest.xml``."""

    metadata_paths = S3_SLSTR_MTD_METADATA_PATHS

    def find_acquisition_start_datetime(self) -> dt.datetime:
        date_str = self.find_value("acquisition_start_time")
        if date_str is None:
            raise ValueError("acquisition_start_time not found in metadata")
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))

    def find_acquisition_stop_datetime(self) -> dt.datetime:
        date_str = self.find_value("acquisition_stop_time")
        if date_str is None:
            raise ValueError("acquisition_stop_time not found in metadata")
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))

    def find_product_name(self) -> str:
        return self.find_value("product_name")

    def find_product_type(self) -> str:
        return self.find_value("product_type")

    def find_processing_baseline(self) -> str:
        return self.find_value("processing_baseline")

    def find_spacecraft_name(self) -> str:
        family = self.find_value("spacecraft_family_name", default="Sentinel-3")
        number = self.find_value("spacecraft_number", default="A")
        return f"{family}{number}"

    def find_orbit_number(self) -> Optional[int]:
        val = self.find_value("orbit_number")
        return int(val) if val is not None else None

    def find_relative_orbit_number(self) -> Optional[int]:
        val = self.find_value("relative_orbit_number")
        return int(val) if val is not None else None

    def find_orbit_direction(self) -> str:
        path = self.metadata_paths["orbit_number"]
        elem = self.xml_root.find(path, self.xml_ns)
        if elem is not None and "groundTrackDirection" in elem.attrib:
            return elem.attrib["groundTrackDirection"].upper()
        return "UNKNOWN"

    def find_cycle_number(self) -> Optional[int]:
        val = self.find_value("cycle_number")
        return int(val) if val is not None else None

    def find_bounds(self):
        Polygon, _, _, _, _ = lazy_shapely()

        coords_flat = self.find_value("bounds", as_array=True)
        if coords_flat is None or len(coords_flat) == 0:
            raise ValueError("Invalid posList in product metadata")

        try:
            coords_flat = [float(x) for x in coords_flat]
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid coordinate values in posList") from e

        if len(coords_flat) < 6:
            raise ValueError("Invalid posList format in product metadata")

        pairs_latlon = [(coords_flat[i], coords_flat[i + 1]) for i in range(0, len(coords_flat), 2)]
        coords_lonlat = [(lon, lat) for (lat, lon) in pairs_latlon]
        if coords_lonlat[0] != coords_lonlat[-1]:
            coords_lonlat.append(coords_lonlat[0])

        poly = Polygon(coords_lonlat)
        if not poly.is_valid:
            raise ValueError("Invalid footprint polygon constructed from metadata")
        return poly

    def find_crs_code(self) -> Optional[str]:
        srs = self.find_value("srs_name")
        if srs is None:
            path = self.metadata_paths.get("srs_name")
            if path is not None:
                elem = self.xml_root.find(path, self.xml_ns)
                if elem is not None and "srsName" in elem.attrib:
                    srs = elem.attrib.get("srsName")
        if srs is None:
            return None
        crs = srs.rstrip("/").split("/")[-1]
        return str("EPSG:" + crs)

    def find_image_size(self) -> dict[str, Optional[int]]:
        rows = self.find_value("nadir_image_rows")
        cols = self.find_value("nadir_image_columns")
        return {
            "rows": int(rows) if rows is not None else None,
            "columns": int(cols) if cols is not None else None,
        }

    # ------------------------------------------------------------------
    # Spectral metadata
    # ------------------------------------------------------------------

    def find_band_central_wavelength(self, band_name: str) -> Optional[float]:
        path = (
            ".//{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}slstrProductInformation/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}bandDescriptions/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}band"
            f"[@name='{band_name}']/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}centralWavelength"
        )
        elem = self.xml_root.find(path, self.xml_ns)
        if elem is not None and elem.text:
            return float(elem.text)
        return None

    def find_band_bandwidth(self, band_name: str) -> Optional[float]:
        path = (
            ".//{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}slstrProductInformation/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/slstr/1.0}bandDescriptions/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}band"
            f"[@name='{band_name}']/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}bandwidth"
        )
        elem = self.xml_root.find(path, self.xml_ns)
        if elem is not None and elem.text:
            return float(elem.text)
        return None

    def find_all_band_central_wavelengths(self) -> dict[str, float]:
        out: dict[str, float] = {}
        # find band elements under the bandDescriptions container
        container_path = self.metadata_paths.get("band_descriptions")
        if container_path is None:
            return out
        container = self.xml_root.find(container_path, self.xml_ns)
        if container is None:
            return out
        for band_elem in list(container):
            name = band_elem.attrib.get("name")
            if not name:
                continue
            cw = None
            for child in list(band_elem):
                if child.tag.rsplit("}", 1)[-1] == "centralWavelength":
                    if child.text:
                        try:
                            cw = float(child.text)
                        except Exception:
                            cw = None
                    break
            if cw is not None:
                out[name] = cw
        return out

    def find_all_band_bandwidths(self) -> dict[str, float]:
        out: dict[str, float] = {}
        container_path = self.metadata_paths.get("band_descriptions")
        if container_path is None:
            return out
        container = self.xml_root.find(container_path, self.xml_ns)
        if container is None:
            return out
        for band_elem in list(container):
            name = band_elem.attrib.get("name")
            if not name:
                continue
            bw = None
            for child in list(band_elem):
                if child.tag.rsplit("}", 1)[-1] == "bandwidth":
                    if child.text:
                        try:
                            bw = float(child.text)
                        except Exception:
                            bw = None
                    break
            if bw is not None:
                out[name] = bw
        return out


if __name__ == "__main__":
    pass
