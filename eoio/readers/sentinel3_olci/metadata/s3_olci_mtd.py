"""XML metadata reader for Sentinel-3 OLCI product manifests.

This module implements :class:`S3OLCIXMLReader`, a thin semantic layer on
top of the generic :class:`eoio.readers.xml.XMLReader` that exposes
commonly-used fields from the `xfdumanifest.xml` file in a convenient
Pythonic form.

The class provides helpers to parse acquisition times, product
identifiers, orbit information, the footprint polygon and band spectral
characteristics (central wavelength and bandwidth). Calling code can use
these accessors to populate higher-level metadata structures without
dealing with XML XPath expressions directly.
"""

import datetime as dt
from typing import Optional
from eoio.deps import lazy_shapely
from eoio.readers.xml import XMLReader

# Location of fields of interest within the xfdumanifest.xml file
S3_OLCI_MTD_METADATA_PATHS: dict[str, str] = {
    # ------------------------------------------------------------------
    # Temporal metadata
    # ------------------------------------------------------------------
    "acquisition_start_time": (
        ".//{http://www.esa.int/safe/sentinel/1.1}acquisitionPeriod/{http://www.esa.int/safe/sentinel/1.1}startTime"
    ),
    "acquisition_stop_time": (
        ".//{http://www.esa.int/safe/sentinel/1.1}acquisitionPeriod/{http://www.esa.int/safe/sentinel/1.1}stopTime"
    ),
    # ------------------------------------------------------------------
    # Identification / provenance
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # Platform information
    # ------------------------------------------------------------------
    "spacecraft_family_name": (
        ".//{http://www.esa.int/safe/sentinel/1.1}platform/{http://www.esa.int/safe/sentinel/1.1}familyName"
    ),
    "spacecraft_number": (
        ".//{http://www.esa.int/safe/sentinel/1.1}platform/{http://www.esa.int/safe/sentinel/1.1}number"
    ),
    # ------------------------------------------------------------------
    # Orbit information
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # Geometry / footprint
    # ------------------------------------------------------------------
    "bounds": (
        ".//{http://www.esa.int/safe/sentinel/1.1}frameSet/"
        "{http://www.esa.int/safe/sentinel/1.1}footPrint/"
        "{http://www.opengis.net/gml}posList"
    ),
    "srs_name": (
        ".//{http://www.esa.int/safe/sentinel/1.1}frameSet/{http://www.esa.int/safe/sentinel/1.1}footPrint[@srsName]"
    ),
    # ------------------------------------------------------------------
    # OLCI-specific metadata
    # ------------------------------------------------------------------
    "measurement_accuracy": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}olciProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}measurementAccuracy/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}relativeMeasurementAccuracy"
    ),
    "image_rows": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}olciProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}imageSize/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}rows"
    ),
    "image_columns": (
        ".//{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}olciProductInformation/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}imageSize/"
        "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}columns"
    ),
}


class S3OLCIXMLReader(XMLReader):
    """
    Sentinel-3 OLCI metadata reader for ``xfdumanifest.xml``.

    This class provides semantic accessors for commonly used fields in the
    Sentinel-3 OLCI product metadata XML. It builds on the generic
    :class:`XMLReader` to return domain-appropriate Python objects.
    """

    metadata_paths = S3_OLCI_MTD_METADATA_PATHS

    # ------------------------------------------------------------------
    # Temporal metadata
    # ------------------------------------------------------------------

    def find_acquisition_start_datetime(self) -> dt.datetime:
        """
        Return the acquisition start time as a timezone-aware datetime.
        """
        date_str = self.find_value("acquisition_start_time")
        if date_str is None:
            raise ValueError("acquisition_start_time not found in metadata")
        try:
            return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid datetime format: {date_str}") from e

    def find_acquisition_stop_datetime(self) -> dt.datetime:
        """
        Return the acquisition stop time as a timezone-aware datetime.
        """
        date_str = self.find_value("acquisition_stop_time")
        if date_str is None:
            raise ValueError("acquisition_stop_time not found in metadata")
        try:
            return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid datetime format: {date_str}") from e

    def find_acquisition_start_date(self) -> dt.date:
        """
        Return the acquisition start date.
        """
        return self.find_acquisition_start_datetime().date()

    # ------------------------------------------------------------------
    # Identification / provenance
    # ------------------------------------------------------------------

    def find_product_name(self) -> str:
        """
        Return the product name.
        """
        return self.find_value("product_name")

    def find_product_type(self) -> str:
        """
        Return the product type (e.g., OL_1_EFR___)
        """
        return self.find_value("product_type")

    def find_processing_baseline(self) -> str:
        """
        Return the processing baseline identifier.
        """
        return self.find_value("processing_baseline")

    def find_spacecraft_name(self) -> str:
        """
        Return the spacecraft name (e.g., Sentinel-3A)
        """
        family = self.find_value("spacecraft_family_name", default="Sentinel-3")
        number = self.find_value("spacecraft_number", default="A")
        return f"{family}{number}"

    # ------------------------------------------------------------------
    # Orbit information
    # ------------------------------------------------------------------

    def find_orbit_number(self) -> Optional[int]:
        """
        Return the orbit number.
        """
        val = self.find_value("orbit_number")
        return int(val) if val is not None else None

    def find_relative_orbit_number(self) -> Optional[int]:
        """
        Return the relative orbit number.
        """
        val = self.find_value("relative_orbit_number")
        return int(val) if val is not None else None

    def find_orbit_direction(self) -> str:
        """
        Return the orbit direction by extracting the groundTrackDirection attribute.
        """
        # The attribute is in the XPath, so we need to extract it from the element
        path = self.metadata_paths["orbit_number"]
        elem = self.xml_root.find(path, self.xml_ns)
        if elem is not None and "groundTrackDirection" in elem.attrib:
            return elem.attrib["groundTrackDirection"].upper()
        return "UNKNOWN"

    def find_cycle_number(self) -> Optional[int]:
        """
        Return the cycle number.
        """
        val = self.find_value("cycle_number")
        return int(val) if val is not None else None

    # ------------------------------------------------------------------
    # Geometry / footprint
    # ------------------------------------------------------------------

    def find_bounds(self):
        """
        Return the product footprint as a Shapely polygon in lon/lat order.

        The polygon is constructed from the posList element, which encodes
        latitude/longitude coordinate pairs.
        """
        Polygon, _, _, _, _ = lazy_shapely()

        coords_flat = self.find_value("bounds", as_array=True)
        if coords_flat is None or len(coords_flat) == 0:
            raise ValueError("Invalid posList in product metadata")

        # Ensure all values are floats
        try:
            coords_flat = [float(x) for x in coords_flat]
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid coordinate values in posList") from e

        if len(coords_flat) < 6:
            raise ValueError("Invalid posList format in product metadata")

        # Convert to (lat, lon) pairs
        pairs_latlon = [(coords_flat[i], coords_flat[i + 1]) for i in range(0, len(coords_flat), 2)]
        # Convert to (lon, lat) for Polygon
        coords_lonlat = [(lon, lat) for (lat, lon) in pairs_latlon]

        # Ensure closure
        if coords_lonlat[0] != coords_lonlat[-1]:
            coords_lonlat.append(coords_lonlat[0])

        poly = Polygon(coords_lonlat)

        if not poly.is_valid:
            raise ValueError("Invalid footprint polygon constructed from metadata")

        return poly

    def find_crs_code(self) -> Optional[str]:
        """
        Return the CRS code (e.g., "EPSG:4326") for the footprint coordinates.
        """
        # Try to get the value via the generic finder (element text).
        srs = self.find_value("srs_name")

        # If the XPath selects an element with the srsName as an attribute
        # (common in SAFE manifests), fall back to extracting the attribute
        # from the matched element.
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

    # ------------------------------------------------------------------
    # OLCI-specific metadata
    # ------------------------------------------------------------------

    def find_image_size(self) -> dict[str, Optional[int]]:
        """
        Return the image size (rows and columns).
        """
        rows = self.find_value("image_rows")
        columns = self.find_value("image_columns")

        return {
            "rows": int(rows) if rows is not None else None,
            "columns": int(columns) if columns is not None else None,
        }

    def find_measurement_accuracy(self) -> Optional[float]:
        """
        Return the measurement accuracy value.
        """
        val = self.find_value("measurement_accuracy")
        return float(val) if val is not None else None

    # ------------------------------------------------------------------
    # Spectral metadata (band-specific)
    # ------------------------------------------------------------------

    def find_band_central_wavelength(self, band_id: int) -> Optional[float]:
        """
        Return the central wavelength (nm) for a given OLCI band.

        :param band_id:
            OLCI band index (1–21).
        """
        path = (
            ".//{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}bandDescriptions/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}band"
            f"[@name='Oa{band_id:02d}']/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}centralWavelength"
        )

        elem = self.xml_root.find(path, self.xml_ns)
        if elem is not None and elem.text:
            return float(elem.text)
        return None

    def find_band_bandwidth(self, band_id: int) -> Optional[float]:
        """
        Return the bandwidth (nm) for a given OLCI band.

        :param band_id:
            OLCI band index (1–21).
        """
        path = (
            ".//{http://www.esa.int/safe/sentinel/sentinel-3/olci/1.0}bandDescriptions/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}band"
            f"[@name='Oa{band_id:02d}']/"
            "{http://www.esa.int/safe/sentinel/sentinel-3/1.0}bandwidth"
        )

        elem = self.xml_root.find(path, self.xml_ns)
        if elem is not None and elem.text:
            return float(elem.text)
        return None

    def find_all_band_central_wavelengths(self) -> dict[int, float]:
        """
        Return a mapping of band index to central wavelength.
        """
        wavelengths = {}
        for band_id in range(1, 22):
            wl = self.find_band_central_wavelength(band_id)
            if wl is not None:
                wavelengths[band_id] = wl
        return wavelengths

    def find_all_band_bandwidths(self) -> dict[int, float]:
        """
        Return a mapping of band index to bandwidth.
        """
        bandwidths = {}
        for band_id in range(1, 22):
            bw = self.find_band_bandwidth(band_id)
            if bw is not None:
                bandwidths[band_id] = bw
        return bandwidths


if __name__ == "__main__":
    pass
