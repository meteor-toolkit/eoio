"""eoio.readers.sentinel2.metadata.s2_prod_mtd - reader for Sentinel-2 product metadata."""

import datetime as dt
from typing import Optional
from eoio.deps import lazy_shapely
from eoio.readers.xml import XMLReader
from eoio.readers.sentinel2.metadata.var_names import BAND_ID_MEAS_VARS

# Location of fields of interest within the product metadata file
S2_PROD_MTD_METADATA_PATHS: dict[str, str] = {
    # ------------------------------------------------------------------
    # Temporal metadata
    # ------------------------------------------------------------------
    "product_start_time": "./n1:General_Info/Product_Info/PRODUCT_START_TIME",
    "product_stop_time": "./n1:General_Info/Product_Info/PRODUCT_STOP_TIME",
    "generation_time": "./n1:General_Info/Product_Info/GENERATION_TIME",
    # ------------------------------------------------------------------
    # Identification / provenance
    # ------------------------------------------------------------------
    "product_uri": "./n1:General_Info/Product_Info/PRODUCT_URI",
    "processing_level": "./n1:General_Info/Product_Info/PROCESSING_LEVEL",
    "processing_baseline": "./n1:General_Info/Product_Info/PROCESSING_BASELINE",
    "product_type": "./n1:General_Info/Product_Info/PRODUCT_TYPE",
    "product_doi": "./n1:General_Info/Product_Info/PRODUCT_DOI",
    # ------------------------------------------------------------------
    # Datatake / orbit
    # ------------------------------------------------------------------
    "spacecraft_name": "./n1:General_Info/Product_Info/Datatake/SPACECRAFT_NAME",
    "datatake_type": "./n1:General_Info/Product_Info/Datatake/DATATAKE_TYPE",
    "sensing_orbit_number": "./n1:General_Info/Product_Info/Datatake/SENSING_ORBIT_NUMBER",
    "sensing_orbit_direction": "./n1:General_Info/Product_Info/Datatake/SENSING_ORBIT_DIRECTION",
    # ------------------------------------------------------------------
    # Geometry / footprint
    # ------------------------------------------------------------------
    "bounds": "./n1:Geometric_Info/Product_Footprint/Product_Footprint/Global_Footprint/EXT_POS_LIST",
    # ------------------------------------------------------------------
    # Radiometric metadata
    # ------------------------------------------------------------------
    "quantification_value": "./n1:General_Info/Product_Image_Characteristics/QUANTIFICATION_VALUE",
    "boa_quantification_value": "./n1:General_Info/"
    "Product_Image_Characteristics/QUANTIFICATION_VALUES_LIST/"
    "BOA_QUANTIFICATION_VALUE",
    "aot_quantification_value": "./n1:General_Info/"
    "Product_Image_Characteristics/QUANTIFICATION_VALUES_LIST/"
    "AOT_QUANTIFICATION_VALUE",
    "wvp_quantification_value": "./n1:General_Info/"
    "Product_Image_Characteristics/QUANTIFICATION_VALUES_LIST/"
    "WVP_QUANTIFICATION_VALUE",
    "radio_add_offset": "./n1:General_Info/Product_Image_Characteristics/Radiometric_Offset_List",
    "boa_add_offset": "./n1:General_Info/Product_Image_Characteristics/BOA_ADD_OFFSET_VALUES_LIST",
    "physical_gains": "./n1:General_Info/Product_Image_Characteristics",
    "reflectance_conversion_u": ("./n1:General_Info/Product_Image_Characteristics/Reflectance_Conversion/U"),
    "solar_irradiance": "./n1:General_Info/Product_Image_Characteristics/Reflectance_Conversion/Solar_Irradiance_List",
    "reference_band": "./n1:General_Info/Product_Image_Characteristics/REFERENCE_BAND",
    # ------------------------------------------------------------------
    # Spectral metadata (band-specific)
    # ------------------------------------------------------------------
    # NOTE: these are intended to be formatted with bandId, e.g.
    # spectral_response_band_3, spectral_resolution_band_3, etc.
    **{
        f"spectral_response_band_{band_id}": "./n1:General_Info/"
        "Product_Image_Characteristics/"
        "Spectral_Information_List/"
        f"Spectral_Information[@bandId='{band_id}']/"
        "Spectral_Response/VALUES"
        for band_id in range(13)
    },
    **{
        f"spectral_resolution_band_{band_id}": "./n1:General_Info/"
        "Product_Image_Characteristics/"
        "Spectral_Information_List/"
        f"Spectral_Information[@bandId='{band_id}']/"
        "RESOLUTION"
        for band_id in range(13)
    },
    **{
        f"spectral_central_wavelength_band_{band_id}": "./n1:General_Info/"
        "Product_Image_Characteristics/"
        "Spectral_Information_List/"
        f"Spectral_Information[@bandId='{band_id}']/"
        "Wavelength/CENTRAL"
        for band_id in range(13)
    },
}


class S2ProdXMLReader(XMLReader):
    """
    Sentinel-2 Level-1C product metadata reader.

    This class provides semantic accessors for commonly used fields in the
    Sentinel-2 product metadata XML (MTD_MSIL1C.xml). It builds on the generic
    :class:`XMLReader` to return domain-appropriate Python objects.
    """

    metadata_paths = S2_PROD_MTD_METADATA_PATHS

    # ------------------------------------------------------------------
    # Temporal metadata
    # ------------------------------------------------------------------

    def find_product_start_datetime(self) -> dt.datetime:
        """
        Return the product sensing start time as a timezone-aware datetime.
        """
        date_str = self.find_value("product_start_time")
        return dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)

    def find_product_start_date(self) -> dt.date:
        """
        Return the product sensing start date.
        """
        return self.find_product_start_datetime().date()

    # ------------------------------------------------------------------
    # Identification / provenance
    # ------------------------------------------------------------------

    def find_product_type(self) -> str:
        """
        Return the product type
        """
        return self.find_value("product_type")

    def find_product_uri(self) -> str:
        """
        Return the SAFE product identifier (PRODUCT_URI).
        """
        return self.find_value("product_uri")

    def find_processing_level(self) -> str:
        """
        Return the processing level (e.g. ``Level-1C``).
        """
        return self.find_value("processing_level")

    def find_processing_baseline(self) -> str:
        """
        Return the processing baseline identifier.
        """
        return self.find_value("processing_baseline")

    def find_spacecraft_name(self) -> str:
        """
        Return the spacecraft name (e.g. ``Sentinel-2A``).
        """
        return self.find_value("spacecraft_name")

    def find_orbit_number(self) -> int:
        """
        Return the sensing orbit number.
        """
        return self.find_value("sensing_orbit_number")

    def find_orbit_direction(self) -> str:
        """
        Return the orbit direction (ASCENDING or DESCENDING).
        """
        return self.find_value("sensing_orbit_direction")

    # ------------------------------------------------------------------
    # Geometry / footprint
    # ------------------------------------------------------------------

    def find_bounds(self):
        """
        Return the product footprint as a Shapely polygon in lon/lat order.

        The polygon is constructed from the ``EXT_POS_LIST`` element, which
        encodes latitude/longitude coordinate pairs.
        """
        Polygon, _, _, _, _ = lazy_shapely()

        latlon = self.find_value("bounds")
        if latlon is None or len(latlon) < 6:
            raise ValueError("Invalid EXT_POS_LIST in product metadata")

        pairs_latlon = [(latlon[i], latlon[i + 1]) for i in range(0, len(latlon), 2)]
        coords_lonlat = [(lon, lat) for (lat, lon) in pairs_latlon]

        # Ensure closure
        if coords_lonlat[0] != coords_lonlat[-1]:
            coords_lonlat.append(coords_lonlat[0])

        poly = Polygon(coords_lonlat)

        if not poly.is_valid:
            raise ValueError("Invalid footprint polygon constructed from metadata")

        return poly

    # ------------------------------------------------------------------
    # Radiometric metadata
    # ------------------------------------------------------------------

    def find_quantification_values(self) -> dict[str, float | None]:
        """
        Return quantification values for reflectance and L2A auxiliary layers.

        For L1C products, only ``reflectance`` is expected.
        For L2A products, ``reflectance`` corresponds to
        ``BOA_QUANTIFICATION_VALUE`` and may be accompanied by
        ``AOT_QUANTIFICATION_VALUE`` and ``WVP_QUANTIFICATION_VALUE``.

        :returns:
            Dictionary with keys:
                - ``"reflectance"``
                - ``"aot"``
                - ``"wvp"``
            Missing values are returned as ``None``.
        """
        reflectance = self.find_value("boa_quantification_value")
        if reflectance is None:
            reflectance = self.find_value("quantification_value")

        return {
            "reflectance": reflectance,
            "aot": self.find_value("aot_quantification_value"),
            "wvp": self.find_value("wvp_quantification_value"),
        }

    def find_reflectance_offsets(self) -> dict[str, int]:
        """
        Return reflectance additive offsets keyed by Sentinel-2 band token.

        For L2A products, offsets are read from ``BOA_ADD_OFFSET``.
        For L1C products, offsets are read from ``RADIO_ADD_OFFSET``.

        :returns:
            Dictionary keyed by band token, for example ``"B02"``.
        """
        result = self.find_mapping(
            "boa_add_offset",
            key="BOA_ADD_OFFSET",
            key_attr="band_id",
            key_cast=int,
            value_cast=int,
        )

        if not result:
            result = self.find_mapping(
                "radio_add_offset",
                key="RADIO_ADD_OFFSET",
                key_attr="band_id",
                key_cast=int,
                value_cast=int,
            )

        return {
            BAND_ID_MEAS_VARS[band_id]: offset
            for band_id, offset in (result or {}).items()
            if band_id in BAND_ID_MEAS_VARS
        }

    def find_physical_gains(self) -> Optional[dict[int, float]]:
        """
        Return physical gains per band.
        """
        result = self.find_mapping(
            "physical_gains",
            key="PHYSICAL_GAINS",
            key_attr="bandId",
            key_cast=int,
            value_cast=float,
        )

        return result

    def find_reflectance_conversion_u(self) -> float:
        """
        Return the reflectance conversion factor U.
        """
        return self.find_value("reflectance_conversion_u")

    def find_solar_irradiance(self) -> Optional[dict[int, float]]:
        """
        Return solar irradiance values per band (W/m²/µm).
        """
        result = self.find_mapping(
            "solar_irradiance",
            key="SOLAR_IRRADIANCE",
            key_attr="bandId",
            key_cast=int,
            value_cast=float,
        )

        return result

    def find_solar_irradiance_unit(self) -> str:
        """
        Return the unit of the solar irradiance values.

        This is read from the ``unit`` attribute of the first
        ``SOLAR_IRRADIANCE`` element and is assumed to be consistent
        across all bands (as per Sentinel-2 product specification).

        :returns:
            Solar irradiance unit string (e.g. ``"W/m²/µm"``).
        :raises ValueError:
            If no solar irradiance elements are found or the unit attribute
            is missing.
        """
        container = self.xml_root.find(self.metadata_paths["solar_irradiance"], self.xml_ns)

        def local(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        first = next(e for e in list(container or []) if local(e.tag) == "SOLAR_IRRADIANCE")

        return first.attrib["unit"]

    # ------------------------------------------------------------------
    # Spectral metadata
    # ------------------------------------------------------------------

    def find_spectral_response(self, band_id: int) -> list[float]:
        """
        Return the spectral response function for a given band.

        :param band_id:
            Sentinel-2 band index (0–12).
        """
        key = f"spectral_response_band_{band_id}"
        return self.find_value(key, deep_text=True, as_array=True)

    def find_band_resolution(self, band_id: int) -> int:
        """
        Return the spatial resolution (m) for a given band.
        """
        key = f"spectral_resolution_band_{band_id}"
        return self.find_value(key)

    def find_band_central_wavelength(self, band_id: int) -> float:
        """
        Return the central wavelength (nm) for a given band.
        """
        key = f"spectral_central_wavelength_band_{band_id}"
        return self.find_value(key)

    # ------------------------------------------------------------------
    # Convenience aggregations
    # ------------------------------------------------------------------

    def find_all_band_resolutions(self) -> dict[int, int]:
        """
        Return a mapping of band index to spatial resolution.
        """
        return {band_id: self.find_band_resolution(band_id) for band_id in range(13)}


if __name__ == "__main__":
    pass
