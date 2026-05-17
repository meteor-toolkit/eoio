"""
eoio.readers.landsat.metadata.ls_prod_mtd - reader for Landsat product metadata (MTL XML).

This module provides a Landsat product metadata reader analogous to the Sentinel-2
S2ProdXMLReader. It defines a single LSProdXMLReader class built on the generic XMLReader
and a LS_PROD_MTD_METADATA_PATHS mapping of commonly used fields.

It targets Collection 2 Level-1 MTL XML files structured as in the attached example:
<LANDSAT_METADATA_FILE>
  <PRODUCT_CONTENTS>…</PRODUCT_CONTENTS>
  <IMAGE_ATTRIBUTES>…</IMAGE_ATTRIBUTES>
  <PROJECTION_ATTRIBUTES>…</PROJECTION_ATTRIBUTES>
  <LEVEL1_PROCESSING_RECORD>…</LEVEL1_PROCESSING_RECORD>
  <LEVEL1_MIN_MAX_RADIANCE>…</LEVEL1_MIN_MAX_RADIANCE>
  <LEVEL1_MIN_MAX_REFLECTANCE>…</LEVEL1_MIN_MAX_REFLECTANCE>
  <LEVEL1_MIN_MAX_PIXEL_VALUE>…</LEVEL1_MIN_MAX_PIXEL_VALUE>
  <LEVEL1_RADIOMETRIC_RESCALING>…</LEVEL1_RADIOMETRIC_RESCALING>
  <LEVEL1_THERMAL_CONSTANTS>…</LEVEL1_THERMAL_CONSTANTS>
  <LEVEL1_PROJECTION_PARAMETERS>…</LEVEL1_PROJECTION_PARAMETERS>
</LANDSAT_METADATA_FILE>
"""

from __future__ import annotations

import datetime as dt
from eoio.deps import lazy_shapely
from eoio.readers.xml import XMLReader

# Location of fields of interest within the Landsat MTL product metadata file
# XPaths are based on the attached XML structure (Collection 2 L1).
LS_PROD_MTD_METADATA_PATHS: dict[str, str] = {
    # --- Identification / provenance ---
    "landsat_product_id": ".//PRODUCT_CONTENTS/LANDSAT_PRODUCT_ID",
    "processing_level": ".//PRODUCT_CONTENTS/PROCESSING_LEVEL",
    "collection_number": ".//PRODUCT_CONTENTS/COLLECTION_NUMBER",
    "collection_category": ".//PRODUCT_CONTENTS/COLLECTION_CATEGORY",
    "output_format": ".//PRODUCT_CONTENTS/OUTPUT_FORMAT",
    # --- Temporal metadata ---
    # Landsat L1 exposes DATE_ACQUIRED and SCENE_CENTER_TIME (UTC, trailing Z).
    "date_acquired": ".//IMAGE_ATTRIBUTES/DATE_ACQUIRED",
    "scene_center_time": ".//IMAGE_ATTRIBUTES/SCENE_CENTER_TIME",
    # --- Image attributes ---
    "spacecraft_id": ".//IMAGE_ATTRIBUTES/SPACECRAFT_ID",
    "sensor_id": ".//IMAGE_ATTRIBUTES/SENSOR_ID",
    "wrs_path": ".//IMAGE_ATTRIBUTES/WRS_PATH",
    "wrs_row": ".//IMAGE_ATTRIBUTES/WRS_ROW",
    "cloud_cover": ".//IMAGE_ATTRIBUTES/CLOUD_COVER",
    "cloud_cover_land": ".//IMAGE_ATTRIBUTES/CLOUD_COVER_LAND",
    "sun_azimuth": ".//IMAGE_ATTRIBUTES/SUN_AZIMUTH",
    "sun_elevation": ".//IMAGE_ATTRIBUTES/SUN_ELEVATION",
    "earth_sun_distance": ".//IMAGE_ATTRIBUTES/EARTH_SUN_DISTANCE",
    "image_quality_oli": ".//IMAGE_ATTRIBUTES/IMAGE_QUALITY_OLI",
    "image_quality_tirs": ".//IMAGE_ATTRIBUTES/IMAGE_QUALITY_TIRS",
    # --- Projection attributes (product corner coords & raster geometry) ---
    "map_projection": ".//PROJECTION_ATTRIBUTES/MAP_PROJECTION",
    "datum": ".//PROJECTION_ATTRIBUTES/DATUM",
    "ellipsoid": ".//PROJECTION_ATTRIBUTES/ELLIPSOID",
    "utm_zone": ".//PROJECTION_ATTRIBUTES/UTM_ZONE",
    "grid_cell_size_panchromatic": ".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_PANCHROMATIC",
    "grid_cell_size_reflective": ".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_REFLECTIVE",
    "grid_cell_size_thermal": ".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_THERMAL",
    "orientation": ".//PROJECTION_ATTRIBUTES/ORIENTATION",
    # Corner coordinates (product lat/lon)
    "corner_ul_lat_product": ".//PROJECTION_ATTRIBUTES/CORNER_UL_LAT_PRODUCT",
    "corner_ul_lon_product": ".//PROJECTION_ATTRIBUTES/CORNER_UL_LON_PRODUCT",
    "corner_ur_lat_product": ".//PROJECTION_ATTRIBUTES/CORNER_UR_LAT_PRODUCT",
    "corner_ur_lon_product": ".//PROJECTION_ATTRIBUTES/CORNER_UR_LON_PRODUCT",
    "corner_lr_lat_product": ".//PROJECTION_ATTRIBUTES/CORNER_LR_LAT_PRODUCT",
    "corner_lr_lon_product": ".//PROJECTION_ATTRIBUTES/CORNER_LR_LON_PRODUCT",
    "corner_ll_lat_product": ".//PROJECTION_ATTRIBUTES/CORNER_LL_LAT_PRODUCT",
    "corner_ll_lon_product": ".//PROJECTION_ATTRIBUTES/CORNER_LL_LON_PRODUCT",
    # --- Level-1 processing record ---
    "date_product_generated": ".//LEVEL1_PROCESSING_RECORD/DATE_PRODUCT_GENERATED",
    "processing_software_version": ".//LEVEL1_PROCESSING_RECORD/PROCESSING_SOFTWARE_VERSION",
}

# Programmatically add radiometric rescaling keys and thermal constants (bands)
for band_num in range(1, 12):
    LS_PROD_MTD_METADATA_PATHS[f"radiance_mult_band_{band_num}"] = (
        f".//LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_MULT_BAND_{band_num}"
    )
    LS_PROD_MTD_METADATA_PATHS[f"radiance_add_band_{band_num}"] = (
        f".//LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_ADD_BAND_{band_num}"
    )

# Reflectance coefficients exist only for reflective bands (B1..B9).
for band_num in range(1, 10):
    LS_PROD_MTD_METADATA_PATHS[f"reflectance_mult_band_{band_num}"] = (
        f".//LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_MULT_BAND_{band_num}"
    )
    LS_PROD_MTD_METADATA_PATHS[f"reflectance_add_band_{band_num}"] = (
        f".//LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_ADD_BAND_{band_num}"
    )

# Thermal constants (K1, K2) for B10/B11.
LS_PROD_MTD_METADATA_PATHS.update(
    {
        "k1_constant_band_10": ".//LEVEL1_THERMAL_CONSTANTS/K1_CONSTANT_BAND_10",
        "k2_constant_band_10": ".//LEVEL1_THERMAL_CONSTANTS/K2_CONSTANT_BAND_10",
        "k1_constant_band_11": ".//LEVEL1_THERMAL_CONSTANTS/K1_CONSTANT_BAND_11",
        "k2_constant_band_11": ".//LEVEL1_THERMAL_CONSTANTS/K2_CONSTANT_BAND_11",
    }
)

# Projection parameters (resampling etc.).
LS_PROD_MTD_METADATA_PATHS.update(
    {
        "proj_params_map_projection": ".//LEVEL1_PROJECTION_PARAMETERS/MAP_PROJECTION",
        "proj_params_datum": ".//LEVEL1_PROJECTION_PARAMETERS/DATUM",
        "proj_params_ellipsoid": ".//LEVEL1_PROJECTION_PARAMETERS/ELLIPSOID",
        "proj_params_utm_zone": ".//LEVEL1_PROJECTION_PARAMETERS/UTM_ZONE",
        "proj_params_grid_cell_size_panchromatic": ".//LEVEL1_PROJECTION_PARAMETERS/GRID_CELL_SIZE_PANCHROMATIC",
        "proj_params_grid_cell_size_reflective": ".//LEVEL1_PROJECTION_PARAMETERS/GRID_CELL_SIZE_REFLECTIVE",
        "proj_params_grid_cell_size_thermal": ".//LEVEL1_PROJECTION_PARAMETERS/GRID_CELL_SIZE_THERMAL",
        "proj_params_orientation": ".//LEVEL1_PROJECTION_PARAMETERS/ORIENTATION",
        "proj_params_resampling_option": ".//LEVEL1_PROJECTION_PARAMETERS/RESAMPLING_OPTION",
    }
)


class LSL1ProdXMLReader(XMLReader):
    """
    Landsat Collection-2 Level-1 product metadata reader.

    This class provides semantic accessors for commonly used fields in the
    Landsat product metadata XML (MTL.xml). It builds on the generic
    :class:`XMLReader` to return domain-appropriate Python objects.
    """

    metadata_paths = LS_PROD_MTD_METADATA_PATHS

    # ---------------------------------------------------------------------
    # Temporal metadata
    # ---------------------------------------------------------------------
    def find_acquisition_datetime(self) -> dt.datetime:
        """
        Return the scene center sensing time as a timezone-aware datetime (UTC).

        Combines DATE_ACQUIRED (YYYY-MM-DD) and SCENE_CENTER_TIME (HH:MM:SS[.fffffff]Z).

        :returns: Acquisition datetime as a datetime.datetime object (UTC).
        """
        date_str = self.find_value("date_acquired")
        time_str = self.find_value("scene_center_time")

        if not date_str or not time_str:
            raise ValueError("Missing DATE_ACQUIRED or SCENE_CENTER_TIME in metadata")

        # Robust parse of fractional seconds; 'Z' denotes UTC
        dt_str = f"{date_str}T{time_str}"
        try:
            # if str between . and Z has length > 6, truncate to microseconds
            if "." in dt_str:
                frac_part = dt_str.split(".")[1].rstrip("Z")
                if len(frac_part) > 6:
                    dt_str = dt_str.replace(frac_part + "Z", frac_part[:6] + "Z")
            dt_obj = dt.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            dt_obj = dt.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")

        return dt_obj.replace(tzinfo=dt.timezone.utc)

    def find_acquisition_date(self) -> dt.date:
        """
        Return the acquisition date (DATE_ACQUIRED).

        :returns: Acquisition date as a datetime.date object.
        """
        return self.find_acquisition_datetime().date()

    # ---------------------------------------------------------------------
    # Identification / provenance
    # ---------------------------------------------------------------------
    def find_product_id(self) -> str:
        """Return the LANDSAT_PRODUCT_ID string."""
        return self.find_value("landsat_product_id")

    def find_processing_level(self) -> str:
        """Return the PROCESSING_LEVEL (e.g., 'L1TP')."""
        return self.find_value("processing_level")

    def find_collection_number(self) -> str:
        """Return the COLLECTION_NUMBER (e.g., '02')."""
        return self.find_value("collection_number")

    def find_collection_category(self) -> str:
        """Return the COLLECTION_CATEGORY (e.g., 'T1')."""
        return self.find_value("collection_category")

    def find_spacecraft_name(self) -> str:
        """Return the spacecraft name (e.g., 'LANDSAT_8')."""
        return self.find_value("spacecraft_id")

    def find_sensor_id(self) -> str:
        """Return the SENSOR_ID (e.g., 'OLI_TIRS')."""
        return self.find_value("sensor_id")

    def find_wrs_path(self) -> int:
        """Return the WRS_PATH as integer if possible."""
        return self.find_value("wrs_path")

    def find_wrs_row(self) -> int:
        """Return the WRS_ROW as integer if possible."""
        return self.find_value("wrs_row")

    # ---------------------------------------------------------------------
    # Geometry / footprint
    # ---------------------------------------------------------------------
    def find_bounds(self):
        """
        Return the product footprint as a Shapely polygon in lon/lat order.

        Uses PROJECTION_ATTRIBUTES corner lat/lon (PRODUCT) coordinates: UL, UR, LR, LL.
        The polygon is ordered UL -> UR -> LR -> LL -> UL and validated.
        """
        Polygon, _, _, _, _ = lazy_shapely()

        ul = (
            self.find_value("corner_ul_lon_product"),
            self.find_value("corner_ul_lat_product"),
        )
        ur = (
            self.find_value("corner_ur_lon_product"),
            self.find_value("corner_ur_lat_product"),
        )
        lr = (
            self.find_value("corner_lr_lon_product"),
            self.find_value("corner_lr_lat_product"),
        )
        ll = (
            self.find_value("corner_ll_lon_product"),
            self.find_value("corner_ll_lat_product"),
        )

        coords_lonlat = [ul, ur, lr, ll]
        # Ensure closure
        if coords_lonlat[0] != coords_lonlat[-1]:
            coords_lonlat.append(coords_lonlat[0])

        poly = Polygon(coords_lonlat)

        if not poly.is_valid:
            raise ValueError("Invalid footprint polygon constructed from corner coordinates")

        return poly

    # ---------------------------------------------------------------------
    # Radiometric metadata
    # ---------------------------------------------------------------------
    def get_radiometric_rescaling(self) -> dict:
        """
        Extract all radiometric rescaling parameters for bands B1..B11.

        Returns a dict mapping band name ('B1'..'B11') to:
        - radiance_mult, radiance_add (all bands)
        - reflectance_mult, reflectance_add (B1..B9 only)
        """
        rescaling: dict[str, dict] = {}
        for band_num in range(1, 12):
            band_name = f"B{band_num}"
            rescaling[band_name] = {
                "radiance_mult": self.find_value(f"radiance_mult_band_{band_num}", default=None),
                "radiance_add": self.find_value(f"radiance_add_band_{band_num}", default=None),
            }
            if band_num <= 9:
                rescaling[band_name]["reflectance_mult"] = self.find_value(
                    f"reflectance_mult_band_{band_num}", default=None
                )
                rescaling[band_name]["reflectance_add"] = self.find_value(
                    f"reflectance_add_band_{band_num}", default=None
                )
        return rescaling

    def get_thermal_constants(self) -> dict:
        """
        Extract thermal constants (K1, K2) for thermal bands B10, B11.
        Returns dict: {"B10": {"k1": ..., "k2": ...}, "B11": {"k1": ..., "k2": ...}}
        """
        return {
            "B10": {
                "k1": self.find_value("k1_constant_band_10", default=None),
                "k2": self.find_value("k2_constant_band_10", default=None),
            },
            "B11": {
                "k1": self.find_value("k1_constant_band_11", default=None),
                "k2": self.find_value("k2_constant_band_11", default=None),
            },
        }

    # ---------------------------------------------------------------------
    # Projection parameters (optional convenience)
    # ---------------------------------------------------------------------
    def find_projection_parameters(self) -> dict:
        """
        Return Level-1 projection parameters (map projection, grid sizes,
        orientation, resampling option).
        """
        return {
            "map_projection": self.find_value("proj_params_map_projection", default=None),
            "datum": self.find_value("proj_params_datum", default=None),
            "ellipsoid": self.find_value("proj_params_ellipsoid", default=None),
            "utm_zone": self.find_value("proj_params_utm_zone", default=None),
            "grid_cell_size_panchromatic": self.find_value("proj_params_grid_cell_size_panchromatic", default=None),
            "grid_cell_size_reflective": self.find_value("proj_params_grid_cell_size_reflective", default=None),
            "grid_cell_size_thermal": self.find_value("proj_params_grid_cell_size_thermal", default=None),
            "orientation": self.find_value("proj_params_orientation", default=None),
            "resampling_option": self.find_value("proj_params_resampling_option", default=None),
        }


if __name__ == "__main__":
    pass
