"""eoio.readers.constellr_skybee.metadata - extractor class for constellr Skybee VNIR metadata."""

from __future__ import annotations

import datetime as dt
import json
import warnings
from typing import Any

from processor_tools.utils.dict_tools import get_value

from eoio.readers.footprint_utils import normalize_footprint
from eoio.readers.metadata import BaseMetadataExtractor


# mask variable names (see eoio.readers.constellr_skybee.masks)
MASK_NAMES = ("blackfill_mask", "saturated_mask", "untested_mask")


class ConstellrSkybeeMetadataExtractorError(ValueError):
    pass


class ConstellrSkybeeMetadataExtractor(BaseMetadataExtractor):
    """Metadata helper for constellr Skybee L1C scene products."""

    def __init__(self, reader):
        super().__init__(reader)
        self.metadata_filepath = reader.layout.metadata_file()
        self._full_md: dict[str, Any] | None = None

    def read_metadata(self) -> dict[str, Any]:
        """The product's ``metadata.json`` as a dict (``{}`` if absent or unreadable, with a warning for
        the latter). Parsed once: it is asked for per band."""
        if self._full_md is None:
            self._full_md = self._load_metadata()
        return self._full_md

    def _load_metadata(self) -> dict[str, Any]:
        if self.metadata_filepath is None or not self.metadata_filepath.exists():
            return {}
        if self.metadata_filepath.suffix.lower() != ".json":
            return {}
        try:
            with open(self.metadata_filepath, encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
        except (OSError, ValueError) as exc:
            warnings.warn(f"Could not read {self.metadata_filepath}: {exc}; metadata will be empty.")
            return {}

    def get_basic_metadata(self) -> dict[str, Any]:
        full_md = self.read_metadata()

        product_datetime = None
        acquisition_datetime = get_value(full_md, "acquisition_datetime")
        if isinstance(acquisition_datetime, str):
            try:
                # naive UTC isoformat, as the other readers
                product_datetime = (
                    dt.datetime.fromisoformat(acquisition_datetime.replace("Z", "+00:00"))
                    .astimezone(dt.timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat()
                )
            except ValueError:
                product_datetime = None

        meas_var_res = self.reader.meas_var_res  # type: ignore[attr-defined]
        geometry = get_value(full_md, "geometry")
        observation_geometry = self._observation_geometry(
            get_value(full_md, "solar_angles"), get_value(full_md, "viewing_angles")
        )
        satellite = get_value(full_md, "platform")

        basic_md = {
            "collection_name": "HIVE",
            "product_name": get_value(full_md, "product_id"),
            "platform": "Skybee",
            "instrument": "VNIR",
            "processing_level": get_value(full_md, "product_type"),
            "processing_version": get_value(full_md, "workflow_version"),
            "satellite_id": str(satellite).upper() if satellite else None,
            "constellation": "HIVE",
            "spatial_resolution": [meas_var_res[var] for var in meas_var_res.keys()],
            "geometry_ids": [f"{meas_var_res[var]}m" for var in meas_var_res.keys()],
            "product_geospatial_bounds": get_value(full_md, "bbox"),  # EPSG:4326
            "product_datetime": product_datetime,
            "description": "constellr Skybee multispectral scene product",
            "institution": "constellr",
            "keywords": ["constellr", "hive", "skybee", "satellite", "remote sensing", "multispectral"],
            "source": "constellr HIVE Skybee satellite product",
            "platform_type": "satellite",
            "observation_geometry": observation_geometry,
            "geospatial_bounds_crs": self._product_crs(full_md),
            "footprint": normalize_footprint(geometry_input=geometry, crs_input=4326) if geometry else None,
            "cloud_area_fraction": self._cloud_area_fraction(full_md),
        }

        return basic_md

    @staticmethod
    def _observation_geometry(solar_angles: Any, viewing_angles: Any) -> dict[str, float]:
        """Whole-scene sun/view angles (degrees), or ``{}`` when the product does not carry them -- so a
        product with a gap there still reads (and can be harvested) rather than failing on a missing key."""
        geometry: dict[str, float] = {}
        if isinstance(solar_angles, dict):
            if "view:sun_azimuth" in solar_angles:
                geometry["solar_azimuth_angle"] = solar_angles["view:sun_azimuth"]
            if "view:sun_elevation" in solar_angles:
                geometry["solar_zenith_angle"] = 90 - solar_angles["view:sun_elevation"]
        if isinstance(viewing_angles, dict):
            if "view:azimuth" in viewing_angles:
                geometry["viewing_azimuth_angle"] = viewing_angles["view:azimuth"]
            # incident angle: measured against the local vertical at the ground (off_nadir is at the satellite)
            if "view:incident_angle" in viewing_angles:
                geometry["viewing_zenith_angle"] = viewing_angles["view:incident_angle"]
        return geometry

    @staticmethod
    def _cloud_area_fraction(full_md: dict[str, Any]) -> float | None:
        """Scene cloud cover (%), or ``None`` when constellr did not compute it (``cloud_cover_computed``
        false): the percentage then is a placeholder, and reporting it would read as a clear scene."""
        if get_value(full_md, "cloud_cover_computed") is False:
            return None
        return get_value(full_md, "total_cloud_percentage")

    def _product_crs(self, full_md: dict[str, Any]) -> str:
        """CRS of the product imagery; "epsg" in the json refers to the Sentinel-2 georef reference, so prefer the tiff."""
        try:
            from eoio.deps import lazy_rasterio

            with lazy_rasterio().open(self.reader.layout.image_file()) as src:  # type: ignore[attr-defined]
                epsg = src.crs.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
        except Exception:
            pass
        epsg = get_value(full_md, "epsg")
        if epsg is None:
            raise ConstellrSkybeeMetadataExtractorError(
                "Cannot determine the product CRS: the imagery could not be opened and metadata.json has no epsg."
            )
        return f"EPSG:{epsg}"

    def get_product_metadata(self) -> dict[str, Any]:
        return self.get_basic_metadata()

    def get_variable_product_metadata(self, var: str) -> dict[str, Any]:
        if var == "observation_geometry":
            return {"units": "degrees"}

        if var in MASK_NAMES:
            return {"spatial_resolution": 10, "spatial_resolution_units": "m", "geometry_id": "10m"}

        full_md = self.read_metadata()

        # band names also appear as keys in gver_metrics, so index the sensor block directly
        band_key = f"VNIR{int(var[1:]):02d}"
        try:
            bnd_mtd = full_md["sensors"]["VNIR"]["bands"][band_key]
        except KeyError as exc:
            raise ConstellrSkybeeMetadataExtractorError(f"No {band_key} entry in {self.metadata_filepath}.") from exc

        meas_var_res = self.reader.meas_var_res  # type: ignore[attr-defined]

        return {
            "band_id": var,
            "band_name": var,
            "band_central_wavelength": round(float(bnd_mtd["band_centre"]) * 1000, 6),
            "band_central_wavelength_units": "nm",
            "band_full_width_half_max": round(float(bnd_mtd["band_width"]) * 1000, 6),
            "standard_name": "toa_radiance",
            "long_name": f"TOA radiance in band {var}",
            "units": "W/(m²·sr·μm)",
            "spatial_resolution": meas_var_res.get(var),
            "spatial_resolution_units": "m",
            "geometry_id": f"{meas_var_res.get(var)}m",
            "ancillary_variables": [],
            "measurand": "radiance",
            "band_solar_irradiance": bnd_mtd["esun"]["value"],
            "earth_sun_distance": get_value(full_md, "earth_sun_distance"),
        }

    def get_variable_basic_metadata(self, var: str) -> dict[str, Any]:
        if var == "observation_geometry":
            return {}
        if var in MASK_NAMES:
            return {}
        return {
            "standard_name": "toa_radiance",
            "long_name": f"TOA radiance in band {var}",
            "units": "W/(m²·sr·μm)",
            "measurand": "radiance",
        }
