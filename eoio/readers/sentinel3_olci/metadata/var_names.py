"""Sentinel-3 OLCI variable name definitions for auxiliary and mask variables."""

AUX_OPTIONS = [
    "observation_geometry",
    "removed_pixels",
    "humidity",
    "sea_level_pressure",
    "total_columnar_water_vapour",
    "total_ozone",
    "horizontal_wind",
    "atmospheric_temperature_profile",
    "FWHM",
    "detector_index",
    "frame_offset",
    "lambda0",
    "relative_spectral_covariance",
    "solar_flux",
]

MASK_OPTIONS = [
    "saturated",
    "dubious",
    "sun-glint_risk",
    "duplicated",
    "cosmetic",
    "invalid",
    "straylight_risk",
    "bright",
    "tidal_region",
    "fresh_inland_water",
    "coastline",
    "land",
]
