"""Sentinel-2 reader variable names"""

# MAPPING BETWEEN MEAS_VARS NAMES AND BAND_ID IN METADATA
MEAS_VAR_BAND_IDS = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}

BAND_ID_MEAS_VARS = {v: k for k, v in MEAS_VAR_BAND_IDS.items()}

# Additional IMG_DATA layers present in Level-2A (Sen2Cor outputs)
L2A_IMG_VARS = ["AOT", "WVP", "SCL", "TCI"]

# Common QI_DATA probability layers (Level-2A)
L2A_QI_VARS = ["CLDPRB", "SNWPRB"]


# AUX_OPTIONS - Available auxiliary variable names

# METEO VARIABLES
# NB: depends on the processing baseline, more were added in 04.00
AUX_ECMWF_VARS_OLD = ["tco3", "tcwv", "msl"]

AUX_ECMWF_VARS_NEW = [
    "tco3",
    "tcwv",
    "msl",
    "u10",
    "v10",
    "r",
]

AUX_CAMS_VARS = [
    "aod550",
    "z",
    "bcaod550",
    "duaod550",
    "omaod550",
    "ssaod550",
    "suaod550",
    "aod469",
    "aod670",
    "aod865",
    "aod1240",
]

# ANGLE VARIABLES
ANGLE_VARS = [
    "viewing_zenith_angle_B01",
    "viewing_zenith_angle_B02",
    "viewing_zenith_angle_B03",
    "viewing_zenith_angle_B04",
    "viewing_zenith_angle_B05",
    "viewing_zenith_angle_B06",
    "viewing_zenith_angle_B07",
    "viewing_zenith_angle_B08",
    "viewing_zenith_angle_B8A",
    "viewing_zenith_angle_B09",
    "viewing_zenith_angle_B10",
    "viewing_zenith_angle_B11",
    "viewing_zenith_angle_B12",
    "viewing_azimuth_angle_B01",
    "viewing_azimuth_angle_B02",
    "viewing_azimuth_angle_B03",
    "viewing_azimuth_angle_B04",
    "viewing_azimuth_angle_B05",
    "viewing_azimuth_angle_B06",
    "viewing_azimuth_angle_B07",
    "viewing_azimuth_angle_B08",
    "viewing_azimuth_angle_B8A",
    "viewing_azimuth_angle_B09",
    "viewing_azimuth_angle_B10",
    "viewing_azimuth_angle_B11",
    "viewing_azimuth_angle_B12",
    "solar_zenith_angle",
    "solar_azimuth_angle",
]

# MASK_OPTIONS - Available mask variable names
MASK_OPTIONS = [
    "detector_footprint",
    "ancillary_lost",
    "ancillary_degraded",
    "msi_lost",
    "msi_degraded",
    "defective",
    "nodata",
    "partially_corrected_crosstalk",
    "saturated_l1a",
    "opaque",
    "cirrus",
    "snow_and_ice_areas",
]
