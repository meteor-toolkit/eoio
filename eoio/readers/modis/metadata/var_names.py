"""MODIS reader variable names"""

MEAS_VAR_BAND_IDS_Q = {
    "Band 1": "EV_250_RefSB",
    "Band 2": "EV_250_RefSB",
}

MEAS_VAR_BAND_IDS_H = {
    "Band 1": "EV_250_Aggr500_RefSB",
    "Band 2": "EV_250_Aggr500_RefSB",
    "Band 3": "EV_500_RefSB",
    "Band 4": "EV_500_RefSB",
    "Band 5": "EV_500_RefSB",
    "Band 6": "EV_500_RefSB",
    "Band 7": "EV_500_RefSB",
}

MEAS_VAR_BAND_IDS_1 = {
    "Band 1": "EV_250_Aggr1km_RefSB",
    "Band 2": "EV_250_Aggr1km_RefSB",
    "Band 3": "EV_500_Aggr1km_RefSB",
    "Band 4": "EV_500_Aggr1km_RefSB",
    "Band 5": "EV_500_Aggr1km_RefSB",
    "Band 6": "EV_500_Aggr1km_RefSB",
    "Band 7": "EV_500_Aggr1km_RefSB",
    "Band 8 ": "EV_1KM_RefSB",
    "Band 9 ": "EV_1KM_RefSB",
    "Band 10": "EV_1KM_RefSB",
    "Band 11": "EV_1KM_RefSB",
    "Band 12": "EV_1KM_RefSB",
    "Band 13": "EV_1KM_RefSB",
    "Band 14": "EV_1KM_RefSB",
    "Band 15": "EV_1KM_RefSB",
    "Band 16": "EV_1KM_RefSB",
    "Band 17": "EV_1KM_RefSB",
    "Band 18": "EV_1KM_RefSB",
    "Band 19": "EV_1KM_RefSB",
    "Band 20": "EV_1KM_Emissive",
    "Band 21": "EV_1KM_Emissive",
    "Band 22": "EV_1KM_Emissive",
    "Band 23": "EV_1KM_Emissive",
    "Band 24": "EV_1KM_Emissive",
    "Band 25": "EV_1KM_Emissive",
    "Band 26": "EV_1KM_RefSB",
    "Band 27": "EV_1KM_Emissive",
    "Band_28": "EV_1KM_Emissive",
    "Band_29": "EV_1KM_Emissive",
    "Band_30": "EV_1KM_Emissive",
    "Band_31": "EV_1KM_Emissive",
    "Band_32": "EV_1KM_Emissive",
    "Band_33": "EV_1KM_Emissive",
    "Band_34": "EV_1KM_Emissive",
    "Band_35": "EV_1KM_Emissive",
    "Band_36": "EV_1KM_Emissive",
}

L2_MEAS_VAR_BAND_IDS_Q = {
    "Band 1": "250m Surface Reflectance Band 1",
    "Band 2": "250m Surface Reflectance Band 2",
}

L2_MEAS_VAR_BAND_IDS_H = {
    "Band 1": "500m Surface Reflectance Band 1",
    "Band 2": "500m Surface Reflectance Band 2",
    "Band 3": "500m Surface Reflectance Band 3",
    "Band 4": "500m Surface Reflectance Band 4",
    "Band 5": "500m Surface Reflectance Band 5",
    "Band 6": "500m Surface Reflectance Band 6",
    "Band 7": "500m Surface Reflectance Band 7",
}

L2_MEAS_VAR_BAND_IDS_1 = {}  # todo: add 1km file parsing

# AUX_OPTIONS - Available auxiliary variable names
# ANGLE VARIABLES
ANGLE_VARS = [
    "viewing_zenith_angle",
    "viewing_azimuth_angle",
    "solar_zenith_angle",
    "solar_azimuth_angle",
]

ATMOS_VARS = [
    "TCWV",
    "AOD_Band1",
    "AOD_Band3",
    "AOD_Band8",
]
