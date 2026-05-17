from eoio.interface import (
    read,
)
import os
import socket

if __name__ == "__main__":
    if socket.gethostname() == "eoserver.npl.co.uk":
        s2_path = r"/mnt/t/data/product_archive_temp/S2A/S2_MSI_L1C/2022/01/10/S2A_MSIL1C_20220110T041141_N0301_R047_T48VXL_20220110T060721.SAFE"
        l8_path = (
            r"/mnt/t/data/product_archive_temp/LC08/LANDSAT_C2L1/2022/01/10/LC08_L1TP_134018_20220110_20220114_02_T1"
        )
        OUTPUT_DIRECTORY = r"/mnt/t/matchup_pipeline"
    elif socket.gethostname() == "lyon.npl.co.uk":
        s2_path = r"/mnt/t/data/product_archive_temp/S2A/S2_MSI_L1C/2022/01/10/S2A_MSIL1C_20220110T041141_N0301_R047_T48VXL_20220110T060721.SAFE"
        l8_path = (
            r"/mnt/t/data/product_archive_temp/LC08/LANDSAT_C2L1/2022/01/10/LC08_L1TP_134018_20220110_20220114_02_T1"
        )
        OUTPUT_DIRECTORY = r"../../../matchup_pipeline"
    else:
        s2_path = r"T:\ECO\EOServer\data\product_archive_temp\S2A\S2_MSI_L1C\2022\01\10\S2A_MSIL1C_20220110T041141_N0301_R047_T48VXL_20220110T060721.SAFE"
        l8_path = r"T:\ECO\EOServer\data\product_archive_temp\LC08\LANDSAT_C2L1\2022\01\10\LC08_L1TP_134018_20220110_20220114_02_T1"
        OUTPUT_DIRECTORY = r"T:\ECO\EOServer\matchup_pipeline"

    roi = [
        (106.76408088371669, 59.45361822597861),
        (106.76776536571, 59.526408304952),
        (108.70528899271, 59.485839959672),
        (108.66798951247867, 59.13410307779184),
        (106.76408088371669, 59.45361822597861),
    ]

    print(s2_path, l8_path, roi)
    s2_ds = read(
        path=s2_path,
        subset_info={
            "meas": ["B01"],  # , "B10"],
            "read_img": True,
            "roi": roi,
            "roi_crs": 4326,
            "metadataLevel": True,
            "aux": ["observation_geometry"],
        },
        process_params={
            "convert": ["radiance"],
            # "angles": {"angle_type": 'both'},
        },
    )
    print(s2_ds.B01.attrs)
    for var in s2_ds:
        s2_ds[var].attrs = {}
    s2_ds.attrs = {}
    s2_ds.to_netcdf(os.path.join(OUTPUT_DIRECTORY, "eoio_radiance_s2_ds.nc"))

    l8_ds = read(
        l8_path,
        subset_info={
            # "meas": ["B01", "B10"],
            "read_img": True,
            "roi": roi,
            "roi_crs": 4326,
            "metadataLevel": True,
            "aux": ["observation_geometry"],
        },
        process_params={
            "convert": ["radiance"],
            # "angles": {"angle_type": 'both'},
        },
    )
    for var in l8_ds:
        l8_ds[var].attrs = {}
    l8_ds.attrs = {}
    l8_ds.to_netcdf(os.path.join(OUTPUT_DIRECTORY, "eoio_radiance_l8_ds.nc"))
