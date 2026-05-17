from eoio.interface import read

ds = read(
    r"T:/ECO/EOServer/data/satellite/EMIT/EMIT_L1B_OBS_001_20230729T113445_2321008_034.nc",
    vars_sel={"aux": "elev"},
    subset={
        "roi": ((23.38, 28.79), 100),  # (mid_lon_lat(s2_l1c_filepaths[0]), 3000),
        "roi_crs": 4326,
        "wavelength": {"min": 500, "max": 1000},
    },
)

print(ds)
