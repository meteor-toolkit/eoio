from eoio.interface import read
import time

start_time = time.time()
ds = read(
    r"T:\ECO\EOServer\data\unittest_datasets\MODIS\MOD09.A2024122.0830.061.2024124062519.hdf",
    vars_sel={"aux": "all", "meas": "all"},
    subset={
        "roi": (15.1, 30, 15.13, 30.1),  # ((5, -20), 10000), # (mid_lon_lat(s2_l1c_filepaths[0]), 3000),
        "roi_crs": 4326,
    },
    read_params={
        "preferred_resolution": 500,
        "metadata_level": "basic",
        "geolocation_dir": r"T:\ECO\EOServer\data\unittest_datasets\MODIS",
        # "use_chunks": True,
        # "chunks": {"x": 256, "y": 256},
    },
    processors={
        "interpolate": {
            "coords": ["latitude_1000m", "longitude_1000m"],
            "target_grid": ["latitude_500m", "longitude_500m"],
            "method": "cubic",
        }
    },
)

print(ds)
print(ds["Band 1"].values)
print(ds["solar_zenith_angle"].values)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
