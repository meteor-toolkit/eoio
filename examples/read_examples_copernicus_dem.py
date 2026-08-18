"""read_examples_copernicus_dem - Example usage script for interface functions"""

from eoio.interface import read
import os
import configparser

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "eoio",
    "etc",
    "test_file_paths.config",
)

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# Assume config has a section [hypernets] and key 'l1b_file'
example_data_path = config.get("CopernicusDEM", "copernicus_dem_30m", fallback=None)

ds = read(
    path=example_data_path,
    vars_sel={"meas": ["elevation"],
                "mask": ["water_body_mask", "filling_mask", "editing_mask", "height_error_mask"],
                "aux": []
                },
    subset={
        "roi": ((9, 9), 100),
        "roi_crs": 4326,
   },
   read_params ={
       "use_chunks": False,
       "metadata_level": "basic", # all/None
       "save_extracted" : False
   }
)

print(ds)

# try processing here
for attr in ds.attrs:
    print(f"{attr}: {ds.attrs[attr]}")

for var in ds.data_vars:
    print(f"\nVariable: {var}")
    for attr in ds[var].attrs:
        print(f"  {attr}: {ds[var].attrs[attr]}")
