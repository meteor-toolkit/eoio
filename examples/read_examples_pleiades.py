"""read_examples_pleiades - Example usage script for interface functions for L1 data (Airbus MS ORT)"""

import os
import matplotlib.pyplot as plt
import configparser
from shapely.geometry import Polygon
import time

from eoio.interface import (
    read,
    product_processors,
    product_options,
)

__author__ = "Samantha Malone <samantha.malone@npl.co.uk>"

__all__ = []

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "eoio",
    "etc",
    "test_file_paths.config",
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# Assume config has a section [Airbus] and keys 'phr_dir_GONA_crop', 'phr_dir_l4_full_scene'
example_data_path = config.get("Airbus", "phr_dir_GONA_crop", fallback=None)

# define roi for subsetting (example data is over RCN-GONA site and Libya-4)
if "cropped_tile" in example_data_path:
    GONA_coords = (
        (15.11465119, -23.60472105),
        (15.12446109, -23.60472105),
        (15.12446109, -23.59568076),
        (15.11465119, -23.59568076),
        (15.11465119, -23.60472105),
    )
    roi = Polygon(GONA_coords)
elif "full_scene" in example_data_path:
    L4_coords = (
        (23.38, 28.54),
        (23.4, 28.54),
        (23.4, 28.56),
        (23.38, 28.56),
    )  # 28.55, 23.39
    roi = Polygon(L4_coords)


# example retrieve valid processors and their params
phr_process_params = product_processors(example_data_path)
print("PHR process params: \n", phr_process_params)

# example retrieve valid product subsetting parameters
phr_subset_params = product_options(example_data_path)
print("PHR subset params: \n", phr_subset_params)

# test read

# start timer for read
t_0 = time.time()

ds = read(
    example_data_path,
    vars_sel={"meas": "all", "aux": ["observation_geometry"]},
    subset={"roi": roi, "roi_crs": 4326},
    read_params={"metadata_level": "all", "save_extracted": True, "use_chunks": True},
    processors={"units.convert": {"to": "reflectance", "var_names": ["B1", "B2", "B3", "B4"]}},
)
# stop timer for read and print
print("READ TIME: " + f"{time.time() - t_0}")

for attr in ds.attrs:
    print(f"{attr}: {ds.attrs[attr]}")

for var in ds.data_vars:
    print(f"\nVariable: {var}")
    for attr in ds[var].attrs:
        print(f"  {attr}: {ds[var].attrs[attr]}")

print(ds)

# Plot reflectance imagery
fig, axs = plt.subplots(2, 2, figsize=(17, 14), sharex=True, sharey=True)

ds.B1.plot(ax=axs[0, 0], x="longitude_2m", y="latitude_2m", robust=True)
ds.B2.plot(ax=axs[0, 1], x="longitude_2m", y="latitude_2m", robust=True)
ds.B3.plot(ax=axs[1, 0], x="longitude_2m", y="latitude_2m", robust=True)
ds.B4.plot(ax=axs[1, 1], x="longitude_2m", y="latitude_2m", robust=True)

plt.savefig(
    f"pleiades_{ds['B2'].attrs['product_metadata']['measurand']}_{os.path.splitext(os.path.basename(example_data_path))[0]}.png",
    dpi=300,
)
print(
    f"Saved: pleiades_{ds['B2'].attrs['product_metadata']['measurand']}_{os.path.splitext(os.path.basename(example_data_path))[0]}.png"
)

plt.show(block=False)
plt.pause(2)
plt.close()

# # remove ds.attrs that are dicts - must be str, Number, ndarray, number, list, tuple for saving to netcdf
# del_list = []
# for attr in ds.attrs:
#     if isinstance(ds.attrs[attr], dict):
#         del_list.append(attr)
#     # change ds.attrs that are datetime to str
#     if isinstance(ds.attrs[attr], datetime.datetime):
#         ds.attrs[attr] = str(ds.attrs[attr])

# for attr in del_list:
#     del ds.attrs[attr]

# Save ds to netcdf
file = f"ds_{os.path.splitext(os.path.basename(example_data_path))[0]}.nc"
# ds.to_netcdf(file)  # save here

print(f"{file} processed")

if __name__ == "__main__":
    pass
