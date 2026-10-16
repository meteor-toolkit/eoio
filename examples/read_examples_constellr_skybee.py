"""read_examples_constellr_skybee - Example usage script for interface functions for L1C data (ConstellR Skybee)"""

import configparser
import os
import time

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from eoio.interface import (
    product_options,
    product_processors,
    read,
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

# Assume config has a section [constellr] and keys 'HIVE_L1C_dir'
# example_data_path = config.get("constellr", "HIVE_L1C_dir", fallback=None)

example_data_path = r"T:\ECO\EOServer\data\unittest_datasets\constellr\HIVE_L1C_SBA02_20260703T115605Z_A715C2314D_20260903T142658Z"

# define roi for subsetting (example data is over RCN-GONA site)
GONA_coords = (
    (15.11465119, -23.60472105),
    (15.12446109, -23.60472105),
    (15.12446109, -23.59568076),
    (15.11465119, -23.59568076),
    (15.11465119, -23.60472105),
)
roi = Polygon(GONA_coords)


# example retrieve valid processors and their params
skybee_process_params = product_processors(example_data_path)
print("Skybee process params: \n", skybee_process_params)

# example retrieve valid product subsetting parameters
skybee_subset_params = product_options(example_data_path)
print("Skybee subset params: \n", skybee_subset_params)

# test read

# start timer for read
t_0 = time.time()

ds = read(
    example_data_path,
    vars_sel={"meas": "all", "aux": ["observation_geometry"]},
    subset={"roi": roi, "roi_crs": 4326},
    read_params={"metadata_level": "all", "save_extracted": True, "use_chunks": True},
    processors={"units.convert": {"to": "reflectance", "var_names": ["B2", "B3", "B4"]}},
)
# stop timer for read and print
print("READ TIME: " + f"{time.time() - t_0}")


# # check attrs & vars
# for attr in ds.attrs:
#     print(f"{attr}: {ds.attrs[attr]}")

# for var in ds.data_vars:
#     print(f"\nVariable: {var}")
#     for attr in ds[var].attrs:
#         print(f"  {attr}: {ds[var].attrs[attr]}")

print(ds)

# Plot reflectance imagery
fig, axs = plt.subplots(2, 2, figsize=(17, 14), sharex=True, sharey=True)

ds.B2.plot(ax=axs[0, 0], x="longitude_10m", y="latitude_10m", robust=True)
ds.B3.plot(ax=axs[0, 1], x="longitude_10m", y="latitude_10m", robust=True)
ds.B4.plot(ax=axs[1, 0], x="longitude_10m", y="latitude_10m", robust=True)
ds.B5.plot(ax=axs[1, 1], x="longitude_20m", y="latitude_20m", robust=True)

plt.savefig(
    f"skybee_{ds['B2'].attrs['product_metadata']['measurand']}_{os.path.splitext(os.path.basename(example_data_path))[0]}.png",
    dpi=300,
)
print(
    f"Saved: skybee_{ds['B2'].attrs['product_metadata']['measurand']}_{os.path.splitext(os.path.basename(example_data_path))[0]}.png"
)

plt.show()#block=False)
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
