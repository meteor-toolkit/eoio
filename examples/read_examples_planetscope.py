"""read_examples_planetscope - Example usage script for interface functions for L1 data (Planet L3b)"""

import os
import matplotlib.pyplot as plt
import configparser
from shapely.geometry import Polygon
from processor_tools.utils.dict_tools import get_value
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
# Assume config has a section [PlanetScope] and key 'psd_tif'
example_data_path = config.get("PlanetScope", "psd_tif", fallback=None)


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
psd_process_params = product_processors(example_data_path)
print("PSD process params: \n", psd_process_params)

# example retrieve valid product subsetting parameters
psd_subset_params = product_options(example_data_path)
print("PSD subset params: \n", psd_subset_params)

# test read

# start timer for read
t_0 = time.time()

ds = read(
    example_data_path,
    vars_sel={"meas": "all", "aux": ["observation_geometry"]},
    subset={"roi": roi, "roi_crs": 4326},
    read_params={"metadata_level": "all", "save_extracted": True, "use_chunks": True},
    processors={"units.convert": {"to": "reflectance", "var_names": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]}},
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

# quick plot
fig, axs = plt.subplots(1, 3, figsize=(15, 8))

ds.B2.plot(ax=axs[0], x="x_3m", y="y_3m", robust=True)  # previously was x="longitude_3m", y="latitude_3m"
ds.B4.plot(ax=axs[1], x="x_3m", y="y_3m", robust=True)
ds.B6.plot(ax=axs[2], x="x_3m", y="y_3m", robust=True)

plt.show(block=False)
plt.pause(2)
plt.savefig(
    f"planet_{ds['B2'].attrs['product_metadata']['measurand']}_{os.path.splitext(os.path.basename(example_data_path))[0]}.png"
)
print(
    f"Saved: planet_{ds['B2'].attrs['product_metadata']['measurand']}_{os.path.splitext(os.path.basename(example_data_path))[0]}.png"
)
plt.close()

# # remove ds.attrs that are dicts - must be str, Number, ndarray, number, list, tuple for saving to netcdf
# for attr in ds.attrs:
#     if isinstance(ds.attrs[attr], dict):
#         del_list.append(attr)
#     # change ds.attrs that are datetime to str
#     if isinstance(ds.attrs[attr], datetime.datetime):
#         ds.attrs[attr] = str(ds.attrs[attr])

# for attr in del_list:
#     del ds.attrs[attr]

# Save ds to netcdf
file = f"ds_{get_value(ds.attrs, 'platform').lower()}_{os.path.splitext(os.path.basename(example_data_path))[0]}.nc"
# ds.to_netcdf(file)  # save here

print(f"{file} processed")

if __name__ == "__main__":
    pass
