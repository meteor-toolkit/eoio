"""read_examples_hypernets - Example usage script for interface functions"""

import os
import matplotlib.pyplot as plt
import configparser
import xarray as xr

from eoio.interface import (
    read,
    product_processors,
    product_options,
)

__author__ = "Pieter De Vis <pieter.de.vis@npl.co.uk>"

__all__ = []

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "eoio",
    "etc",
    "test_file_paths.config",
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# Assume config has a section [hypernets] and key 'l1b_file'
example_data_path = config.get("HYPERNETS", "l2a_file", fallback=None)

# example retrieve valid product subsetting parameters

hyp_subset_params = product_options(example_data_path)
print("Hypernets subset params: \n", hyp_subset_params["subset"])

# example retrieve valid processors and their params

hyp_process_params = product_processors(example_data_path)
print("Hypernets process params: \n", hyp_process_params)

L2b_path = r"T:\ECO\EOServer\data\insitu\hypernets\archive\GHNA\2025\06\06\SEQ20250606T133127\HYPERNETS_L_GHNA_L2B_REF_20250606T1331_20251114T1124_v2.2.nc"
L2a_path = r"T:\ECO\EOServer\data\insitu\hypernets\archive\GHNA\2025\06\06\SEQ20250606T133127\HYPERNETS_L_GHNA_L2A_REF_20250606T1331_20250625T1615_v2.1.nc"
# example read

vars_sel = {
    "meas": [
        "reflectance",
        "u_rel_systematic_reflectance",
        "err_corr_systematic_reflectance",
    ],
    "aux": "basic",
}
subset_dict = {
    "mask": True,
    #"angle": {"vza": {"nearest": 9.9, "tolerance": 1}, "vaa": {"min": 60, "max": 120}},
    "wavelength": {"nearest": 550, "tolerance": 10},
    #"datetime": {"nearest": "2022-08-29T1600", "tolerance_minutes": 30},
}

import numpy as np
import xarray as xr

vza = [0, 5, 10, 20, 30]
vaa = [83, 293]

vza_grid, vaa_grid = np.meshgrid(vza, vaa, indexing="ij")

reference_grid = xr.Dataset(
    data_vars={
        "viewing_zenith_angle": (
            ["series"],
            vza_grid.ravel(),
        ),
        "viewing_azimuth_angle": (
            ["series"],
            vaa_grid.ravel(),
        ),
    },
    coords={"series": range(len(vza_grid.ravel()))},
)

print(reference_grid)

hypernets_ds = read(L2b_path, vars_sel=vars_sel, subset=subset_dict,
                    processors={"populate_grid": {"reference_grid": reference_grid,
                                                  "dims_to_populate": ["viewing_zenith_angle", "viewing_azimuth_angle"],
                                                  }})

print(hypernets_ds)

fig, axs = plt.subplots(1, 3)

hypernets_ds.reflectance.plot.line(ax=axs[0], x="wavelength")
axs[0].set_ylabel("reflectance")
hypernets_ds.u_rel_systematic_reflectance.plot.line(ax=axs[1], x="wavelength", ylim=[0, 10])
axs[1].set_ylabel("systematic uncertainty (%)")
hypernets_ds.err_corr_systematic_reflectance.plot.imshow(ax=axs[2], cmap="gnuplot")

plt.show()
# plt.savefig("hypernets.png")

if __name__ == "__main__":
    pass
