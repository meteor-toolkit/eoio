"""read_examples_hypernets - Example usage script for interface functions"""

import os
import matplotlib.pyplot as plt
import configparser

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
    "angle": {"vza": {"nearest": 9.9, "tolerance": 1}, "vaa": {"min": 60, "max": 120}},
    "wavelength": {"nearest": 550, "tolerance": 10},
    "datetime": {"nearest": "2022-08-29T1600", "tolerance_minutes": 30},
}
hypernets_ds = read(example_data_path, vars_sel=vars_sel, subset=subset_dict)

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
