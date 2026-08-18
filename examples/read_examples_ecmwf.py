"""read_examples_ecmwf - Example usage script for ECMWF products"""

import os
import configparser

from shapely import wkt
from eoio.interface import read, product_processors, product_options

__author__ = "MetEOR Toolkit Team"

__all__ = []

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "eoio",
    "etc",
    "test_file_paths.config",
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

ERA_PATH = config.get("ECMWF", "ERA_file", fallback=None)
CAMS_PATH = config.get("ECMWF", "CAMS_file", fallback=None)

print("ERA_PATH:", ERA_PATH)
print("CAMS_PATH:", CAMS_PATH)

if ERA_PATH is None:
    raise RuntimeError("ERA_file is not configured in examples/eoi/etc/test_file_paths.config")

if CAMS_PATH is None:
    raise RuntimeError("CAMS_file is not configured in examples/eoi/etc/test_file_paths.config")

# Example: inspect available options for the ERA product
era_subset_params = product_options(ERA_PATH)
print("ERA subset params:\n", era_subset_params)
era_process_params = product_processors(ERA_PATH)
print("ERA process params:\n", era_process_params)

# Example read for ERA5 netCDF
era_ds = read(
    ERA_PATH,
    vars_sel={"meas": "all"},
    read_params={"metadata_level": "all", "save_extracted": False},
)

print("ERA dataset:", era_ds)
print("ERA variables:", list(era_ds.data_vars))

# Example read for CAMS data (zip archive)
cams_ds = read(
    CAMS_PATH,
    vars_sel={"meas": "all"},
    read_params={"metadata_level": "all", "save_extracted": False},
)

print("CAMS dataset:", cams_ds)
print("CAMS variables:", list(cams_ds.data_vars))

if __name__ == "__main__":
    pass
