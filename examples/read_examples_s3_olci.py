"""Example Sentinel-3 OLCI L1 reading"""

import socket
import os

from shapely import wkt
from matplotlib import pyplot as plt

from eoio import read


# Define input path from test datasets
# Note: this example product isn't ideal as the measurements in the radiance bands are odd (e.g. negative values) - use BT bands for demonstration
if socket.gethostname() == "lyon.npl.co.uk" or socket.gethostname() == "leipzig.npl.co.uk":
    DATA_DIRECTORY = r"/mnt/t/data/unittest_datasets"
else:
    DATA_DIRECTORY = r"T:\ECO\EOServer\data\unittest_datasets"
SEN3_PATH = os.path.join(
    DATA_DIRECTORY,
    "S3OLCI",
    "S3A_OL_1_EFR____20250712T110239_20250712T110539_20250713T115701_0179_128_094_1980_PS1_O_NT_004.SEN3",
)
print("SEN3 PATH:", SEN3_PATH)

# Define ROI
# lat_max = 58.0,
# lat_min = 54.0
# lon_max = 4.0
# lon_min = -4.0

roi_string = "POLYGON ((\
    -4.0 54.0,\
    -4.0 58.0,\
    4.0 58.0,\
    4.0 54.0,\
    -4.0 54.0\
))"

geom = wkt.loads(roi_string)

# Read dataset
ds = read(
    SEN3_PATH,
    vars_sel={
        "meas": ["Oa01", "Oa02"],  # "all"  ["Oa01", "Oa02"]
        "aux": ["observation_geometry", "solar_flux", "detector_index"],
        "mask": "all",
    },
    subset={"roi": geom, "roi_crs": "EPSG:4326"},
    read_params={"include_uncertainties": True},
)
for attr in ds.attrs:
    print(f"{attr}: {ds.attrs[attr]}")

for var in ds.data_vars:
    print(f"\nVariable: {var}")
    for attr in ds[var].attrs:
        print(f"  {attr}: {ds[var].attrs[attr]}")

# Make plot
ds["Oa01"].plot(robust=True)
plt.show()
