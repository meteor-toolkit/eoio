"""Example Sentinel-3 SLSTR L1B reading"""

from shapely import wkt
from eoio import read
from matplotlib import pyplot as plt


# Define input path from test datasets
# Note: this example product isn't ideal as the measurements in the radiance bands are odd (e.g. negative values) - use BT bands for demonstration
SEN3_PATH = r"T:\ECO\EOServer\data\unittest_datasets\S3SLSTR\S3B_SL_1_RBT____20230819T211849_20230819T212149_20230819T235829_0179_083_086_0900_PS2_O_NR_004.SEN3"
print("SEN3 PATH:", SEN3_PATH)

# Define ROI
# lat_max = 60.0,
# lat_min = 56.0
# lon_max = 2.0
# lon_min = -2.0

# Define ROI
roi_string = "POLYGON ((\
    -2.0 56.0,\
    -2.0 60.0,\
    2.0 60.0,\
    2.0 56.0,\
    -2.0 56.0\
))"

geom = wkt.loads(roi_string)


# Read dataset
ds = read(
    SEN3_PATH,
    vars_sel={
        "meas": ["S8_BT_in", "S9_BT_in"],
        "aux": ["observation_geometry"],
        "mask": ["cloud"],
    },
    subset={"roi": geom, "roi_crs": "EPSG:4326"},
    read_params=None,
)

# Make plot
ds["S7_BT_in"].plot(robust=True)
plt.show()
