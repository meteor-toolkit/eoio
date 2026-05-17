"""Example Sentinel-2 MSI L1c reading"""

from shapely import wkt
from eoio import read
from matplotlib import pyplot as plt
import time

# Define input path from test datasets
SAFE_PATH = (
    r"T:\ECO\EOServer\data\unittest_datasets\S2MSIL1C\S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"
)
print("SAFE PATH:", SAFE_PATH)

# Define ROI
roi_string = "POLYGON ((\
 -0.913910 53.786950,\
 -0.382900 53.776490,\
 -0.406240 53.464690,\
 -0.925060 53.474770,\
 -0.913910 53.786950\
))"

geom = wkt.loads(roi_string)


# start timer for read
t_0 = time.time()
# Read dataset
ds = read(
    SAFE_PATH,
    vars_sel={
        "meas": ["B01", "B02", "B03", "B04"],
        "aux":
        # "all",
        ["solar_zenith_angle", "viewing_zenith_angle_B03", "tcwv"],
        # "all"  #{"observation_geometry": True} #["viewing_zenith_angle_B03", "tcwv"]
    },
    subset={"roi": geom, "roi_crs": "EPSG:4326"},
    read_params={"use_chunks": True, "ave_va_det": True},
    processors={
        # 'interpolate': {'coords':['x_5000m', 'y_5000m'],
        #                             'target_grid': ['x_60m', 'y_60m'],
        #                             'data_vars': ['solar_zenith_angle'],
        #                             },
        #             'interpolate': {'coords':['x_5000m', 'y_5000m'],
        #                             'target_grid': ['x_10m', 'y_10m'],
        #                             'data_vars': ['solar_zenith_angle']
        #                             },
        #             'interpolate': {'coords':['x_5000m', 'y_5000m'],
        #                             'target_grid': ['x_20m', 'y_20m'],
        #                             'data_vars': ['solar_zenith_angle']
        #                             },
        "units.convert": {
            "to": "radiance",
            "var_names": ["B01", "B02", "B03", "B04"],  #'all'
        }
    },
)

for attr in ds.attrs:
    print(f"{attr}: {ds.attrs[attr]}")

for var in ds.data_vars:
    print(f"\nVariable: {var}")
    for attr in ds[var].attrs:
        print(f"  {attr}: {ds[var].attrs[attr]}")

# stop timer for read and print
print("READ TIME: " + f"{time.time() - t_0}")

# Make plot
ds["B02"].plot(robust=True)
plt.show()
