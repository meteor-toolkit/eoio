"""Example Sentinel-2 MSI L1c reading"""

from eoio import read
from matplotlib import pyplot as plt

# Define input path from test datasets
SAFE_PATH = "T:/ECO/EOServer/data/product_archive/S2A/S2_MSI_L2A/2023/07/18/S2A_MSIL2A_20230718T084601_N0510_R107_T33KWP_20240816T230324.SAFE"
SAFE_PATH = "T:/ECO/EOServer/data\satellite\S2A_MSI\L2\FRM4FLUO\S2A_MSIL2A_20250613T100701_N0511_R022_T32TPN_20250613T121215.SAFE"

# Define ROI
roi_string = "POLYGON ((\
 15.25 -23.5,\
 15.25 -23.75,\
 15 -23.75,\
 15 -23.5,\
 15.25 -23.5 \
))"

geom = ((15.2, -23.6), 100)  # wkt.loads(roi_string)
geom = ((669601.078305950504728, 4744936.569929973222315), 150)  # UTM coordinates

# Read dataset
ds = read(
    SAFE_PATH,
    vars_sel={
        "meas": "all",
        "aux": ["viewing_zenith_angle_B03"],
    },
    subset={"roi": geom, "roi_crs": "EPSG:32632"},
    read_params={"use_chunks": True, "preferred_resolution": 20},
    processors={"add_lat_lon": {}},
)

# Make plot
ds["B02"].plot(robust=True)
plt.show()
plt.clf()
ds[["B04", "B03", "B02"]].to_array().plot.imshow(robust=True)
plt.show()

print(ds)
