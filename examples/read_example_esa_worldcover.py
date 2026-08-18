
"""read_examples_esa_worldcover- Example usage script for interface functions"""
import os
import configparser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from eoio import read
PLOT=True

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "eoio",
    "etc",
    "test_file_paths.config",
)

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

example_data_path = config.get("ESAWorldCover", "esaworldcover_tif", fallback=None)

# Read dataset
ds = read(
    example_data_path,
     subset={
        "roi": ((9, 9), 500),
        "roi_crs": 4326,
   },

)

print(ds)

def plot_landcover(
    ds,
    ax=None,
    add_legend=True,
    title="ESA WorldCover",
    figsize=(10, 8),
):
    """
    Plot ESA WorldCover land cover map from an xarray Dataset.
    """

    da = ds["landcover_map"]

    flag_values = da.attrs["flag_values"]
    flag_colors = da.attrs["flag_colors"]

    meanings = da.attrs["flag_meanings"]
    if isinstance(meanings, str):
        flag_meanings = meanings.split()
    else:
        flag_meanings = meanings

    cmap = ListedColormap(flag_colors)

    value_to_index = {
        value: idx
        for idx, value in enumerate(flag_values)
    }

    data = da.values
    indexed = np.full(data.shape, np.nan)

    for value, idx in value_to_index.items():
        indexed[data == value] = idx

    norm = BoundaryNorm(
        np.arange(len(flag_values) + 1) - 0.5,
        cmap.N,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(
        indexed,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(title)

    if {"x", "y"}.issubset(da.coords):
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if add_legend:
        handles = [
            Patch(
                facecolor=color,
                edgecolor="black",
                label=f"{value}: {meaning.replace('_', ' ').title()}",
            )
            for value, meaning, color in zip(
                flag_values,
                flag_meanings,
                flag_colors,
            )
        ]

        ax.legend(
            handles=handles,
            title="Land Cover",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
        )

    plt.tight_layout()

    return fig, ax

if PLOT:
   plot_landcover(ds)
   plt.show()