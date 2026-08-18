"""eoio - Data readers and preprocessing for satellite data"""

# import h5py  # type: ignore

# from eoio._show_versions import show_versions
from eoio.interface import read, product_options  # (

#     product_bounds,
#     product_processors,
#     product_subsetting_params,
#     read,
#     write,
# )

__all__ = ["read", "write"]  # , "show_versions"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eoio")
except PackageNotFoundError:
    __version__ = "unknown"
