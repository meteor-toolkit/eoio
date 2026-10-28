"""eoio - Data readers and preprocessing for satellite data"""

# import h5py  # type: ignore

# from eoio._show_versions import show_versions
from eoio import (
    serialisation,
    writers,
)
from eoio.interface import product_options, read, read_multi, write, writer_options

__all__ = [
    "product_options",
    "read",
    "read_multi",
    "serialisation",
    "write",
    "writer_options",
    "writers",
]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eoio")
except PackageNotFoundError:
    __version__ = "unknown"
