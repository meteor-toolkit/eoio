"""
eoio.readers.flex.layout
========================

Filename parsing for FLEX products.

FLEX L2 products are single NetCDF files named::

    flx_l2__flxsyn_<start>_<stop>_<created>_<tile>_<n1>_<n2>_<n3>.nc

for example ``flx_l2__flxsyn_20190930T102242_20190930T102254_20260312T190507_31TDG_100_036_01.nc``.

Only the L2 ``FLXSYN`` product is supported at present; the level and product type are parsed
generically so further levels/products can be registered in :py:data:`SUPPORTED_PRODUCTS`.

Classes
-------
.. autosummary::
   :toctree: generated/

   FlexLayout
"""

from __future__ import annotations
import datetime as dt
import re
from pathlib import Path

__all__ = ["FlexLayout", "SUPPORTED_PRODUCTS"]

#: (processing level, product type) combinations the FLEX reader can read.
SUPPORTED_PRODUCTS = {("L2", "FLXSYN")}

_FILENAME_PATTERN = re.compile(
    r"^flx_(?P<level>l\d[a-z]?)_+(?P<product>[a-z0-9]+)_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<created>\d{8}T\d{6})_"
    r"(?P<tile>[0-9A-Z]+)_(?P<n1>\d+)_(?P<n2>\d+)_(?P<n3>\d+)\.nc$",
    re.IGNORECASE,
)
_TIME_FORMAT = "%Y%m%dT%H%M%S"


class FlexLayout:
    """
    Parsed FLEX product filename.

    :param path: path to the FLEX product file.
    :raises ValueError: if the filename is not a recognised FLEX product name, or is a
        FLEX product type that is not yet supported.

    Attributes
    ----------
    path : str
        Path to the product.
    level : str
        Processing level, e.g. ``"L2"``.
    product_type : str
        Product type, e.g. ``"FLXSYN"``.
    start_time, stop_time, creation_time : datetime.datetime
        Sensing start/stop and processing time parsed from the filename (UTC).
    tile : str
        MGRS tile identifier, e.g. ``"31TDG"``.
    numbers : tuple of int
        The three trailing numeric fields of the filename (e.g. ``(100, 36, 1)``), kept as parsed.
    """

    def __init__(self, path: Path | str):
        self.path = str(path)
        name = Path(self.path).name
        match = _FILENAME_PATTERN.match(name)
        if match is None:
            raise ValueError(f"'{name}' is not a recognised FLEX product filename")

        self.level = match["level"].upper()
        self.product_type = match["product"].upper()
        if (self.level, self.product_type) not in SUPPORTED_PRODUCTS:
            raise ValueError(
                f"Unsupported FLEX product {self.level} {self.product_type}; supported: "
                + ", ".join(f"{lvl} {prod}" for lvl, prod in sorted(SUPPORTED_PRODUCTS))
            )

        self.start_time = self._parse_time(match["start"])
        self.stop_time = self._parse_time(match["stop"])
        self.creation_time = self._parse_time(match["created"])
        self.tile = match["tile"].upper()
        self.numbers = (int(match["n1"]), int(match["n2"]), int(match["n3"]))

    @staticmethod
    def _parse_time(value: str) -> dt.datetime:
        return dt.datetime.strptime(value, _TIME_FORMAT).replace(tzinfo=dt.timezone.utc)

    @property
    def name(self) -> str:
        """Filename of the product."""
        return Path(self.path).name

    @property
    def processing_level(self) -> int:
        """Numeric processing level, e.g. ``2`` for L2."""
        return int(self.level[1])


if __name__ == "__main__":
    pass
