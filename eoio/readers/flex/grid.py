"""
eoio.readers.flex.grid
======================

Product grid of a FLEX product, derived from the global ``GeoTransform`` and ``spatial_ref``
attributes (the product has no x/y coordinate variables). The CRS differs between products
(the UTM zone follows the scene), so it is always read from the file.

Classes
-------
.. autosummary::
   :toctree: generated/

   FlexGrid
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Tuple
import numpy as np

from eoio.deps import lazy_pyproj

__all__ = ["FlexGrid"]


@dataclass(frozen=True)
class FlexGrid:
    """
    North-up regular grid in a projected CRS.

    :param x0: x coordinate of the upper-left corner of the upper-left pixel.
    :param y0: y coordinate of the upper-left corner of the upper-left pixel.
    :param dx: pixel width (positive).
    :param dy: pixel height (positive, y decreases with row index).
    :param nx: number of columns.
    :param ny: number of rows.
    :param crs: CRS as a string, e.g. ``"EPSG:32631"``.
    """

    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int
    crs: str

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any], nx: int, ny: int) -> "FlexGrid":
        """
        Build from the global attributes of a FLEX product.

        :param attrs: global attributes (must contain ``GeoTransform`` and ``spatial_ref``).
        :param nx: size of the easting dimension.
        :param ny: size of the northing dimension.
        :raises ValueError: if the grid attributes are missing or the grid is rotated/south-up.
        """
        try:
            gt = [float(v) for v in str(attrs["GeoTransform"]).split()]
            crs = str(attrs["spatial_ref"])
        except KeyError as e:
            raise ValueError(f"FLEX product is missing grid attribute {e}") from e
        if len(gt) != 6:
            raise ValueError(f"Unexpected GeoTransform {attrs['GeoTransform']!r}")
        x0, dx, rot_x, y0, rot_y, dy = gt
        if rot_x != 0 or rot_y != 0 or dx <= 0 or dy >= 0:
            raise ValueError(f"Only north-up, unrotated grids are supported, got GeoTransform {gt}")
        return cls(x0=x0, y0=y0, dx=dx, dy=-dy, nx=nx, ny=ny, crs=crs)

    @property
    def x(self) -> np.ndarray:
        """Pixel-centre easting coordinates."""
        return self.x0 + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        """Pixel-centre northing coordinates (decreasing)."""
        return self.y0 - (np.arange(self.ny) + 0.5) * self.dy

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Outer ``(xmin, ymin, xmax, ymax)`` bounds in the grid CRS."""
        return (self.x0, self.y0 - self.ny * self.dy, self.x0 + self.nx * self.dx, self.y0)

    @property
    def epsg(self) -> int:
        """EPSG code of the grid CRS."""
        code = lazy_pyproj().CRS.from_user_input(self.crs).to_epsg()
        if code is None:
            raise ValueError(f"CRS {self.crs!r} has no EPSG code")
        return code


if __name__ == "__main__":
    pass
