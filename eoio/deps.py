"""eoio.deps - Lazy imports for optional dependencies."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyproj  # noqa: F401
    import rasterio  # noqa: F401
    import rioxarray  # noqa: F401
    import cfgrib  # noqa: F401
    from osgeo import gdal, ogr, osr  # noqa: F401
    import shapely  # noqa: F401


def _require_extra(extra: str, import_name: str) -> None:
    raise ModuleNotFoundError(
        f"Optional dependency '{import_name}' is required for this operation. Install with: pip install eoio[{extra}]"
    )


def lazy_pyproj():
    # Work around a macOS PROJ 9.7 issue where pyproj emits a UserWarning
    # ("pyproj unable to set PROJ database path") during import because the
    # PROJ database path is not in the environment when network.py tries to
    # initialise the CA-bundle context.  Set PROJ_DATA *before* importing so
    # pyproj finds the database on first load.
    # Linux/macOS conda: <prefix>/share/proj; Windows conda: <prefix>/Library/share/proj
    import os
    import sys

    _candidates = [
        os.path.join(sys.prefix, "share", "proj"),
        os.path.join(sys.prefix, "Library", "share", "proj"),
    ]
    proj_data = next(
        (c for c in _candidates if os.path.exists(os.path.join(c, "proj.db"))),
        os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"),
    )
    if proj_data:
        os.environ.setdefault("PROJ_DATA", proj_data)

    try:
        import pyproj  # type: ignore
    except ModuleNotFoundError:
        _require_extra("geo", "pyproj")

    if proj_data:
        try:
            pyproj.datadir.set_data_dir(proj_data)
        except Exception:
            pass
    return pyproj


def lazy_shapely():
    try:
        from shapely.geometry import Polygon, box, mapping, shape  # type: ignore
        from shapely.ops import transform as shp_transform  # type: ignore
    except ModuleNotFoundError:
        _require_extra("geo", "shapely")
    return Polygon, box, mapping, shape, shp_transform


def lazy_rasterio():
    try:
        import rasterio  # type: ignore
    except ModuleNotFoundError:
        _require_extra("raster", "rasterio")
    return rasterio


def lazy_rioxarray():
    try:
        import rioxarray as rxr  # type: ignore
    except ModuleNotFoundError:
        _require_extra("raster", "rioxarray")
    return rxr


def lazy_cfgrib():
    try:
        import cfgrib  # type: ignore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Dependency 'cfgrib' is required for this operation."
            "Please see the 'cfgrib' instructions at: https://github.com/ecmwf/cfgrib?tab=readme-ov-file#installation"
        )
    return cfgrib


if __name__ == "__main__":
    pass
