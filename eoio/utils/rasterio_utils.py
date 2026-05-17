"""eoio.utils.rasterio_utils - utilities for working with rasterio"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class RasterChunkHint:
    """Suggested chunk sizes for (y, x) dims."""

    y: int
    x: int
    block_y: int
    block_x: int
    approx_mb: float


def suggest_raster_chunks(
    path: str,
    *,
    target_mb: float = 32.0,
    max_mb: float = 128.0,
    min_blocks: int = 1,
) -> Dict[str, int]:
    """
    Suggest Dask chunk sizes aligned to the raster's internal block/tile size.

    :param path:
        Path to a raster readable by rasterio.
    :param target_mb:
        Approximate target chunk size in megabytes.
    :param max_mb:
        Upper bound on suggested chunk size in megabytes (prevents huge chunks).
    :param min_blocks:
        Minimum number of internal blocks per chunk in each dimension.
    :returns:
        A mapping suitable for ``open_rasterio(..., chunks=...)`` (keys ``"x"`` and ``"y"``).
    """
    import rasterio  # lazy-ish: only used if chunking enabled

    with rasterio.open(path) as src:
        # Some drivers (notably netCDF) present variables as subdatasets and
        # may report `count==0` with an empty `dtypes`. In that case open the
        # first subdataset and use its metadata instead.
        real_src = src
        if getattr(src, "count", 0) == 0 and getattr(src, "subdatasets", None):
            try:
                real_src = rasterio.open(src.subdatasets[0])
            except Exception:
                real_src = src

        # Prefer per-band block shape; fall back to profile keys if needed.
        if getattr(real_src, "block_shapes", None):
            block_y, block_x = real_src.block_shapes[0]
        else:
            block_x = int(real_src.profile.get("blockxsize", 256))
            block_y = int(real_src.profile.get("blockysize", 256))

        # Determine itemsize safely; fall back to 1 byte if unavailable.
        try:
            dtype = np.dtype(real_src.dtypes[0])
            itemsize = dtype.itemsize
        except Exception:
            itemsize = 1

        # Start with at least min_blocks blocks in each dimension
        chunk_y = block_y * max(1, min_blocks)
        chunk_x = block_x * max(1, min_blocks)

        # Bytes in current chunk (single band)
        def chunk_mb(cy: int, cx: int) -> float:
            return (cy * cx * itemsize) / (1024.0 * 1024.0)

        # Grow chunk by doubling along the larger dimension each time, but always in block multiples.
        # This is simple + robust.
        while chunk_mb(chunk_y, chunk_x) < target_mb:
            if chunk_x <= chunk_y:
                chunk_x *= 2
            else:
                chunk_y *= 2

            # Clamp if we exceed max_mb
            if chunk_mb(chunk_y, chunk_x) > max_mb:
                break

        # Ensure we don't exceed raster dimensions
        chunk_y = min(chunk_y, getattr(real_src, "height", chunk_y))
        chunk_x = min(chunk_x, getattr(real_src, "width", chunk_x))
        try:
            if real_src is not src:
                real_src.close()
        except Exception:
            pass

    return {"y": int(chunk_y), "x": int(chunk_x)}


def _first_path(path_like):
    """Return a usable first path from a path-like object.

    Handles mappings, lists/tuples and plain strings. Returns None when
    no usable path is found.
    """
    if not path_like:
        return None
    # dict-like with .values()
    if hasattr(path_like, "values"):
        vals = list(path_like.values())
        return vals[0] if vals else None
    # list/tuple
    if isinstance(path_like, (list, tuple)):
        return path_like[0] if len(path_like) > 0 else None
    # assume string / Path-like
    return path_like


if __name__ == "__main__":
    pass
