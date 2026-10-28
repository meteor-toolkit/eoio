"""Example dataset writing with eoio.write"""

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import eoio
from eoio.serialisation import decode_attrs
from eoio.writers import WriterFactory

# Write somewhere temporary so the example is runnable as-is. Replace with a real
# output directory to keep the results.
OUT_DIR = Path(tempfile.mkdtemp())
print("Writing to:", OUT_DIR)

# ---------------------------------------------------------------------------
# A dataset shaped like reader output
#
# In practice this would be `ds = eoio.read(PRODUCT_PATH)`. It is built by hand here so
# the example runs without a product to hand - the point is the attributes, which carry
# a nested metadata dict and a datetime, neither of which netCDF or Zarr can store.
# ---------------------------------------------------------------------------


def synthetic_scene(seed: int, shape: tuple = (100, 100)) -> np.ndarray:
    """
    A smooth field with a little noise on top.

    Deliberately not uniform random: real imagery is spatially correlated and therefore
    compressible, and using noise here would make the compression comparison below
    misleading (it comes out *larger*, because incompressible data still pays the
    overhead).
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    smooth = np.sin(x / 12.0) * np.cos(y / 15.0)
    return (0.25 + 0.15 * smooth + 0.01 * rng.standard_normal(shape)).astype("float32")


ds = xr.Dataset(
    data_vars={"B02": (("y", "x"), synthetic_scene(0))},
    coords={"x": np.arange(100), "y": np.arange(100)},
    attrs={
        "title": "Sentinel-2A MSI Level-1C",
        "platform": "Sentinel-2A",
        "instrument": "MSI",
        "product_version": "03.01",
        "date_created": dt.datetime.now(dt.timezone.utc),
        "product_metadata": {"datatake": {"id": "GS2A_20230702", "orbit": 65}},
    },
)
ds["B02"].attrs = {"units": "1", "measurand": "toa_reflectance"}

print("\nAvailable writers:", WriterFactory.list_writers())

# ---------------------------------------------------------------------------
# netCDF
# ---------------------------------------------------------------------------

nc_path = eoio.write(ds, OUT_DIR / "product.nc", writer="netcdf")
print("\nWrote netCDF:", nc_path)

# Attributes are encoded on the way out, so decode them on the way back in to get the
# Python objects rather than their serialised form.
with xr.open_dataset(nc_path) as opened:
    restored = decode_attrs(opened)

print("  product_metadata:", restored.attrs["product_metadata"])
print("  date_created is a", type(restored.attrs["date_created"]).__name__)
print("  history:", restored.attrs["history"].splitlines()[-1])

# Compression is on by default; compare against an uncompressed write.
plain = eoio.write(ds, OUT_DIR / "uncompressed.nc", writer="netcdf", write_params={"compress": False})
print(f"  compressed {Path(nc_path).stat().st_size} B vs uncompressed {Path(plain).stat().st_size} B")

# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------

zarr_path = eoio.write(ds, OUT_DIR / "product.zarr", writer="zarr", write_params={"chunks": {"x": 50, "y": 50}})
print("\nWrote Zarr:", zarr_path)

# ---------------------------------------------------------------------------
# Building a time series, one observation at a time
#
# The same call serves the first write and every later one: it creates the store if it
# is not there, and appends to it if it is. Only the new data is written each time.
# ---------------------------------------------------------------------------

stack = OUT_DIR / "stack.zarr"

for day in range(5):
    scene = xr.Dataset(
        {"B02": (("time", "y", "x"), synthetic_scene(day)[np.newaxis, ...])},
        coords={
            "time": [np.datetime64("2023-07-02") + np.timedelta64(day, "D")],
            "x": np.arange(100),
            "y": np.arange(100),
        },
        attrs={"title": "S2 time stack", "platform": "Sentinel-2A"},
    )
    eoio.write(scene, stack, writer="zarr", write_params={"append_dim": "time"})

with xr.open_zarr(stack) as stacked:
    print("\nStack grew to:", dict(stacked.sizes))
    print("  time steps:", stacked.time.values)

# ---------------------------------------------------------------------------
# Inspecting what a writer accepts
# ---------------------------------------------------------------------------

print("\nZarr write_params:")
for name, allowed in sorted(eoio.writer_options("zarr")["write_params"].items()):
    print(f"  {name:20s} {allowed}")
