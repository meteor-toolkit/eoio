"""Example: read_multi for a set of Sentinel-2 tiles (multi-date, same tile).

Reads several Sentinel-2 L1C SAFE products over the same MGRS tile/ROI - one
per acquisition date - and concatenates them into a single time-stacked
reflectance cube via ``eoio.read_multi``.

Each file is first passed through the ``stack`` processor (run per file, via
``processors``) to build a per-scene ``(band, y, x)`` cube; ``read_multi``
then concatenates those cubes along a new ``time`` dimension, using each
scene's acquisition timestamp (``product_metadata.sensing_time``) as the
``concat_coord``.

Output dataset structure::

    reflectance_10m  (time, band_10m, y_10m, x_10m)   <- B02, B03, B04, B08

Fill in ``SAFE_PATHS`` below with real SAFE product paths over the same tile
before running - the list is empty by default so importing this module
doesn't do anything.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from shapely import wkt

from eoio import read_multi

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"

__all__ = []

# Fill in with several Sentinel-2 SAFE product paths over the *same* tile,
# one per acquisition date, e.g.:
#   "path/to/S2A_MSIL1C_20260701T111431_..._T30UXE_....SAFE",
#   "path/to/S2A_MSIL1C_20260706T111431_..._T30UXE_....SAFE",
#   "path/to/S2A_MSIL1C_20260711T111431_..._T30UXE_....SAFE",
SAFE_PATHS = [
    # "path/to/S2A_MSIL1C_..._T30UXE_....SAFE",
]

# Region of interest, reused across every date (same tile -> same ROI is
# valid for all of them)
ROI_WKT = "POLYGON ((\
 -0.785469 53.706338,\
 -0.519964 53.701108,\
 -0.531634 53.545207,\
 -0.791044 53.550248,\
 -0.785469 53.706338\
))"

BANDS = ["B02", "B03", "B04", "B08"]

if __name__ == "__main__":
    if not SAFE_PATHS:
        print(
            "SAFE_PATHS is empty - fill it in with real Sentinel-2 SAFE paths over the same tile to run this example."
        )
        raise SystemExit(0)

    roi = wkt.loads(ROI_WKT)

    ds = read_multi(
        SAFE_PATHS,
        vars_sel={"meas": BANDS},
        subset={"roi": roi, "roi_crs": 4326},
        processors={"stack": {"stack_dim": "band"}},  # per-scene band cube, run before concatenation
        concat_dim="time",
        concat_coord="product_metadata.sensing_time",  # per-file attr -> time coord
        on_mismatch="error",  # raise if a scene's post-subset grid doesn't match the others
    )

    # ------------------------------------------------------------------ #
    # Dataset overview
    # ------------------------------------------------------------------ #
    print(f"Concatenated {ds.sizes['time']} scenes\n")
    print("Dataset attributes:")
    for k, v in ds.attrs.items():
        print(f"  {k}: {v}")

    cube = ds["reflectance_10m"]
    print(f"\n{cube.dims} {cube.shape}")
    print("Acquisition times:", list(ds["time"].values))

    # ------------------------------------------------------------------ #
    # Plot: reflectance time series for each band at a sample pixel
    # ------------------------------------------------------------------ #
    band_dim = [d for d in cube.dims if d.startswith("band")][0]
    ny, nx = cube.sizes["y_10m"], cube.sizes["x_10m"]
    py, px = ny // 2, nx // 2

    fig, ax = plt.subplots(figsize=(8, 4))
    for band in cube.coords[band_dim].values:
        series = cube.sel({band_dim: band}).isel(y_10m=py, x_10m=px).values
        ax.plot(ds["time"].values, series, marker="o", label=str(band))
    ax.set_xlabel("Acquisition time")
    ax.set_ylabel("TOA reflectance")
    ax.set_title(f"Sentinel-2 — pixel ({py}, {px}) time series")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    plt.tight_layout()
    out_dir = Path(SAFE_PATHS[0]).parent
    plt.savefig(out_dir / "s2_read_multi_timeseries.png", dpi=150)
    print(f"Saved: {out_dir / 's2_read_multi_timeseries.png'}")
    plt.show()

    # ------------------------------------------------------------------ #
    # Plot: RGB composite per date (B04=Red, B03=Green, B02=Blue)
    # ------------------------------------------------------------------ #
    def normalise(arr, lo=2, hi=98):
        """Percentile stretch to [0, 1], NaN -> 0."""
        vmin, vmax = np.nanpercentile(arr, lo), np.nanpercentile(arr, hi)
        out = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
        out[~np.isfinite(arr)] = 0
        return out

    n_time = ds.sizes["time"]
    fig, axes = plt.subplots(1, n_time, figsize=(4 * n_time, 4))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, range(n_time)):
        red = cube.sel({band_dim: "B04"}).isel(time=t).values
        green = cube.sel({band_dim: "B03"}).isel(time=t).values
        blue = cube.sel({band_dim: "B02"}).isel(time=t).values
        rgb = np.dstack([normalise(red), normalise(green), normalise(blue)])
        ax.imshow(rgb, origin="upper")
        ax.set_title(str(ds["time"].values[t])[:10])
        ax.axis("off")
    plt.suptitle("Sentinel-2 — RGB per acquisition (B04/B03/B02, 2–98 % stretch)")
    plt.tight_layout()
    plt.savefig(out_dir / "s2_read_multi_rgb.png", dpi=150)
    print(f"Saved: {out_dir / 's2_read_multi_rgb.png'}")
    plt.show()
