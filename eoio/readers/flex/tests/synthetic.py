"""Small synthetic FLEX L2 FLXSYN product for tests, mirroring the real group/dimension layout."""

from __future__ import annotations
import numpy as np
import xarray as xr
from pyproj import Transformer

from eoio.readers.flex.variables import ANC, ATM, FLU, VEG, VARIABLES, WAVELENGTH_COORDS

N = 12  # grid size
PIXEL = 300.0

#: eoio trailing dim -> (product dim name, size)
TRAILING = {
    "instrument": ("number_of_instruments", 4),
    "wavelength_olci": ("number_of_olci_spectral_channels", 5),
    "wavelength_slstr": ("number_of_slstr_vswir_spectral_channels", 3),
    "wavelength_floris": ("number_of_apparent_reflectance_spectral_channels", 8),
    "wavelength_irradiance": ("number_of_par_spectral_samples", 6),
    "wavelength_sif": ("number_of_sif_spectral_samples", 7),
    "wavelength_reflectance": ("number_of_real_reflectance_spectral_samples", 4),
    "sif_peak": ("number_of_sif_peaks", 2),
    "o2_band": ("number_of_sif_o2_values", 2),
}
WAVELENGTHS = {
    "wavelength_olci": np.linspace(400.0, 1000.0, 5),
    "wavelength_slstr": np.array([555.0, 659.0, 865.0]),
    "wavelength_floris": np.linspace(680.0, 780.0, 8),
    "wavelength_irradiance": np.linspace(400.0, 780.0, 6),
    "wavelength_sif": np.linspace(670.0, 780.0, 7),
    "wavelength_reflectance": np.linspace(500.0, 780.0, 4),
}
SPATIAL = ("number_of_northing_pixels", "number_of_easting_pixels")
N_ACROSS = 3


def write_synthetic_flex(path, epsg: int = 32631, x0: float = 400020.0, y0: float = 4700040.0, seed: int = 0) -> None:
    """
    Write a synthetic product containing every variable the reader knows about.

    Spectral cubes are float, except ``olci_apparent_reflectance`` which is packed as ushort with
    a scale factor and fill value (like the real product) to exercise unpacking. Pixel (0, 0)
    of every variable is set to NaN/fill.
    """
    rng = np.random.default_rng(seed)
    xr.Dataset(
        attrs={
            "Product_level": "L2",
            "GeoTransform": f"{x0:f} {PIXEL:f} 0.000000 {y0:f} 0.000000 {-PIXEL:f}",
            "spatial_ref": f"EPSG:{epsg}",
            "creation_time": "UTC=2026-03-17T11.24.40.470",
        }
    ).to_netcdf(path, mode="w")

    x = x0 + (np.arange(N) + 0.5) * PIXEL
    y = y0 - (np.arange(N) + 0.5) * PIXEL
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(xx, yy)

    groups: dict = {g: {} for g in (ANC, ATM, FLU, VEG)}
    groups[ANC]["latitude"] = xr.DataArray(lat.astype("f4"), dims=SPATIAL, attrs={"units": "degrees_north"})
    groups[ANC]["longitude"] = xr.DataArray(lon.astype("f4"), dims=SPATIAL, attrs={"units": "degrees_east"})

    encodings: dict = {g: {} for g in groups}
    for name, spec in VARIABLES.items():
        dims = SPATIAL + ((TRAILING[spec.dim][0],) if spec.dim else ())
        shape = (N, N) + ((TRAILING[spec.dim][1],) if spec.dim else ())
        if spec.kind == "mask":
            groups[spec.group][name] = xr.DataArray(np.zeros(shape, "u1"), dims=dims)
            continue
        data = rng.uniform(0.1, 0.9, shape).astype("f4")
        data[0, 0] = np.nan
        attrs = {"long_name": name.replace("_", " "), "standard_name": name, "units": "-"}
        groups[spec.group][name] = xr.DataArray(data, dims=dims, attrs=attrs)
        if name == "olci_apparent_reflectance":
            encodings[spec.group][name] = {"dtype": "u2", "scale_factor": 1.52592547379986e-05, "_FillValue": 65535}
        if spec.has_uncertainty:
            unc_dims = dims
            if name == "sif_O2_bands_value":
                # the real product dimensions the O2 uncertainty with the peaks dimension
                unc_dims = SPATIAL + ("number_of_sif_peaks",)
            groups[spec.group][f"{name}_uncertainty"] = xr.DataArray(
                data * 0.1, dims=unc_dims, attrs={"long_name": f"{name} uncertainty", "units": "-"}
            )

    for dim, (group, name) in WAVELENGTH_COORDS.items():
        wvl = WAVELENGTHS[dim]
        pdim = TRAILING[dim][0]
        if dim == "wavelength_floris":
            # real product: (across_track, channel) with only the first sample populated
            vals = np.full((N_ACROSS, wvl.size), np.nan, "f4")
            vals[0] = wvl
            groups[group][name] = xr.DataArray(vals, dims=("number_of_across_track_samples", pdim))
        else:
            groups[group][name] = xr.DataArray(wvl.astype("f4"), dims=(pdim,), attrs={"long_name": name})

    for group, variables in groups.items():
        xr.Dataset(variables).to_netcdf(path, mode="a", group=group, encoding=encodings[group])
