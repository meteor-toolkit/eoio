from eoio.interface import read

# FLEX L2 FLXSYN product. Every science variable is "meas"; retrieved atmosphere/irradiance/
# geometry are "aux"; the product's cirrus cloud flag is the only "mask".
# Each spectral grid keeps its own wavelength dimension (wavelength_olci, wavelength_floris, ...),
# so the resample_wavelength processor puts spectra from different sensors on one common grid.
ds = read(
    r"flx_l2__flxsyn_20190930T102242_20190930T102254_20260312T190507_31TDG_100_036_01.nc",
    vars_sel={
        "meas": ["floris_apparent_reflectance", "olci_apparent_reflectance", "total_integrated_sif"],
        "aux": ["sun_zenith_angle", "cloud_probability"],
        "mask": "all",
    },
    subset={
        "roi": (2.2, 41.8, 2.5, 42.0),  # lon/lat bounding box
        "roi_crs": 4326,
        "wavelength": {"min": 600, "max": 780},
    },
    read_params={"include_uncertainties": True},
    processors={
        "resample_wavelength": {
            "target_wavelength": {"min": 680, "max": 780, "step": 1},
            "drop_original": True,
        }
    },
)

print(ds)
