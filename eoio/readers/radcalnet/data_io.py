import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional

import obsarray  # noqa: F401 -- registers the `ds.unc` accessor used below

from eoio.readers.radcalnet.subset import RADCALNETSubset
from processor_tools.utils.formatters import datetime_from_yearday

# RadCalNet's own ascii format uses short, abbreviated column headers (its
# own convention, not eoio's -- see the RadCalNet ATBD); every other reader
# in eoio exposes descriptive snake_case variable names instead, so this
# reader translates at the parsing boundary. Maps eoio's dataset variable
# name -> the raw ascii column header it's read from.
AUX_COLUMN_NAMES = {
    "air_pressure": "P",
    "air_temperature": "T",
    "water_vapour": "WV",
    "ozone": "O3",
    "aerosol_optical_depth": "AOD",
    "angstrom_exponent": "Ang",
    "aerosol_type": "Type",
    "earth_sun_distance": "esd",
    "solar_zenith_angle": "Zen",
    "solar_azimuth_angle": "Azi",
}

# CF-style attrs for every variable this reader can produce. Values are taken
# from the RadCalNet ATBD's documented field units; kept here (rather than
# inline in read_file()) so they're easy to audit/correct in one place.
#
# Every entry carries a standard_name, even where CF has no matching term (aerosol_type,
# earth_sun_distance, local_time) -- BaseMetadataExtractor.get_variable_basic_metadata
# treats standard_name as a required basic-metadata key and warns for every variable
# missing one, so a var with no real CF equivalent falls back to its own eoio variable
# name, matching the fallback convention already used elsewhere in eoio (e.g.
# eoio.readers.sentinel2.metadata.extractor/eoio.readers.sentinel3_slstr.metadata.extractor,
# both of which set standard_name = var when no CF term applies) rather than leaving it
# unset.
VARIABLE_ATTRS: Dict[str, Dict[str, str]] = {
    # long_name/standard_name for "reflectance" are TOA-/BOA-specific and set by the reader
    # (RadCalNetReader vs. RadCalNetInputReader) rather than here, where that context isn't
    # available; units still apply either way.
    "reflectance": {"units": "1"},
    "reflectance_uncertainty": {
        "long_name": "Reflectance uncertainty",
        "standard_name": "reflectance_uncertainty",
        "units": "1",
    },
    "air_pressure": {"long_name": "Surface air pressure", "standard_name": "air_pressure", "units": "hPa"},
    "air_temperature": {"long_name": "Air temperature", "standard_name": "air_temperature", "units": "K"},
    "water_vapour": {
        "long_name": "Total column water vapour",
        "standard_name": "atmosphere_mass_content_of_water_vapor",
        "units": "g cm-2",
    },
    "ozone": {
        "long_name": "Total column ozone",
        "standard_name": "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
        "units": "DU",
    },
    "aerosol_optical_depth": {
        "long_name": "Aerosol optical depth at 550 nm",
        "standard_name": "atmosphere_optical_thickness_due_to_aerosol",
        "units": "1",
    },
    "angstrom_exponent": {
        "long_name": "Angstrom exponent",
        "standard_name": "angstrom_exponent_of_ambient_aerosol_in_air",
        "units": "1",
    },
    # Aerosol type is a site-specific categorical code (e.g. "R"); the RadCalNet ATBD doesn't
    # publish a single fixed code table for it, so no flag_values/flag_meanings are asserted
    # here rather than guessing an incorrect mapping. No CF standard_name for a categorical
    # aerosol-type code either -- falls back to the variable's own name (see module note above).
    "aerosol_type": {"long_name": "Aerosol type", "standard_name": "aerosol_type"},
    "earth_sun_distance": {"long_name": "Earth-Sun distance", "standard_name": "earth_sun_distance", "units": "AU"},
    "solar_zenith_angle": {"long_name": "Solar zenith angle", "standard_name": "solar_zenith_angle", "units": "degree"},
    "solar_azimuth_angle": {
        "long_name": "Solar azimuth angle",
        "standard_name": "solar_azimuth_angle",
        "units": "degree",
    },
    "local_time": {"long_name": "Local solar time", "standard_name": "local_time"},
}

# Suffix used for every uncertainty variable's dataset name, replacing the
# raw ascii file's own inconsistent "_unc" suffix (see AUX_COLUMN_NAMES).
UNCERTAINTY_SUFFIX = "_uncertainty"

_RAW_AUX_TO_NAME = {raw: name for name, raw in AUX_COLUMN_NAMES.items()}


# err_corr definitions below describe how each uncertainty component is correlated across
# its own dimensions, in obsarray's convention (see obsarray.err_corr.err_corr_forms --
# "random" = independent/uncorrelated, "systematic" = fully correlated). RadCalNet's own
# ATBD does not document a correlation model for these uncertainties, and the raw ascii
# product carries no correlation metadata at all -- there is nothing in eoio or the product
# itself to read this from. These forms are *inferred*, not authoritative:
#
# - across time: each of a day's 13 rows comes from its own independently-retrieved
#   atmospheric state (pressure/water vapour/ozone/AOD/Angstrom), so treated as "random"
#   (uncorrelated) between rows -- the more conservative default in the absence of a stated
#   model, and not contradicted by anything seen in RadCalNet's outputs or s2radval.
# - across wavelength (reflectance only): treated as "systematic" (fully correlated) --
#   this specifically matches s2radval's own explicit, documented choice when combining
#   RadCalNet's per-wavelength reflectance uncertainty across a sensor SRF (see
#   s2radval_matchup.py's rcn_interpolate(): "The RadCalNet uncertainty is also interpolated
#   and convolved. This assumes perfect correlation!!!"). That's itself s2radval's own
#   simplifying assumption rather than a documented RadCalNet error model, so it's carried
#   here as the closest thing to precedent, not as a verified fact.
#
# Anyone relying on these for real Monte Carlo uncertainty propagation (rather than as a
# placeholder pending that work) should confirm them against RadCalNet's error budget /
# ATBD first.
def _err_corr_time_random() -> List[Dict[str, Any]]:
    return [{"dim": ["time"], "form": "random", "params": [], "units": []}]


def _err_corr_reflectance() -> List[Dict[str, Any]]:
    return [
        {"dim": ["wavelength"], "form": "systematic", "params": [], "units": []},
        {"dim": ["time"], "form": "random", "params": [], "units": []},
    ]


def _uncertainty_attrs(base_name: str, err_corr: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Attrs for an aux variable's uncertainty counterpart: same units (these are absolute,
    not relative/percentage uncertainties -- same scale as the value they describe), a
    "... uncertainty" long_name, and a standard_name. CF has no standard convention for an
    uncertainty variable's own standard_name, so this falls back to the variable's own
    output name (``base_name`` + :py:data:`UNCERTAINTY_SUFFIX`), matching
    :py:data:`VARIABLE_ATTRS`'s own fallback convention for vars with no CF equivalent.

    :param err_corr: obsarray ``err_corr`` definition to attach (see the module-level note
        above) -- omitted for callers (e.g. existing tests) that don't need it.
    """
    base = VARIABLE_ATTRS.get(base_name, {})
    attrs: Dict[str, Any] = {}
    if "long_name" in base:
        attrs["long_name"] = f"{base['long_name']} uncertainty"
    if "units" in base:
        attrs["units"] = base["units"]
    attrs["standard_name"] = f"{base_name}{UNCERTAINTY_SUFFIX}"
    if err_corr is not None:
        attrs["err_corr"] = err_corr
    return attrs


def read_file(
    path: str,
    aux_vars: list[str],
) -> xr.Dataset:
    """
    Read RadCalNet file from ascii
    """
    DATA: Dict[str, Any] = {}
    f = open(path, "r")
    lines = f.read().split("\n")
    for line in lines:
        x = [item.strip().replace(":", "") for item in line.split("\t")]
        k = x[0] if x[0] not in DATA else f"{x[0]}_unc"
        try:
            DATA[k] = [float(_x) for _x in x[1:]]
        except ValueError:
            DATA[k] = x[1:]
    f.close()

    dims = ["wavelength", "time"]
    # Site/Lat/Lon/Alt are single-valued header lines, but DATA[k] always
    # parses to a list (see above) -- unwrap to scalars here, once, rather
    # than leaving every consumer to remember to index [0] itself (see
    # get_basic_metadata's WKT/product_bounds strings, which used to format
    # the list's repr directly into the text, e.g. "POINT ([15.12] [-23.6])"
    # instead of "POINT (15.12 -23.6)").
    attrs = {
        "Site": DATA.pop("Site")[0],
        "Lattitude": DATA.pop("Lat")[0],
        "Longitude": DATA.pop("Lon")[0],
        "Altitude": DATA.pop("Alt")[0],
    }
    data_vars: Dict[str, Any] = {
        "reflectance": (dims, [], VARIABLE_ATTRS.get("reflectance", {})),
    }
    # Collected here rather than added straight to data_vars -- unlike every other
    # variable in this dict, these are registered as obsarray uncertainty components
    # (via ds.unc[...], below) once the base dataset exists, not as plain data_vars.
    reflectance_unc_rows: List[Any] = []
    aux_unc: Dict[str, Any] = {}
    years = DATA.pop("Year")
    doy = DATA.pop("DOY(U)")
    doyl = DATA.pop("DOY(L)")
    utc = DATA.pop("UTC")
    local = DATA.pop("Local")
    time = [datetime_from_yearday(years[t], doy[t], utc[t]) for t in range(13)]
    l_time = [datetime_from_yearday(years[t], doyl[t], local[t]) for t in range(13)]
    wavelengths = []

    if "Zen" in DATA:
        data_vars["solar_zenith_angle"] = ("time", DATA.pop("Zen"), VARIABLE_ATTRS["solar_zenith_angle"])
    if "Azi" in DATA:
        data_vars["solar_azimuth_angle"] = (
            "time",
            np.array([row % 360 for row in DATA.pop("Azi")]),  # normalise to 360 degrees
            VARIABLE_ATTRS["solar_azimuth_angle"],
        )

    data_vars["local_time"] = (
        "time",
        l_time,
        VARIABLE_ATTRS["local_time"],
    )  # store local time as a data_var for later
    for aux in aux_vars:
        if "angle" not in aux:
            raw_key = AUX_COLUMN_NAMES.get(aux, aux)
            data_vars[aux] = ("time", DATA.pop(raw_key), VARIABLE_ATTRS.get(aux, {}))

    for k, val in DATA.items():
        if "unc" in k:
            if k.split("_")[0].isnumeric():
                reflectance_unc_rows.append(val)
            elif val:
                raw_base, _, _ = k.partition("_unc")
                base_name = _RAW_AUX_TO_NAME.get(raw_base, raw_base)
                aux_unc[base_name] = val
        elif k:
            wavelengths.append(float(k))
            data_vars["reflectance"][1].append(val)

    coords = {
        "wavelength": ("wavelength", wavelengths, {"long_name": "Wavelength", "units": "nm"}),
        "time": ("time", time),
    }
    ds = xr.Dataset(data_vars, coords, attrs=attrs)

    # Register uncertainty components via obsarray's `ds.unc` accessor (rather than as
    # plain data_vars) so they carry real, machine-readable err_corr metadata and are
    # discoverable as uncertainty (via ds.unc.unc_vars) rather than looking like ordinary
    # measurement variables -- see eoalign's uncertainty-detection utilities, which rely on
    # this to skip these vars rather than processing them as if they were data.
    if reflectance_unc_rows:
        ds.unc["reflectance"]["reflectance_uncertainty"] = (
            dims,
            np.array(reflectance_unc_rows),
            {**VARIABLE_ATTRS["reflectance_uncertainty"], "err_corr": _err_corr_reflectance()},
        )
    for base_name, val in aux_unc.items():
        # Guards against registering an uncertainty component for a base variable that
        # wasn't itself selected into aux_vars (obsarray requires the base var to exist)
        # -- the raw file always carries every aux uncertainty row regardless of what the
        # caller requested, so this can happen whenever aux_vars is a strict subset.
        if base_name not in ds:
            continue
        ds.unc[base_name][base_name + UNCERTAINTY_SUFFIX] = (
            ["time"],
            np.asarray(val),
            _uncertainty_attrs(base_name, _err_corr_time_random()),
        )

    return ds


def read_dataset(*, ds: xr.Dataset, include_vars: List[str], subset: RADCALNETSubset) -> xr.Dataset:
    """
    Read RadCalNet file into a dataset
    """

    ds = ds[include_vars]

    # series_indices/wavelength_indices are None when no subset criteria was
    # given for that dimension (e.g. build_subset() leaves series_indices
    # None when time_of_day_utc/time_of_day_local/angle/datetime are all
    # None -- the default since RadCalNetReader stopped hardcoding a
    # universal time_of_day_utc window). Passing None to isel() as an
    # indexer is not the same as omitting it -- xarray tries to use it as an
    # actual index and raises ("invalid indexer array, does not have
    # integer dtype") -- so only pass the dimensions that actually have a
    # resolved index array.
    isel_kwargs = {}
    if subset.series_indices is not None:
        isel_kwargs["time"] = subset.series_indices
    if subset.wavelength_indices is not None:
        isel_kwargs["wavelength"] = subset.wavelength_indices
    if isel_kwargs:
        ds = ds.isel(isel_kwargs)

    return ds
