"""
eoio.readers.flex.variables
===========================

Catalogue of the variables in a FLEX L2 FLXSYN product, and how each is exposed by the reader.

The product stores variables in four NetCDF groups with product-specific dimension names
(``number_of_northing_pixels`` and so on). The reader renames these to ``y``/``x`` plus one
named dimension per spectral grid and per other trailing axis, since the OLCI, SLSTR, FLORIS,
irradiance, SIF and reflectance spectra are all on different wavelength grids:

======================  =========================================================
Dimension               Meaning
======================  =========================================================
``y``, ``x``            Product grid (northing, easting)
``wavelength_olci``     OLCI channels (21)
``wavelength_slstr``    SLSTR VSWIR channels (6)
``wavelength_floris``   FLORIS channels (580)
``wavelength_irradiance``  Irradiance spectral samples (605)
``wavelength_sif``      SIF emission spectral samples (111)
``wavelength_reflectance``  FLORIS "real" reflectance samples (281)
``instrument``          FLORIS, OLCI, SLSTR_nad, SLSTR_obl (view geometry only)
``sif_peak``            Red peak, far-red peak
``o2_band``             O2B (684 nm), O2A (761 nm)
======================  =========================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "SPATIAL_DIMS",
    "VarSpec",
    "VARIABLES",
    "WAVELENGTH_COORDS",
    "UNCERTAINTY_SUFFIX",
    "INSTRUMENTS",
    "SIF_PEAKS",
    "O2_BANDS",
    "MEAS_GROUPS",
    "AUX_GROUPS",
    "variables_for",
    "uncertainty_name",
]

#: product dimension name -> eoio dimension name, for the two spatial dimensions
SPATIAL_DIMS = {"number_of_northing_pixels": "y", "number_of_easting_pixels": "x"}

UNCERTAINTY_SUFFIX = "_uncertainty"

INSTRUMENTS = ["FLORIS", "OLCI", "SLSTR_nad", "SLSTR_obl"]
SIF_PEAKS = ["red", "far_red"]
O2_BANDS = ["O2B", "O2A"]


@dataclass(frozen=True)
class VarSpec:
    """
    How one product variable is read.

    :param group: NetCDF group the variable is stored in.
    :param kind: ``"meas"``, ``"aux"`` or ``"mask"``.
    :param dim: eoio name of the trailing (non-spatial) dimension, if any.
    :param has_uncertainty: whether the product carries an ``<name>_uncertainty`` partner.
    """

    group: str
    kind: str
    dim: Optional[str] = None
    has_uncertainty: bool = False


def _specs(group: str, kind: str, names: str, dim: Optional[str] = None, unc: bool = False) -> Dict[str, VarSpec]:
    return {n: VarSpec(group, kind, dim, unc) for n in names.split()}


ANC, ATM, FLU, VEG = "Ancillary_Information", "L2_Atmosphere", "L2_Fluorescence", "L2_Vegetation"

VARIABLES: Dict[str, VarSpec] = {
    # --- aux -------------------------------------------------------------------------
    **_specs(ANC, "aux", "elevation"),
    **_specs(ANC, "aux", "sun_zenith_angle viewing_zenith_angle relative_azimuth_angle", "instrument"),
    **_specs(
        ATM,
        "aux",
        "columnar_water_vapor aerosol_optical_thickness angstrom_exponent asymmetry_parameter "
        "cloud_probability land_surface_temperature integrated_par",
    ),
    **_specs(ATM, "aux", "direct_irradiance diffuse_irradiance", "wavelength_irradiance", unc=True),
    # --- mask ------------------------------------------------------------------------
    **_specs(ATM, "mask", "cirrus_cloud_mask"),
    # --- meas: apparent reflectance --------------------------------------------------
    **_specs(ATM, "meas", "olci_apparent_reflectance", "wavelength_olci"),
    **_specs(ATM, "meas", "slstr_apparent_reflectance", "wavelength_slstr"),
    **_specs(ATM, "meas", "floris_apparent_reflectance", "wavelength_floris", unc=True),
    # --- meas: fluorescence ----------------------------------------------------------
    **_specs(FLU, "meas", "sif_emission_spectrum", "wavelength_sif", unc=True),
    **_specs(FLU, "meas", "total_integrated_sif", unc=True),
    **_specs(FLU, "meas", "sif_peak_values sif_peak_positions", "sif_peak", unc=True),
    **_specs(FLU, "meas", "sif_O2_bands_value", "o2_band", unc=True),
    **_specs(FLU, "meas", "floris_real_reflectance", "wavelength_reflectance", unc=True),
    # --- meas: vegetation ------------------------------------------------------------
    **_specs(
        VEG,
        "meas",
        "leaf_area_index leaf_chlorophyll_content leaf_carotenoid_content fapar_chlorophyll "
        "apar_chlorophyll_photons fraction_of_absorbed_photosynthetically_active_radiation "
        "electron_transport_rate fluorescence_quantum_efficiency",
        unc=True,
    ),
    **_specs(VEG, "meas", "fluorescence_escape_probability", "wavelength_sif", unc=True),
    **_specs(VEG, "meas", "reversible_energy_dissipation PSI_PSII_relative_contribution"),
}

#: dimension -> (group, product variable holding its wavelengths)
WAVELENGTH_COORDS: Dict[str, tuple] = {
    "wavelength_olci": (ATM, "olci_spectral_channel_central_wavelengths"),
    "wavelength_slstr": (ATM, "slstr_vswir_spectral_channel_central_wavelengths"),
    "wavelength_floris": (ATM, "floris_spectral_channel_central_wavelengths"),
    "wavelength_irradiance": (ATM, "irradiance_wavelength_grid"),
    "wavelength_sif": (FLU, "sif_wavelength_grid"),
    "wavelength_reflectance": (FLU, "reflectance_wavelength_grid"),
}

#: named subsets of ``meas`` selectable as ``vars_sel={"meas": <name>}``
MEAS_GROUPS: Dict[str, List[str]] = {
    "olci": ["olci_apparent_reflectance"],
    "slstr": ["slstr_apparent_reflectance"],
    "floris": ["floris_apparent_reflectance", "floris_real_reflectance"],
    "fluorescence": [n for n, s in VARIABLES.items() if s.kind == "meas" and s.group == FLU],
    "vegetation": [n for n, s in VARIABLES.items() if s.kind == "meas" and s.group == VEG],
}

#: named subsets of ``aux`` selectable as ``vars_sel={"aux": <name>}``
AUX_GROUPS: Dict[str, List[str]] = {
    "atmosphere": [n for n, s in VARIABLES.items() if s.kind == "aux" and s.group == ATM],
    "irradiance": ["direct_irradiance", "diffuse_irradiance"],
    "geometry": ["sun_zenith_angle", "viewing_zenith_angle", "relative_azimuth_angle"],
}


def variables_for(kind: str) -> List[str]:
    """Names of all variables of the given kind (``"meas"``, ``"aux"`` or ``"mask"``)."""
    return [n for n, s in VARIABLES.items() if s.kind == kind]


def uncertainty_name(var: str) -> str:
    """Name of the uncertainty partner of ``var``."""
    return f"{var}{UNCERTAINTY_SUFFIX}"
