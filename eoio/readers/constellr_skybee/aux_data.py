"""eoio.readers.constellr_skybee.aux_data - auxiliary data reading functions for Skybee reader."""

from __future__ import annotations

import re
import warnings

import xarray as xr

from eoio.readers.constellr_skybee.metadata import ConstellrSkybeeMetadataExtractor

#: Attributes of the scene-level angle variables ``add_angles`` creates. Names and attributes follow
#: the PlanetScope / Landsat readers (``viewing_*`` names, CF ``sensor_*`` standard names), which is
#: what eoalign's ``brdf_correction`` looks for. The nested ``product_metadata.geometry_id`` is what
#: ``to_datatree`` reads to place these scalars in a resolution node next to the reflectance (they have
#: no x/y dims to infer it from). No ``measurand`` on purpose: the stack processor would merge all
#: four into one ``angle`` variable.
_ANGLE_ATTRS = {
    "solar_zenith_angle": {
        "long_name": "Solar Zenith Angle",
        "standard_name": "solar_zenith_angle",
        "description": "Angle between the line of sight to the sun and the local vertical.",
        "source_variable_name": "90 - view:sun_elevation",
    },
    "solar_azimuth_angle": {
        "long_name": "Solar Azimuth Angle",
        "standard_name": "solar_azimuth_angle",
        "description": "Horizontal angle between the line of sight to the sun and north, measured clockwise.",
        "source_variable_name": "view:sun_azimuth",
    },
    "viewing_zenith_angle": {
        "long_name": "Viewing Zenith Angle",
        "standard_name": "sensor_zenith_angle",
        "description": "Angle between the line of sight to the sensor and the local zenith at the observation target.",
        "source_variable_name": "view:incident_angle",
    },
    "viewing_azimuth_angle": {
        "long_name": "Viewing Azimuth Angle",
        "standard_name": "sensor_azimuth_angle",
        "description": "Horizontal angle between the line of sight to the sensor and north, measured clockwise.",
        "source_variable_name": "view:azimuth",
    },
}


def read_aux(ds: xr.Dataset, aux: list[str] | None, mtd: ConstellrSkybeeMetadataExtractor):
    if aux is None:
        raise ValueError("No auxiliary data files have been selected. Please set aux before reading.")

    if "observation_geometry" in aux:
        ds = add_angles(ds, mtd)

    unimplemented = sorted(set(aux) - {"observation_geometry"})
    if unimplemented:
        warnings.warn(f"Auxiliary data {unimplemented} is not yet implemented for constellr Skybee data.")

    return ds


def _grid_resolutions(ds: xr.Dataset) -> list[str]:
    """Resolutions (e.g. ``["10m", "20m"]``) of the pixel grids ``ds`` holds, from its ``x_<res>m`` dims."""
    found = {m.group(1) for dim in ds.dims if (m := re.fullmatch(r"x_(\d+m)", str(dim)))}
    return sorted(found, key=lambda res: int(res[:-1]))


def add_angles(ds: xr.Dataset, mtd: ConstellrSkybeeMetadataExtractor) -> xr.Dataset:
    """Add the whole-scene sun/view angles from the metadata as scalar variables.

    The bands sit on several grids (10, 20 and 60 m) and each resolution node of the datatree needs its
    own copy of the angles (``eoalign``'s ``brdf_correction`` reads them from the node holding the
    reflectance), so one copy is added per grid, named ``<angle>_<res>`` as for other multi-grid readers
    (``to_datatree`` drops the suffix). With no bands read there is no grid to follow: a single
    unsuffixed copy is placed in the 10 m node.
    """
    product_metadata = getattr(mtd, "product_metadata", {})
    observation_geometry = product_metadata.get("observation_geometry", {})
    if not observation_geometry:
        return ds

    grids = [(f"_{res}", res) for res in _grid_resolutions(ds)] or [("", "10m")]
    for key, value in observation_geometry.items():
        for suffix, res in grids:
            attrs = {"units": "degrees", **_ANGLE_ATTRS.get(key, {}), "product_metadata": {"geometry_id": res}}
            ds[f"{key}{suffix}"] = xr.DataArray(float(value), attrs=attrs)

    return ds
