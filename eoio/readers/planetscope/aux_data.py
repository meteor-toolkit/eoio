"""eoio.readers.planetscope.aux_data - PlanetScope auxiliary data utilities."""

from typing import Any
import warnings
import xarray as xr
from eoio.readers.planetscope.metadata import PlanetScopeMetadataExtractor


def read_aux(ds, aux, mtd: PlanetScopeMetadataExtractor):
    if not aux:
        raise ValueError(
            "No auxiliary data files have been selected. Please set aux before reading using `.aux = [...]`."
        )
    if "observation_geometry" in aux:
        ds = add_angles(ds, mtd)
    if "mask" in aux:  # TODO: update naming when masks implemented
        warnings.warn(
            "Mask data is not yet implemented for SuperDove data. Please check back later for updates."
        )  # TODO add masks here when implemented
    warnings.warn("Auxiliary data is not yet implemented for SuperDove data. Please check back later for updates.")
    return ds


def add_masks(
    *,
    ds: xr.Dataset,
    layout: Any,
    aux_names: Any,
    # subset: Any,
) -> xr.Dataset:
    """
    Currently no masks implemented for PlanetScope data; placeholder function.

    Will read mask data from PlanetScope metadata and add them to ds.

    Will return original ds unchanged on unexpected formats.
    """
    return ds


def add_angles(
    ds: xr.Dataset,
    mtd: PlanetScopeMetadataExtractor,
    # layout: Any,
    # subset: Any,
) -> xr.Dataset:
    """
    Add sun/view angle grids from PlanetScope metadata.
    """
    metadata = mtd.product_metadata
    angle_attrs = mtd.get_angle_metadata()
    for src, dst in zip(
        ["sun_azimuth", "sun_elevation", "satellite_azimuth", "view_angle"],
        [
            "solar_azimuth_angle",
            "solar_zenith_angle",
            "viewing_azimuth_angle",
            "viewing_zenith_angle",
        ],
    ):
        angle = metadata["product_properties"][src]
        if dst == "solar_zenith_angle":
            angle = 90 - angle

        # One whole-scene value per angle, kept as a scalar (not a 1x1 grid of its own) so
        # to_datatree puts it in the same node as the reflectance, where eoalign's
        # brdf_correction looks for it. The nested product_metadata.geometry_id is what to_datatree
        # (grid_attr) reads to do that placement, as a scalar has no x_3m/y_3m dims to infer it
        # from. No "measurand" on purpose: the stack processor would merge all four into one
        # ``angle`` variable.
        attrs = {k: v for k, v in angle_attrs[dst].items() if k not in ("measurand", "geometry_id")}
        attrs["product_metadata"] = {"geometry_id": "3m"}
        ds[dst] = xr.DataArray(float(angle), attrs=attrs)

    return ds
