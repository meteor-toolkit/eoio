"""eoio.readers.gistda_theos.aux_data - auxiliary data reading functions for THEOS reader."""

from __future__ import annotations

import warnings
from typing import Any

import xarray as xr
from processor_tools.utils.dict_tools import get_value

from eoio.readers.gistda_theos.metadata import GISTDATHEOSMetadataExtractor


#: Per-variable attributes of the scene-level angle variables ``add_angles`` creates. Variable
#: names follow Landsat's convention (``viewing_*``, with the CF ``sensor_*`` as standard_name),
#: which is also what eoalign's ``brdf_correction`` looks for. The nested
#: ``product_metadata.geometry_id`` is what eoio's ``to_datatree`` (``grid_attr``) uses to place
#: a variable in a grid node: these scalars have no ``x_15m``/``y_15m`` dims to infer it from, so
#: without it they would land in the tree's root node instead of next to the reflectance. No
#: ``measurand`` on purpose: the ``stack`` processor would merge all six into one ``angle`` variable.
_ANGLE_ATTRS: dict[str, dict[str, Any]] = {
    "viewing_angle_along_track": {"long_name": "Viewing angle along track", "standard_name": None},
    "viewing_angle_across_track": {"long_name": "Viewing angle across track", "standard_name": None},
    "viewing_zenith_angle": {
        "long_name": "Viewing zenith angle",
        "standard_name": "sensor_zenith_angle",
        "description": "Angle between the line of sight to the sensor and the local zenith at the observation target.",
    },
    "viewing_azimuth_angle": {"long_name": "Viewing azimuth angle", "standard_name": "sensor_azimuth_angle"},
    "solar_zenith_angle": {"long_name": "Solar zenith angle", "standard_name": "solar_zenith_angle"},
    "solar_azimuth_angle": {"long_name": "Solar azimuth angle", "standard_name": "solar_azimuth_angle"},
}


def _angle_attrs(name: str) -> dict:
    attrs: dict[str, Any] = {"units": "degrees"}
    attrs.update({k: v for k, v in _ANGLE_ATTRS.get(name, {}).items() if v is not None})
    attrs["product_metadata"] = {"geometry_id": "15m"}
    return attrs


def read_aux(ds, aux, mtd: GISTDATHEOSMetadataExtractor):
    if aux is None:
        raise ValueError(
            "No auxiliary data files have been selected. Please set aux before reading using `.aux = [...]`."
        )
    handled = set()
    if "observation_geometry" in aux:
        ds = add_angles(ds, mtd)
        handled.add("observation_geometry")
    if "mask" in aux:  # TODO: update naming when masks implemented
        warnings.warn(
            "Mask data is not yet implemented for GISTDA THEOS data. Please check back later for updates."
        )  # TODO add mask here when implemented
        handled.add("mask")
    # Only warn for names this function doesn't already handle above -- observation_geometry
    # is fully implemented (add_angles), so warning about it unconditionally (as this used to)
    # falsely told a caller who successfully got angle data back that aux reading was a no-op.
    unimplemented = sorted(set(aux) - handled)
    if unimplemented:
        warnings.warn(
            f"Auxiliary data {unimplemented} is not yet implemented for GISTDA THEOS data. "
            "Please check back later for updates."
        )
    return ds


def add_aux(
    *,
    ds: xr.Dataset,
    layout: Any,
    aux_names: Any,
    # subset: Any,
) -> xr.Dataset:
    """
    Currently no mask implemented for THEOS data; placeholder function.

    Will read mask data from GISTDA THEOS metadata and add them to ds.

    Will return original ds unchanged on unexpected formats.
    """
    return ds


def add_angles(
    ds: xr.Dataset,
    mtd: GISTDATHEOSMetadataExtractor,
    # layout: Any,
    # subset: Any,
) -> xr.Dataset:
    """
    Read viewing / solar angles from GISTDA THEOS metadata and add them to ds.

    Returns original ds unchanged if angles in unexpected format.
    """

    # locate geometric metadata list (prefer ds.attrs, else parse XML)
    def _locate_angles_list() -> dict[str, Any] | None:
        if "product_metadata" in ds.attrs and "observation_geometry" in ds.attrs["product_metadata"]:
            return ds.attrs["product_metadata"]["observation_geometry"]
        try:  # when extracting basic metadata
            prod_mtd = mtd.product_metadata
            if get_value(prod_mtd, "Scene_Source"):
                md = get_value(prod_mtd, "Scene_Source")

                # deep search for observation_geometry
                def _deep_search(d):
                    if isinstance(d, dict):
                        if get_value(d, "SUN_ELEVATION") is not None:
                            return {
                                "solar_azimuth_angle": get_value(d, "SUN_AZIMUTH"),
                                "solar_zenith_angle": str(90 - float(get_value(d, "SUN_ELEVATION"))),
                            }
                        else:
                            return None
                    if isinstance(d, list):
                        for item in d:
                            val_res = _deep_search(item)
                            if val_res is not None:
                                return val_res
                    return None

                return _deep_search(md)
        except Exception:
            return None
        return None

    angles_list = _locate_angles_list()
    if not angles_list:
        return ds

    for name, value in angles_list.items():
        # A single missing angle (e.g. SUN_ELEVATION absent upstream -- see
        # metadata._solar_zenith_from_elevation) shouldn't crash the whole read when
        # this function's own docstring already promises graceful degradation; skip
        # just that one value instead.
        if value is None:
            continue
        ds = ds.assign({name: xr.DataArray(float(value), attrs=_angle_attrs(name))})

    return ds
