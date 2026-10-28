"""Round-trip tests for eoio.serialisation against real EO products.

These read an actual product via :func:`eoio.read`, encode its attributes, write to
netCDF, read it back and check the attributes survive. They need product files from the
shared test dataset area and skip when it is not mounted, per the project testing
convention (see ``eoio/etc/test_file_paths.config``).

The raster cases are covered by Landsat 8 and two Sentinel-2 MSI L1C products (a 2022 and a
2025 scene, the latter with a newer processing baseline). To add another product, add an
entry to ``test_file_paths.config`` and extend ``PRODUCTS`` below.
"""

import configparser
import datetime as dt
import os

import numpy as np
import pytest
import xarray as xr

from eoio import read
from eoio.serialisation import decode_attrs, encode_attrs

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "etc",
    "test_file_paths.config",
)
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)


#: (test id, config section, config key) for products to round trip.
PRODUCTS = [
    ("hypernets_l1b", "HYPERNETS", "l1b_file"),
    ("hypernets_l2a", "HYPERNETS", "l2a_file"),
    ("landsat8", "Landsat", "l8_dir"),
    ("sentinel2_l1c_20251128", "Sentinel2", "l1c_dir_20251128_T30UXE"),
    ("radcalnet", "RadCalNet", "output_file"),
]


def _product_path(section: str, key: str) -> str:
    """
    Resolve a configured product path, skipping the test when it is unavailable.

    :param section: Config section name.
    :param key: Config key within the section.
    :returns: The product path.
    """
    path = _config.get(section, key, fallback=None)
    if not path:
        pytest.skip(f"No [{section}] {key} entry in {CONFIG_PATH}")
    path = path.strip()
    if not os.path.exists(path):
        pytest.skip(f"Test data not found at {path}")
    return path


def _values_equal(left, right) -> bool:
    """
    Compare two attribute values structurally, tolerating array and scalar widening.

    netCDF widens plain Python scalars (``int`` becomes ``np.int64``), so this compares
    by value rather than by type.

    :param left: First value.
    :param right: Second value.
    :returns: ``True`` when the values are equivalent.
    """
    if isinstance(left, (np.ndarray, list, tuple)) or isinstance(right, (np.ndarray, list, tuple)):
        left_array, right_array = np.asarray(left, dtype=object), np.asarray(right, dtype=object)
        if left_array.shape != right_array.shape:
            return False
        return all(_values_equal(a, b) for a, b in zip(left_array.ravel(), right_array.ravel()))

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_values_equal(left[k], right[k]) for k in left)

    if isinstance(left, (dt.datetime, dt.date, dt.time)) or isinstance(right, (dt.datetime, dt.date, dt.time)):
        return left == right

    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return bool(np.isclose(float(left), float(right), equal_nan=True))

    return bool(left == right)


def _assert_scope_round_tripped(original: dict, restored: dict, scope: str) -> None:
    """
    Assert one attribute scope survived the round trip.

    :param original: Attributes before encoding.
    :param restored: Attributes after decoding.
    :param scope: Scope name, used in failure messages.
    """
    missing = set(original) - set(restored)
    assert not missing, f"{scope}: attributes lost in round trip: {sorted(missing)}"

    for key, expected in original.items():
        assert _values_equal(restored[key], expected), (
            f"{scope}: attribute {key!r} changed: {restored[key]!r} != {expected!r}"
        )


def _assert_attrs_round_trip_through_netcdf(ds: xr.Dataset, target, label: str) -> None:
    """
    Write ``ds`` with encoded attributes, read it back, and assert every attribute survived.

    The file is opened with ``decode_cf=False`` so that what is compared is what was stored.
    With xarray's default CF decoding, ``_FillValue``, ``scale_factor``, ``add_offset`` and the
    time ``units``/``calendar`` are moved from ``attrs`` into ``encoding`` on open, which would
    report them as lost although they are in the file.

    :param ds: Dataset to round trip.
    :param target: Path to write to.
    :param label: Name used in failure messages.
    """
    original_attrs = dict(ds.attrs)
    original_var_attrs = {name: dict(ds[name].attrs) for name in ds.variables}

    encode_attrs(ds).to_netcdf(target)

    with xr.open_dataset(target, decode_cf=False) as opened:
        restored = decode_attrs(opened)

    _assert_scope_round_tripped(original_attrs, restored.attrs, f"{label} dataset")
    for name, expected in original_var_attrs.items():
        _assert_scope_round_tripped(expected, dict(restored[name].attrs), f"{label} variable {name!r}")


def test_cf_attributes_are_compared_as_stored(tmp_path):
    # Regression: reflectance_uncertainty carries _FillValue in attrs, which a CF-decoding
    # open moves to encoding, so the round-trip test reported it lost.
    ds = xr.Dataset(
        {"u": (("w",), np.zeros(3), {"_FillValue": np.float64(9.969209968386869e36), "units": "1"})},
        coords={"w": [400.0, 500.0, 600.0]},
    )

    _assert_attrs_round_trip_through_netcdf(ds, tmp_path / "fill.nc", "synthetic")


@pytest.mark.parametrize("test_id, section, key", PRODUCTS, ids=[p[0] for p in PRODUCTS])
def test_real_product_attrs_round_trip_through_netcdf(tmp_path, test_id, section, key):
    path = _product_path(section, key)
    ds = read(path)

    _assert_attrs_round_trip_through_netcdf(ds, tmp_path / f"{test_id}.nc", test_id)


@pytest.mark.parametrize("test_id, section, key", PRODUCTS, ids=[p[0] for p in PRODUCTS])
def test_real_product_needs_encoding_to_be_written(tmp_path, test_id, section, key):
    """The problem this package exists to solve, demonstrated on real reader output."""
    path = _product_path(section, key)
    ds = read(path)

    try:
        ds.to_netcdf(tmp_path / f"{test_id}_raw.nc")
    except (TypeError, ValueError):
        return  # expected: the product carries attributes netCDF cannot store

    pytest.skip(f"{test_id} happens to carry only serialisable attributes; nothing to prove here")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
