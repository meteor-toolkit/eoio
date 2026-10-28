"""Round-trip tests for the netCDF writer against real EO products.

Reads an actual product with :func:`eoio.read`, writes it with the netCDF writer, reads
it back and checks both data and attributes survive. Needs product files from the shared
test dataset area and skips when it is not mounted, per the project testing convention
(see ``eoio/etc/test_file_paths.config``).

The raster cases are covered by Landsat 8 and two Sentinel-2 MSI L1C products (a 2022 and
a 2025 scene, the latter with a newer processing baseline). To add another product, add an
entry to ``test_file_paths.config`` and extend ``PRODUCTS`` below.
"""

import configparser
import os

import numpy as np
import pytest
import xarray as xr

from eoio import read
from eoio.serialisation import decode_attrs
from eoio.writers.netcdf.writer import NetCDFWriter

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
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


@pytest.mark.parametrize("test_id, section, key", PRODUCTS, ids=[p[0] for p in PRODUCTS])
def test_real_product_round_trips_through_the_writer(tmp_path, test_id, section, key):
    path = _product_path(section, key)
    ds = read(path)

    written = NetCDFWriter().write(ds, tmp_path / f"{test_id}.nc")

    with xr.open_dataset(written) as opened:
        back = decode_attrs(opened)

        for name, expected in ds.data_vars.items():
            assert name in back, f"{test_id}: variable {name!r} missing after round trip"
            if np.issubdtype(expected.dtype, np.number):
                np.testing.assert_allclose(
                    back[name].values,
                    expected.values,
                    equal_nan=True,
                    err_msg=f"{test_id}: variable {name!r} changed",
                )

        missing = set(ds.attrs) - set(back.attrs)
        assert not missing, f"{test_id}: attributes lost: {sorted(missing)}"

        assert "written to" in back.attrs["history"]


@pytest.mark.parametrize("test_id, section, key", PRODUCTS, ids=[p[0] for p in PRODUCTS])
def test_real_product_is_compressed(tmp_path, test_id, section, key):
    path = _product_path(section, key)
    ds = read(path)

    compressed = NetCDFWriter().write(ds, tmp_path / f"{test_id}_c.nc")
    plain = NetCDFWriter(write_params={"compress": False}).write(ds, tmp_path / f"{test_id}_p.nc")

    assert os.path.getsize(compressed) <= os.path.getsize(plain)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
