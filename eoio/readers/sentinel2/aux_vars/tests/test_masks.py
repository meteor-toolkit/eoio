"""eoio.readers.sentinel2.aux_vars.tests.test_masks - tests for eoio.readers.sentinel2.aux_vars.masks"""

import unittest
from pathlib import Path
import xarray as xr
from eoio.readers.sentinel2.layout import S2Layout
from eoio.readers.base import ReaderConfig
from eoio.readers.sentinel2.aux_vars.masks import add_masks

# Path to a real Sentinel-2 SAFE product for integration testing.
# If you run the tests, ensure this path points to a valid SAFE product on your system.
# Otherwise, the integration test will be skipped.
SAFE_PATH = "/Users/seh2/Library/CloudStorage/OneDrive-NationalPhysicalLaboratory/Data/archive/sentinel2/S2A_MSIL1C_20251128T111431_N0511_R137_T30UXE_20251128T121631.SAFE"


class TestAddMasksData(unittest.TestCase):
    def test_real_safe(self):
        safe_path = Path(SAFE_PATH)
        if not safe_path.exists():
            self.skipTest(f"Test SAFE product not found at {safe_path}")

        layout = S2Layout(safe_path)
        config = ReaderConfig(vars_sel={}, subset={}, read_params={})

        ds = xr.Dataset()
        ds = add_masks(ds=ds, var_names=["test"], layout=layout, config=config)


if __name__ == "__main__":
    unittest.main()
