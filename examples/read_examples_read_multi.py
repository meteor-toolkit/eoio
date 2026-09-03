"""read_examples_read_multi - Example usage script for eoio.read_multi"""

import os
import configparser

from eoio.interface import read_multi

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"

__all__ = []

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "eoio",
    "etc",
    "test_file_paths.config",
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# ---------------------------------------------------------------------------
# Example 1 - in-situ time series (Hypernets)
#
# Each Hypernets file already carries its own `time` dimension, so
# `concat_dim="time"` (the default) concatenates directly - no new dimension
# is created.
# ---------------------------------------------------------------------------

hypernets_l2a_file = config.get("HYPERNETS", "l2a_file", fallback=None)
hypernets_dir = os.path.dirname(hypernets_l2a_file) if hypernets_l2a_file else None

if hypernets_dir:
    hypernets_ds = read_multi(
        hypernets_dir,  # every file in the directory that a registered reader recognises
        vars_sel={"meas": "all"},
    )
    print(hypernets_ds)

# ---------------------------------------------------------------------------
# Example 2 - multi-date raster stack (Sentinel-2)
#
# Several scenes over the same region of interest are stacked into a single
# (time, band, y, x) cube. See read_examples_s2_read_multi.py for a full,
# runnable version of this example (plots, RGB composites, etc.).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pass
