"""Sentinel-3 SLSTR reader package.

This package provides the `SLSTRL1Reader` implementation moved from the
legacy module `eoio.readers.s3_slstr_l1` into a package layout mirroring
the `sentinel3_olci` reader.
"""

from .reader import SLSTRL1Reader

__all__ = ["SLSTRL1Reader"]
