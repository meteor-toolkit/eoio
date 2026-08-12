"""Conventions helpers for sentinel3_slstr.

For now provide a noop `apply_conventions` to keep the adapter flow consistent
with the sentinel3_olci pattern.
"""

from __future__ import annotations


def apply_conventions(ds, layout=None, roi_subset=None, config=None):
    # Placeholder for conventions logic (attributes, CF compliance, etc.)

    roi_subset = config.subset
    if roi_subset is None:
        # "" (not None) -- an attr value of None can't be written to netCDF.
        roi_subset_attr = ""

    else:
        roi_subset_attr = roi_subset.clip_box

    eoio_attrs = {"eoio:reader": "sentinel3_slstr", "eoio:subset": roi_subset_attr}

    ds.attrs.update(eoio_attrs)

    return ds
