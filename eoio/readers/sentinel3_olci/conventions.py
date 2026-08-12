"""eoio.readers.sentinel3_olci.conventions - Apply standard conventions to Sentinel-3 OLCI datasets."""


def apply_conventions(ds, layout, config):
    # Placeholder for conventions logic (attributes, CF compliance, etc.)

    roi_subset = config.subset
    if roi_subset is None:
        # "" (not None) -- an attr value of None can't be written to netCDF.
        roi_subset_attr = ""

    else:
        roi_subset_attr = roi_subset.clip_box

    eoio_attrs = {"eoio:reader": "sentinel3_olci", "eoio:subset": roi_subset_attr}

    ds.attrs.update(eoio_attrs)

    return ds
