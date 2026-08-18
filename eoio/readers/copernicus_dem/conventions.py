"""eoio.readers.copernicus_dem.conventions - Apply standard conventions to Copernicus DEM datasets."""


def apply_conventions(ds, layout, config):
    # Placeholder for conventions logic (attributes, CF compliance, etc.)

    roi_subset = config.subset
    if roi_subset is None:
        roi_subset_attr = None

    else:
        roi_subset_attr = roi_subset.clip_box

    eoio_attrs = {"eoio:reader": "copernicus_dem", "eoio:subset": roi_subset_attr}

    ds.attrs.update(eoio_attrs)

    return ds
