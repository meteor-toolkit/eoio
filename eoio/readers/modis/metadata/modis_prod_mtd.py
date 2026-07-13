"""eoio.readers.modis.metadata.modis_prod_mtd - reader for MODIS product metadata."""

from eoio.deps import lazy_rioxarray
from eoio.readers.modis.layout import MODISLayout
from eoio.readers.modis.metadata.var_names import (
    MEAS_VAR_BAND_IDS_1,
    MEAS_VAR_BAND_IDS_H,
    MEAS_VAR_BAND_IDS_Q,
    L2_MEAS_VAR_BAND_IDS_1,
    L2_MEAS_VAR_BAND_IDS_H,
    L2_MEAS_VAR_BAND_IDS_Q,
)

L1B_FILE_RES_DICT = {
    "1": MEAS_VAR_BAND_IDS_1,
    "H": MEAS_VAR_BAND_IDS_H,
    "Q": MEAS_VAR_BAND_IDS_Q,
}

L2_FILE_RES_DICT = {
    "1": L2_MEAS_VAR_BAND_IDS_1,
    "H": L2_MEAS_VAR_BAND_IDS_H,
    "Q": L2_MEAS_VAR_BAND_IDS_Q,
}


def modis_prod_mtd_reader_factory(layout: MODISLayout):
    # choose the right subreader based on the processing level
    if layout.processing_level == "L1B":
        return MODISL1BProdHDFReader(layout)
    elif layout.processing_level == "L2":
        return MODISL2ProdHDFReader(layout)


class MODISL1BProdHDFReader:
    """
    MODIS product metadata reader.
    """

    def __init__(self, layout: MODISLayout):
        rxr = lazy_rioxarray()
        if layout.file_res_key != "1":
            ds = rxr.open_rasterio(layout.path)
        else:
            ds = rxr.open_rasterio(layout.path)[1]
        self.attrs = ds.attrs
        meas_var_band_id_key = L1B_FILE_RES_DICT.get(layout.file_res_key, {})
        self.var_attrs = {
            var: {**ds[band_id].attrs, "band_id": band_id} for var, band_id in meas_var_band_id_key.items()
        }
        for var, var_attrs in self.var_attrs.items():
            # make any list attributes into single values (e.g. for scaling factors)
            idx = var_attrs["band_names"].split(",").index(var.split(" ")[-1])
            self.var_attrs[var]["band_idx"] = idx
            for attr_key in [x for x in var_attrs.keys() if ("_scales" in x) or ("_offsets" in x)]:
                attr_val = float(var_attrs[attr_key].split(",")[idx])
                self.var_attrs[var][attr_key] = attr_val


class MODISL2ProdHDFReader:
    """
    MODIS product metadata reader.
    """

    def __init__(self, layout: MODISLayout):
        rxr = lazy_rioxarray()
        ds = rxr.open_rasterio(layout.path)[list(L2_FILE_RES_DICT.keys()).index(layout.file_res_key)]
        self.attrs = ds.attrs
        meas_var_band_id_key = L2_FILE_RES_DICT.get(layout.file_res_key, {})
        self.var_attrs = {
            var: {**ds[band_id].attrs, "band_id": band_id} for var, band_id in meas_var_band_id_key.items()
        }
