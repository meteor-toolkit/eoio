"""eoio.readers.modis.metadata.extractor- extractor class for MODIS metadata."""

from __future__ import annotations
from eoio.readers.modis.metadata.var_names import (
    ANGLE_VARS,
    MEAS_VAR_BAND_IDS_Q,
    MEAS_VAR_BAND_IDS_H,
    MEAS_VAR_BAND_IDS_1,
    L2_MEAS_VAR_BAND_IDS_1,
    L2_MEAS_VAR_BAND_IDS_H,
    L2_MEAS_VAR_BAND_IDS_Q,
)

from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.modis.metadata.modis_prod_mtd import modis_prod_mtd_reader_factory

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

FILE_RES_DICT = {
    "L1B": L1B_FILE_RES_DICT,
    "L2": L2_FILE_RES_DICT,
}

FILE_RES_KEY_DICT = {"1": 1000, "H": 500, "Q": 250}


class MODISMetadataExtractor(BaseMetadataExtractor):
    """
    High-level metadata extractor for MODIS products.
    """

    def __init__(self, reader):
        """
        Initialise the MODIS metadata extractor.

        :param reader:
            The EOIO MODIS reader instance providing layout and configuration.
        """
        super().__init__(reader)

        # define attributes
        self.layout = reader.layout
        self.reader = reader
        self.meas_var_band_ids = FILE_RES_DICT.get(self.layout.processing_level, {}).get(self.layout.file_res_key, {})

        # initialise metadata readers
        self.prod_hdf_reader = modis_prod_mtd_reader_factory(self.layout)

    def get_basic_metadata(self) -> dict:
        """
        Return high-level, product-wide metadata.

        This metadata is intended for discovery, cataloguing, and general
        product description, and is shared across all variables in the dataset.

        :returns:
            Dictionary containing basic MODIS product metadata.
        """

        res = [FILE_RES_KEY_DICT.get(self.layout.file_res_key)] * len(
            FILE_RES_DICT.get(self.layout.processing_level, {}).get(self.layout.file_res_key, [])
        )
        geoms = [str(r) + "m" for r in res]

        basic_md = {
            "title": self.prod_hdf_reader.attrs.get("LONGNAME", "TBC"),
            "summary": "TBC",
            "institution": "TBC",
            "source": "TBC",
            "references": "TBC",
            "date_created": "TBC",
            "standard_name_vocabulary": "TBC",
            "license": "TBC",
            "collection_name": self.prod_hdf_reader.attrs.get("SHORTNAME", "TBC"),
            "product_name": self.prod_hdf_reader.attrs.get("PRODUCT_NAME", "TBC"),
            "platform": self.prod_hdf_reader.attrs.get("ASSOCIATEDPLATFORMSHORTNAME.1", "TBC"),
            "platform_type": "satellite",
            "platform_vocabulary": "TBC",
            "instrument": self.prod_hdf_reader.attrs.get("ASSOCIATEDINSTRUMENTSHORTNAME.1", "TBC"),
            "processing_level": self.layout.processing_level,
            "processing_version": self.layout.proc_version,
            "spatial_resolution": res,
            "geometry_ids": geoms,
            "geospatial_bounds": "TBC",
            "product_date": self.prod_hdf_reader.attrs.get("RANGEBEGINNINGDATE", "TBC"),
            "product_datetime": self.prod_hdf_reader.attrs.get("RANGEBEGINNINGDATE", "TBC")
            + "T"
            + self.prod_hdf_reader.attrs.get("RANGEBEGINNINGTIME", "TBC"),
            "description": "TBD",
            "eoio:reader": "modis",
        }

        return basic_md

    def get_product_metadata(self) -> dict:
        """
        Return detailed product- and tile-level metadata.

        This metadata captures acquisition context, quality indicators,
        and tile identifiers that are not variable-specific.

        :returns:
            Dictionary containing MODIS product metadata.
        """

        prod_md = {
            "orbit_number": self.prod_hdf_reader.attrs.get("ORBITNUMBER.1", "TBC"),
            "day_night_flag": self.prod_hdf_reader.attrs.get("DAYNIGHTFLAG", "TBC"),
            "instrument_scan_number": self.prod_hdf_reader.attrs.get("Number of instrument scans", "TBC"),
        }

        return prod_md

    def get_variable_basic_metadata(self, var: str) -> dict:
        """
        Return basic descriptive metadata for a dataset variable.

        This method is intended for lightweight metadata such as units,
        names, or CF-style descriptors. It is currently a placeholder.

        :param var:
            Variable name for which metadata should be extracted.
        :returns:
            Dictionary containing basic variable metadata.
        """

        basic_metadata: dict[str, object] = {
            "units": "",
            "long_name": "",
            "standard_name": "",
        }

        if var in self.meas_var_band_ids:
            if "MOD02" in self.prod_hdf_reader.attrs.get("SHORTNAME", ""):
                basic_metadata["standard_name"] = "toa_reflectance"
                basic_metadata["long_name"] = "TOA HCRF"
                basic_metadata["units"] = "1"
            elif "MOD09" in self.prod_hdf_reader.attrs.get("SHORTNAME", ""):
                basic_metadata["standard_name"] = "boa_reflectance"
                basic_metadata["long_name"] = "BOA HCRF"
                basic_metadata["units"] = "1"

            anc_vars: list[str] = []
            for anc_var in self.reader.aux_def:
                if ("solar" in anc_var) or (var in anc_var):
                    anc_vars.append(anc_var)

            if anc_vars:
                basic_metadata["ancillary_variables"] = anc_vars

        return basic_metadata

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Return detailed metadata for a specific MODIS band variable.

        This includes spatial characteristics, radiometric calibration
        parameters, and tile geometry information relevant to the band.

        :param var:
            Variable name corresponding to a MODIS band (e.g. ``"Band 2"``).
        :returns:
            Dictionary containing variable-specific metadata.
        :raises KeyError:
            If ``var`` is not a recognised MODIS band.
        """

        variable_mtd = {}

        if var in self.meas_var_band_ids:
            var_attr_dict = self.prod_hdf_reader.var_attrs.get(var, {})

            try:
                band_id = self.meas_var_band_ids[var]
            except KeyError as e:
                raise KeyError(f"Unknown MODIS band variable: {var}") from e

            res = FILE_RES_KEY_DICT.get(self.layout.file_res_key)
            geom = str(res) + "m"

            variable_mtd.update(
                {
                    "spatial_resolution": res,
                    "spatial_resolution_units": "m",
                    "geometry_id": geom,
                    "band_central_wavelength": "TBC",
                    "band_central_wavelength_units": "nm",
                    "reflectance_offsets": var_attr_dict.get("reflectance_offsets", "N/A")
                    if self.layout.processing_level == "L1B"
                    else var_attr_dict.get("add_offset", "N/A"),
                    "reflectance_scales": var_attr_dict.get("reflectance_scales", "N/A")
                    if self.layout.processing_level == "L1B"
                    else var_attr_dict.get("scale_factor", "N/A"),
                    "radiance_offsets": var_attr_dict.get("radiance_offsets", "N/A"),
                    "radiance_scales": var_attr_dict.get("radiance_scales", "N/A"),
                    "band_id": band_id,
                    "band_idx": var_attr_dict.get("band_idx", 0),
                    "tile_shape": "TBC",
                    "geoposition": "TBC",
                    "absolute_calibration_accuracy": "TBC",
                    "cross_band_calibration_accuracy": "TBC",
                    "multi_temporal_calibration_accuracy": "TBC",
                    "noise_model_alpha": "TBC",
                    "noise_model_beta": "TBC",
                }
            )

        elif var in ANGLE_VARS:
            res = 1000
            geom = str(res) + "m"

            variable_mtd.update(
                {
                    "spatial_resolution": res,
                    "spatial_resolution_units": "m",
                    "geometry_id": geom,
                    "geoposition": "TBC",
                }
            )

        return variable_mtd
