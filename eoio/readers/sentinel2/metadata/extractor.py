"""eoio.readers.sentinel2.metadata.extractor- extractor class for Sentinel-2 MSI metadata."""

from __future__ import annotations
from eoio.readers.sentinel2.metadata.var_names import (
    MEAS_VAR_BAND_IDS,
    ANGLE_VARS,
    L2A_IMG_VARS,
    L2A_QI_VARS,
    AUX_ECMWF_VARS_OLD,
    AUX_ECMWF_VARS_NEW,
    AUX_CAMS_VARS,
)
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.sentinel2.metadata.s2_prod_mtd import S2ProdXMLReader
from eoio.readers.sentinel2.metadata.s2_tl_mtd import S2TLXMLReader
from eoio.readers.sentinel2.metadata.s2_ds_mtd import S2DSXMLReader
from eoio.readers.footprint_utils import normalize_footprint

# Nominal resolutions for L2A non-band layers (used for geometry naming)
_L2A_VAR_RES_M = {
    "AOT": 10,
    "WVP": 10,
    "SCL": 20,
    "TCI": 10,
    # QI probability layers
    "CLDPRB": 20,
    "SNWPRB": 20,
}


class S2MSIMetadataExtractor(BaseMetadataExtractor):
    """
    High-level metadata extractor for Sentinel-2 MSI products.

    This class orchestrates metadata extraction from both Sentinel-2 product-
    level metadata (``MTD_MSIL1C.xml`` / ``MTD_MSIL2A.xml``) and tile-level
    metadata (``MTD_TL.xml``), and exposes the results as structured
    Python dictionaries suitable for downstream use (e.g. MetEOR tools).

    It delegates low-level XML parsing to :class:`S2ProdXMLReader` and
    :class:`S2TLXMLReader`.
    """

    def __init__(self, reader):
        """
        Initialise the Sentinel-2 MSI metadata extractor.

        :param reader:
            The EOIO Sentinel-2 reader instance providing layout and configuration.
        """
        super().__init__(reader)

        # define attributes
        self.layout = reader.layout
        self.reader = reader

        # initialise xml file readers - one for (a) MTD_MSILXX.xml and (b) MTD_TL.xml
        self.prod_xml_reader = S2ProdXMLReader(self.layout.product_metadata_xml())
        self.tl_xml_reader = S2TLXMLReader(self.layout.tl_metadata_xml())
        self.ds_xml_reader = S2DSXMLReader(self.layout.ds_metadata_xml())

        # cache per variable results to prevent repeated querying
        self._solar_irradiance = self.prod_xml_reader.find_solar_irradiance()
        self._physical_gains = self.prod_xml_reader.find_physical_gains()
        self._absolute_calibration_accuracy = self.ds_xml_reader.find_absolute_calibration_accuracy()
        self._cross_band_calibration_accuracy = self.ds_xml_reader.find_cross_band_calibration_accuracy()
        self._multi_temporal_calibration_accuracy = self.ds_xml_reader.find_multi_temporal_calibration_accuracy()
        self._noise_model_alpha = self.ds_xml_reader.find_noise_model_alpha()
        self._noise_model_beta = self.ds_xml_reader.find_noise_model_beta()

        # handling for radiometric offests - missing for PSD < v15
        self._radiometric_offsets = self.prod_xml_reader.find_reflectance_offsets()

        # set offset as zero if missing
        if self._radiometric_offsets is None:
            self._radiometric_offsets = {}
            for key in self._solar_irradiance.keys():
                self._radiometric_offsets[key] = 0

    def get_basic_metadata(self) -> dict:
        """
        Return high-level, product-wide metadata.

        This metadata is intended for discovery, cataloguing, and general
        product description, and is shared across all variables in the dataset.

        :returns:
            Dictionary containing basic Sentinel-2 product metadata.
        """

        res = [v for k, v in self.prod_xml_reader.find_all_band_resolutions().items()]
        geoms = [str(r) + "m" for r in res]

        basic_md = {
            "title": self.prod_xml_reader.find_product_type(),
            "summary": "TBC",
            "institution": "TBC",
            "source": "TBC",
            "references": "TBC",
            "date_created": "TBC",
            "standard_name_vocabulary": "TBC",
            "license": "TBC",
            "collection_name": self.prod_xml_reader.find_product_type(),
            "product_name": self.prod_xml_reader.find_product_uri(),
            "platform": self.prod_xml_reader.find_spacecraft_name(),
            "platform_type": "satellite",
            "platform_vocabulary": "TBC",
            "instrument": "MSI",
            "processing_level": self.prod_xml_reader.find_processing_level(),
            "processing_version": self.prod_xml_reader.find_processing_baseline(),
            "spatial_resolution": res,
            "geometry_ids": geoms,
            "geospatial_bounds": self.prod_xml_reader.find_bounds().wkt,
            "product_date": self.prod_xml_reader.find_product_start_date(),
            "product_datetime": self.prod_xml_reader.find_product_start_datetime(),
            "description": "TBD",
            "footprint": normalize_footprint(
                geometry_input=self.prod_xml_reader.find_bounds(),
                crs_input=self.tl_xml_reader.find_horizontal_cs_code(),
            ),
        }

        return basic_md

    def get_product_metadata(self) -> dict:
        """
        Return detailed product- and tile-level metadata.

        This metadata captures acquisition context, quality indicators,
        and tile identifiers that are not variable-specific.

        :returns:
            Dictionary containing Sentinel-2 product metadata.
        """

        prod_md = {
            "orbit_number": self.prod_xml_reader.find_orbit_number(),
            "orbit_direction": self.prod_xml_reader.find_orbit_direction(),
            "quantification_level": self.prod_xml_reader.find_quantification_values(),
            "horizontal_cs_code": self.tl_xml_reader.find_horizontal_cs_code(),
            "tile_id": self.tl_xml_reader.find_tile_id(),
            "datastrip_id": self.tl_xml_reader.find_datastrip_id(),
            "downlink_priority": self.tl_xml_reader.find_downlink_priority(),
            "sensing_time": self.tl_xml_reader.find_sensing_datetime(),
            "cloudy_pixel_percentage": self.tl_xml_reader.find_cloudy_pixel_percentage(),
            "snow_pixel_percentage": self.tl_xml_reader.find_snow_pixel_percentage(),
            "degraded_msi_data_percentage": self.tl_xml_reader.find_degraded_msi_data_percentage(),
            "reflectance_conversion_u": self.prod_xml_reader.find_reflectance_conversion_u(),
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

        basic_metadata: dict = {
            "units": "",
            "long_name": "",
            "standard_name": "",
        }

        if var in MEAS_VAR_BAND_IDS:
            if self.prod_xml_reader.find_product_type() == "S2MSI1C":
                basic_metadata["standard_name"] = "toa_reflectance"
                basic_metadata["long_name"] = "TOA HCRF"
                basic_metadata["measurand"] = "reflectance"
                basic_metadata["units"] = "1"

            elif self.prod_xml_reader.find_product_type() == "S2MSI2A":
                basic_metadata["standard_name"] = "boa_reflectance"
                basic_metadata["long_name"] = "BOA HCRF"
                basic_metadata["measurand"] = "reflectance"
                basic_metadata["units"] = "1"

            anc_vars = []
            for anc_var in self.reader.aux_def.get("all", []):
                if ("solar" in anc_var) or (var in anc_var):
                    anc_vars.append(anc_var)

            if anc_vars:
                basic_metadata["ancillary_variables"] = anc_vars

        elif var in ANGLE_VARS:
            basic_metadata["standard_name"] = var
            basic_metadata["long_name"] = var.replace("_", " ")
            basic_metadata["units"] = "degrees"
            if var.startswith("viewing_zenith"):
                basic_metadata["measurand"] = "viewing_zenith_angle"
            elif var.startswith("viewing_azimuth"):
                basic_metadata["measurand"] = "viewing_azimuth_angle"
            else:
                basic_metadata["measurand"] = "angle"

        elif var in set(AUX_ECMWF_VARS_OLD + AUX_ECMWF_VARS_NEW + AUX_CAMS_VARS):
            # Preserve CF-compliant attrs read from AUX GRIB files (e.g. tcwv)
            # by not overwriting them with empty placeholders.
            return {}

        return basic_metadata

    def get_variable_product_metadata(self, var: str) -> dict:
        """
        Return detailed metadata for a specific Sentinel-2 band variable.

        This includes spatial characteristics, radiometric calibration
        parameters, and tile geometry information relevant to the band.

        :param var:
            Variable name corresponding to a Sentinel-2 band (e.g. ``"B02"``).
        :returns:
            Dictionary containing variable-specific metadata.
        :raises KeyError:
            If ``var`` is not a recognised Sentinel-2 band.
        """

        variable_mtd = {}

        if var in MEAS_VAR_BAND_IDS:
            try:
                band_id = MEAS_VAR_BAND_IDS[var]
            except KeyError as e:
                raise KeyError(f"Unknown Sentinel-2 band variable: {var}") from e

            res = self.prod_xml_reader.find_band_resolution(band_id)
            geom = str(res) + "m"

            variable_mtd.update(
                {
                    "spatial_resolution": res,
                    "spatial_resolution_units": "m",
                    "geometry_id": geom,
                    "band_central_wavelength": self.prod_xml_reader.find_band_central_wavelength(band_id),
                    "band_central_wavelength_units": "nm",
                    "solar_irradiance": self._solar_irradiance[band_id],
                    "solar_irradiance_units": self.prod_xml_reader.find_solar_irradiance_unit(),
                    "radiometric_offset": self._radiometric_offsets[var],
                    "physical_gains": self._physical_gains[band_id],
                    "band_id": band_id,
                    "tile_shape": self.tl_xml_reader.find_tile_shape(res),
                    "geoposition": self.tl_xml_reader.find_geoposition(res),
                    "absolute_calibration_accuracy": self._absolute_calibration_accuracy[band_id],
                    "cross_band_calibration_accuracy": self._cross_band_calibration_accuracy[band_id],
                    "multi_temporal_calibration_accuracy": self._multi_temporal_calibration_accuracy[band_id],
                    "noise_model_alpha": self._noise_model_alpha[band_id],
                    "noise_model_beta": self._noise_model_beta[band_id],
                }
            )

        elif var in L2A_IMG_VARS or var in L2A_QI_VARS:
            res = _L2A_VAR_RES_M.get(var, 20)
            geom = str(res) + "m"

            variable_mtd.update(
                {
                    "spatial_resolution": res,
                    "spatial_resolution_units": "m",
                    "geometry_id": geom,
                    "geoposition": self.tl_xml_reader.find_geoposition(res),
                }
            )

        elif var in ANGLE_VARS:
            res = self.tl_xml_reader.find_sun_angle_steps()[0]
            geom = str(res) + "m"

            variable_mtd.update(
                {
                    "spatial_resolution": res,
                    "spatial_resolution_units": "m",
                    "geometry_id": geom,
                    "geoposition": self.tl_xml_reader.find_geoposition(10),
                }
            )

        elif var in AUX_ECMWF_VARS_OLD + AUX_ECMWF_VARS_NEW + AUX_CAMS_VARS:
            # Sentinel-2 auxiliary ECMWF and CAMS data is provided at ~5000m resolution
            variable_mtd.update(
                {
                    "spatial_resolution": 5000,
                    "spatial_resolution_units": "m",
                    "geometry_id": "5000m",
                }
            )

        return variable_mtd


if __name__ == "__main__":
    pass
