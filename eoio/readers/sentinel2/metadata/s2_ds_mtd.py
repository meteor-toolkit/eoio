"""eoio.readers.sentinel2.metadata.s2_ds_mtd - reader for Sentinel-2 datastrip metadata."""

from typing import Optional
from eoio.readers.xml import XMLReader

# Location of fields of interest within the datastrip metadata file
S2_DS_MTD_METADATA_PATHS: dict[str, str] = {
    # Select parent elements that carry bandId
    "radiometric_quality": ("./n1:Quality_Indicators_Info/Radiometric_Info/Radiometric_Quality_List"),
}


class S2DSXMLReader(XMLReader):
    """
    Sentinel-2 datastrip metadata reader for ``MTD_DS.xml``.

    Currently focused on radiometric quality indicators under
    ``Quality_Indicators_Info/Radiometric_Info``.
    """

    metadata_paths = S2_DS_MTD_METADATA_PATHS

    # ------------------------------------------------------------------
    # Radiometric Quality Parameters
    # ------------------------------------------------------------------

    def find_absolute_calibration_accuracy(self) -> Optional[dict[int, float]]:
        """Return ABSOLUTE_CALIBRATION_ACCURACY per bandId."""
        return self.find_mapping(
            "radiometric_quality",
            key="Radiometric_Quality",
            key_attr="bandId",
            key_cast=int,
            value_xpath="ABSOLUTE_CALIBRATION_ACCURACY",
            value_cast=float,
        )

    def find_cross_band_calibration_accuracy(self) -> Optional[dict[int, float]]:
        """Return CROSS_BAND_CALIBRATION_ACCURACY per bandId."""
        return self.find_mapping(
            "radiometric_quality",
            key="Radiometric_Quality",
            key_attr="bandId",
            key_cast=int,
            value_xpath="CROSS_BAND_CALIBRATION_ACCURACY",
            value_cast=float,
        )

    def find_multi_temporal_calibration_accuracy(self) -> Optional[dict[int, float]]:
        """Return MULTI_TEMPORAL_CALIBRATION_ACCURACY per bandId."""
        return self.find_mapping(
            "radiometric_quality",
            key="Radiometric_Quality",
            key_attr="bandId",
            key_cast=int,
            value_xpath="MULTI_TEMPORAL_CALIBRATION_ACCURACY",
            value_cast=float,
        )

    def find_noise_model_alpha(self) -> Optional[dict[int, float]]:
        """Return Noise_Model/ALPHA per bandId."""
        return self.find_mapping(
            "radiometric_quality",
            key="Radiometric_Quality",
            key_attr="bandId",
            key_cast=int,
            value_xpath="Noise_Model/ALPHA",
            value_cast=float,
        )

    def find_noise_model_beta(self) -> Optional[dict[int, float]]:
        """Return Noise_Model/BETA per bandId."""
        return self.find_mapping(
            "radiometric_quality",
            key="Radiometric_Quality",
            key_attr="bandId",
            key_cast=int,
            value_xpath="Noise_Model/BETA",
            value_cast=float,
        )


if __name__ == "__main__":
    pass
