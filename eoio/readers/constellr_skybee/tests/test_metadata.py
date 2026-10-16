"""eoio.readers.constellr_skybee.tests.test_metadata - unit tests for eoio.readers.constellr_skybee.metadata"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from eoio.readers.constellr_skybee.metadata import (
    ConstellrSkybeeMetadataExtractor,
    ConstellrSkybeeMetadataExtractorError,
)


class TestConstellrSkybeeMetadataExtractor(unittest.TestCase):
    def _make_metadata_file(self, root: Path) -> Path:
        metadata = {
            "product_id": "HIVE_L1C_TEST",
            "product_type": "L1C",
            "platform": "Skybee",
            "acquisition_datetime": "2024-05-01T12:00:00Z",
            "total_cloud_percentage": 5,
            "earth_sun_distance": 1.0166,
            "georef_accuracy": {"sentinel2_reference": {"epsg": 4326}},
            "gver_metrics": {"gver_abs": {"VNIR02": {"num_points": 1}}},
            "geometric": {
                "geometry": {"type": "Polygon"},
                "solar_angles": {"view:sun_azimuth": 123.0, "view:sun_elevation": 45.0},
                "viewing_angles": {"view:azimuth": 250.0, "view:incident_angle": 30.0, "view:off_nadir": 27.0},
            },
            "sensors": {
                "VNIR": {
                    "sensor_id": ["123"],
                    "bands": {
                        f"VNIR0{i}": {
                            "name": f"VNIR0{i}",
                            "eo:common_name": "blue",
                            "band_centre": 0.49,
                            "band_width": 0.065,
                            "esun": {"value": v},
                        }
                        for i, v in zip(range(1, 5), (1500.0, 1700.0, 2000.0, 2100.0))
                    },
                }
            },
        }
        path = root / "metadata.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def _make_reader(self, metadata_path: Path):
        reader = Mock()
        reader.path = metadata_path.parent
        reader.layout = Mock()
        reader.layout.metadata_file = Mock(return_value=metadata_path)
        reader.meas_var_res = {"B1": 60, "B2": 10, "B3": 10, "B4": 10}
        return reader

    def _extractor(self, td: str, **overrides):
        """An extractor over a metadata.json with *overrides* applied at the top level (None deletes)."""
        path = self._make_metadata_file(Path(td))
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in overrides.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        path.write_text(json.dumps(data), encoding="utf-8")
        return ConstellrSkybeeMetadataExtractor(self._make_reader(path))

    def test_cloud_cover_is_unknown_when_constellr_did_not_compute_it(self):
        """total_cloud_percentage is a placeholder then; reporting it would read as a clear scene."""
        with TemporaryDirectory() as td:
            extractor = self._extractor(td, cloud_cover_computed=False, total_cloud_percentage=0)
            self.assertIsNone(extractor.get_basic_metadata()["cloud_area_fraction"])

    def test_cloud_cover_reported_when_computed(self):
        with TemporaryDirectory() as td:
            extractor = self._extractor(td, cloud_cover_computed=True, total_cloud_percentage=60.1)
            self.assertEqual(extractor.get_basic_metadata()["cloud_area_fraction"], 60.1)

    def test_missing_angles_do_not_stop_the_product_being_read(self):
        with TemporaryDirectory() as td:
            path = self._make_metadata_file(Path(td))
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["geometric"]["solar_angles"]
            path.write_text(json.dumps(data), encoding="utf-8")
            basic = ConstellrSkybeeMetadataExtractor(self._make_reader(path)).get_basic_metadata()
            self.assertNotIn("solar_zenith_angle", basic["observation_geometry"])
            self.assertEqual(basic["observation_geometry"]["viewing_zenith_angle"], 30.0)

    def test_satellite_id_is_none_when_the_platform_is_absent(self):
        with TemporaryDirectory() as td:
            self.assertIsNone(self._extractor(td, platform=None).get_basic_metadata()["satellite_id"])

    def test_unreadable_metadata_json_warns_and_is_empty(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "metadata.json"
            path.write_text("{not json", encoding="utf-8")
            extractor = ConstellrSkybeeMetadataExtractor(self._make_reader(path))
            with self.assertWarnsRegex(UserWarning, "Could not read"):
                self.assertEqual(extractor.read_metadata(), {})

    def test_metadata_json_is_parsed_once(self):
        with TemporaryDirectory() as td:
            extractor = self._extractor(td)
            self.assertIs(extractor.read_metadata(), extractor.read_metadata())

    def test_band_missing_from_the_metadata_is_a_clear_error(self):
        with TemporaryDirectory() as td:
            extractor = self._extractor(td)
            with self.assertRaisesRegex(ConstellrSkybeeMetadataExtractorError, "VNIR09"):
                extractor.get_variable_product_metadata("B9")

    def test_read_metadata_returns_dict(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = self._make_metadata_file(root)
            reader = self._make_reader(metadata_path)
            extractor = ConstellrSkybeeMetadataExtractor(reader)

            self.assertIsInstance(extractor.read_metadata(), dict)
            self.assertEqual(extractor.read_metadata()["product_id"], "HIVE_L1C_TEST")

    def test_get_basic_metadata(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = self._make_metadata_file(root)
            reader = self._make_reader(metadata_path)
            extractor = ConstellrSkybeeMetadataExtractor(reader)

            basic = extractor.get_basic_metadata()
            self.assertEqual(basic["platform"], "Skybee")
            self.assertEqual(basic["instrument"], "VNIR")
            self.assertEqual(basic["product_name"], "HIVE_L1C_TEST")
            self.assertEqual(basic["geospatial_bounds_crs"], "EPSG:4326")
            self.assertEqual(basic["cloud_area_fraction"], 5)
            self.assertEqual(basic["collection_name"], "HIVE")
            self.assertEqual(basic["constellation"], "HIVE")
            self.assertEqual(basic["satellite_id"], "SKYBEE")
            self.assertEqual(basic["product_datetime"], "2024-05-01T12:00:00")
            self.assertEqual(basic["observation_geometry"]["viewing_zenith_angle"], 30.0)
            self.assertEqual(basic["observation_geometry"]["solar_zenith_angle"], 45.0)

    def test_get_product_metadata(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = self._make_metadata_file(root)
            reader = self._make_reader(metadata_path)
            extractor = ConstellrSkybeeMetadataExtractor(reader)

            product_md = extractor.get_product_metadata()
            self.assertIn("platform", product_md)
            self.assertIn("instrument", product_md)
            self.assertEqual(product_md["product_name"], "HIVE_L1C_TEST")

    def test_get_variable_product_metadata(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = self._make_metadata_file(root)
            reader = self._make_reader(metadata_path)
            extractor = ConstellrSkybeeMetadataExtractor(reader)

            band_md = extractor.get_variable_product_metadata("B1")
            self.assertEqual(band_md["band_id"], "B1")
            self.assertEqual(band_md["band_name"], "B1")
            self.assertEqual(band_md["spatial_resolution"], 60)
            self.assertEqual(band_md["band_solar_irradiance"], 1500.0)
            self.assertEqual(extractor.get_variable_product_metadata("B2")["band_solar_irradiance"], 1700.0)
            self.assertEqual(band_md["band_central_wavelength"], 490.0)
            self.assertEqual(band_md["band_central_wavelength_units"], "nm")
            self.assertEqual(band_md["band_full_width_half_max"], 65.0)
            self.assertEqual(band_md["earth_sun_distance"], 1.0166)
            self.assertEqual(band_md["measurand"], "radiance")

    def test_get_variable_metadata_for_mask_does_not_look_up_band(self):
        with TemporaryDirectory() as td:
            metadata_path = self._make_metadata_file(Path(td))
            extractor = ConstellrSkybeeMetadataExtractor(self._make_reader(metadata_path))

            self.assertEqual(extractor.get_variable_product_metadata("blackfill_mask")["geometry_id"], "10m")
            self.assertEqual(extractor.get_variable_basic_metadata("blackfill_mask"), {})

    def test_get_variable_basic_metadata(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = self._make_metadata_file(root)
            reader = self._make_reader(metadata_path)
            extractor = ConstellrSkybeeMetadataExtractor(reader)

            band_md = extractor.get_variable_basic_metadata("B2")
            self.assertEqual(band_md["standard_name"], "toa_radiance")
            self.assertEqual(band_md["units"], "W/(m²·sr·μm)")


if __name__ == "__main__":
    unittest.main()
