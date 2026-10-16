"""eoio.readers.gistda_theos.tests.test_metadata - unit tests for eoio.readers.gistda_theos.metadata"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from eoio.readers.gistda_theos.metadata import (
    GISTDATHEOSMetadataExtractor,
    GISTDATHEOSMetadataExtractorError,
    _dataset_frame_to_geojson_polygon,
    _earth_sun_distance_au,
    _parse_imaging_datetime,
    _solar_zenith_from_elevation,
)


class TestGistdaTheosMetadataExtractor(unittest.TestCase):
    """Unit tests for GistdaTheosMetadataExtractor class."""

    def _make_minimal_dim(
        self, root: Path, *, sun_elevation: str | None = "  52.628636", imaging_date: str | None = "2022-02-09"
    ) -> Path:
        sun_elevation_element = f"<SUN_ELEVATION>{sun_elevation}</SUN_ELEVATION>" if sun_elevation is not None else ""
        imaging_date_element = f"<IMAGING_DATE>{imaging_date}</IMAGING_DATE>" if imaging_date is not None else ""
        dim = root / "METADATA.DIM"
        dim.write_text(
            f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Dimap_Document>
  <Dataset_Sources>
    <Source_Information>
      <Scene_Source>
        <MISSION>THEOS</MISSION>
        <INSTRUMENT>TOP</INSTRUMENT>
        <INSTRUMENT_INDEX>2</INSTRUMENT_INDEX>
        {imaging_date_element}
        <IMAGING_TIME>08:38:45.550720</IMAGING_TIME>
        <VIEWING_ANGLE_ALONG_TRACK>   0.555224</VIEWING_ANGLE_ALONG_TRACK>
        <VIEWING_ANGLE_ACROSS_TRACK>  -9.370350</VIEWING_ANGLE_ACROSS_TRACK>
        <SATELLITE_INCIDENCE_ANGLE>  10.634494</SATELLITE_INCIDENCE_ANGLE>
        <SATELLITE_AZIMUTH_ANGLE> 100.364357</SATELLITE_AZIMUTH_ANGLE>
        <SUN_AZIMUTH>  84.182195</SUN_AZIMUTH>
        {sun_elevation_element}
      </Scene_Source>
    </Source_Information>
  </Dataset_Sources>
  <Coordinate_Reference_System>
    <Horizontal_CS>
      <HORIZONTAL_CS_CODE>epsg:4326</HORIZONTAL_CS_CODE>
    </Horizontal_CS>
  </Coordinate_Reference_System>
  <Band_Parameters>
    <BAND_INDEX>1</BAND_INDEX>
      <Gain_Section>
        <PHYSICAL_GAIN>7.677</PHYSICAL_GAIN>
        <PHYSICAL_BIAS>0.00</PHYSICAL_BIAS>
      </Gain_Section>
  </Band_Parameters>
  <Band_Parameters>
    <BAND_INDEX>2</BAND_INDEX>
      <Gain_Section>
        <PHYSICAL_GAIN>9.455</PHYSICAL_GAIN>
        <PHYSICAL_BIAS>0.00</PHYSICAL_BIAS>
      </Gain_Section>
  </Band_Parameters>
</Dimap_Document>
""",
            encoding="utf-8",
        )
        return dim

    def _make_mock_reader(self, dim_path: Path):
        reader = Mock()
        reader.path = dim_path.parent / "TH_CAT_22021001412034_1"
        reader.layout = Mock()
        reader.layout.metadata_file.return_value = str(dim_path)
        return reader

    def test_metadata_dim_is_parsed_once(self):
        with TemporaryDirectory() as td:
            extractor = GISTDATHEOSMetadataExtractor(self._make_mock_reader(self._make_minimal_dim(Path(td))))
            self.assertIs(extractor.read_metadata_dim(), extractor.read_metadata_dim())

    def test_a_lone_band_parameters_block_is_still_found(self):
        """xmltodict returns a single child as a dict, not a one-item list."""
        with TemporaryDirectory() as td:
            extractor = GISTDATHEOSMetadataExtractor(self._make_mock_reader(self._make_minimal_dim(Path(td))))
            full = extractor.read_metadata_dim()
            blocks = full["Dimap_Document"]["Band_Parameters"]
            full["Dimap_Document"]["Band_Parameters"] = blocks[0]  # as xmltodict would for a single block
            self.assertEqual(extractor.get_variable_product_metadata("B1")["band_gain"], "7.677")

    def test_read_metadata_dim(self):
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td))
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            result = extractor.read_metadata_dim()

            self.assertIsInstance(result, dict)
            self.assertIn("Dimap_Document", result)

    def test_get_basic_metadata(self):
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td))
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            result = extractor.get_basic_metadata()

            self.assertIsInstance(result, dict)
            self.assertIn("collection_name", result)
            self.assertIn("product_datetime", result)
            self.assertIn("geospatial_bounds_crs", result)
            self.assertEqual(result["platform"], "THEOS")
            self.assertEqual(result["institution"], "GISTDA")

    def test_get_product_metadata_returns_basic_metadata(self):
        with TemporaryDirectory() as td:
            extractor = GISTDATHEOSMetadataExtractor(
                self._make_mock_reader(self._make_minimal_dim(Path(td)))
            )

            self.assertEqual(extractor.get_product_metadata(), extractor.get_basic_metadata())

    def test_get_variable_product_metadata_for_observation_geometry(self):
        with TemporaryDirectory() as td:
            extractor = GISTDATHEOSMetadataExtractor(
                self._make_mock_reader(self._make_minimal_dim(Path(td)))
            )

            self.assertEqual(
                extractor.get_variable_product_metadata("observation_geometry"),
                {"units": "degrees"},
            )

    def test_get_variable_basic_metadata(self):
        with TemporaryDirectory() as td:
            extractor = GISTDATHEOSMetadataExtractor(
                self._make_mock_reader(self._make_minimal_dim(Path(td)))
            )

            self.assertEqual(
                extractor.get_variable_basic_metadata("B2"),
                {
                    "standard_name": "toa_radiance",
                    "long_name": "TOA radiance in band B2",
                    "units": "W/(m²·sr·μm)",
                    "measurand": "radiance",
                },
            )
            self.assertEqual(
                extractor.get_variable_basic_metadata("observation_geometry"),
                {"units": "degrees"},
            )

    def test_band_solar_irradiance_and_wavelength_values(self):
        with TemporaryDirectory() as td:
            extractor = GISTDATHEOSMetadataExtractor(
                self._make_mock_reader(self._make_minimal_dim(Path(td)))
            )

            expected = {
                "B1": (1980.359954, 483.0),
                "B2": (1809.344616, 565.0),
                "B3": (1535.632489, 659.0),
                "B4": (1069.662891, 830.0),
            }
            for band, (irradiance, wavelength) in expected.items():
                with self.subTest(band=band):
                    self.assertEqual(extractor.get_solar_irradiance(band), irradiance)
                    self.assertEqual(extractor.get_central_wavelength(band), wavelength)

    def test_get_variable_product_metadata(self):
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td))
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            result = extractor.get_variable_product_metadata("B1")

            self.assertIsInstance(result, dict)
            self.assertIn("band_id", result)
            self.assertEqual(result["spatial_resolution"], 15)
            # TOA radiance; conversion to reflectance is the units processor's job
            self.assertEqual(result["standard_name"], "toa_radiance")
            self.assertEqual(result["measurand"], "radiance")
            self.assertEqual(result["units"], "W/(m²·sr·μm)")
            self.assertAlmostEqual(result["earth_sun_distance"], _earth_sun_distance_au(40), places=6)
            self.assertEqual(result["band_central_wavelength"], 483.0)
            self.assertEqual(result["band_central_wavelength_units"], "nm")

    def test_get_variable_product_metadata_raises_a_clear_error_for_an_unmatched_band(self):
        """No Band_Parameters entry has BAND_INDEX '9' -- must raise a clear error,
        not UnboundLocalError from gain/bias never being assigned."""
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td))
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            with self.assertRaises(GISTDATHEOSMetadataExtractorError) as ctx:
                extractor.get_variable_product_metadata("B9")
            self.assertIn("B9", str(ctx.exception))

    def test_viewing_zenith_angle_is_the_ground_incidence_angle(self):
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td))
            extractor = GISTDATHEOSMetadataExtractor(self._make_mock_reader(dim_path))

            vza = extractor.get_basic_metadata()["observation_geometry"]["viewing_zenith_angle"]

            self.assertEqual(vza, "10.634494")

    def test_get_basic_metadata_when_sun_elevation_is_missing(self):
        """A product whose Scene_Source omits SUN_ELEVATION must not crash the
        whole read -- solar_zenith_angle is left None instead."""
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td), sun_elevation=None)
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            result = extractor.get_basic_metadata()

            self.assertIsNone(result["observation_geometry"]["solar_zenith_angle"])
            # the rest of observation_geometry is unaffected
            self.assertEqual(result["observation_geometry"]["solar_azimuth_angle"], "84.182195")

    def test_get_earth_sun_distance(self):
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td), imaging_date="2022-02-09")
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            result = extractor.get_earth_sun_distance()

            # 2022-02-09 is day-of-year 40 (31 + 9).
            self.assertAlmostEqual(result, _earth_sun_distance_au(40), places=9)
            # sanity bound -- Earth-Sun distance is always within [perihelion, aphelion].
            self.assertTrue(0.9825 < result < 1.0175)

    def test_get_earth_sun_distance_raises_when_imaging_date_missing(self):
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td), imaging_date=None)
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            with self.assertRaises(GISTDATHEOSMetadataExtractorError) as ctx:
                extractor.get_earth_sun_distance()
            self.assertIn("IMAGING_DATE", str(ctx.exception))

    def test_get_earth_sun_distance_raises_on_unparseable_date(self):
        with TemporaryDirectory() as td:
            dim_path = self._make_minimal_dim(Path(td), imaging_date="09/02/2022")
            reader = self._make_mock_reader(dim_path)

            extractor = GISTDATHEOSMetadataExtractor(reader)
            with self.assertRaises(GISTDATHEOSMetadataExtractorError) as ctx:
                extractor.get_earth_sun_distance()
            self.assertIn("09/02/2022", str(ctx.exception))


class TestParseImagingDatetime(unittest.TestCase):
    def test_with_fractional_seconds(self):
        self.assertEqual(_parse_imaging_datetime("2022-02-09", "08:38:45.550720"), "2022-02-09T08:38:45.550720")

    def test_without_fractional_seconds(self):
        """A strict %f format rejected this and failed the whole read."""
        self.assertEqual(_parse_imaging_datetime("2022-02-09", "08:38:45"), "2022-02-09T08:38:45")

    def test_trailing_z_is_accepted(self):
        self.assertEqual(_parse_imaging_datetime("2022-02-09", "08:38:45.5Z"), "2022-02-09T08:38:45.500000")

    def test_missing_parts_give_none_without_a_warning(self):
        self.assertIsNone(_parse_imaging_datetime(None, "08:38:45"))
        self.assertIsNone(_parse_imaging_datetime("2022-02-09", None))

    def test_unparseable_gives_none_with_a_warning(self):
        with self.assertWarnsRegex(UserWarning, "Could not parse"):
            self.assertIsNone(_parse_imaging_datetime("2022-02-09", "noon"))


class TestSolarZenithFromElevation(unittest.TestCase):
    def test_computes_zenith_from_elevation(self):
        self.assertEqual(_solar_zenith_from_elevation("52.628636"), str(90 - 52.628636))

    def test_none_when_elevation_missing(self):
        self.assertIsNone(_solar_zenith_from_elevation(None))


class TestEarthSunDistanceAu(unittest.TestCase):
    """Regression tests for the Spencer (1971) Earth-Sun distance approximation --
    see _earth_sun_distance_au's own docstring for the formula and reference."""

    def test_returns_a_float_in_au(self):
        self.assertIsInstance(_earth_sun_distance_au(1), float)

    def test_matches_published_perihelion_distance(self):
        """Earth is nearest the sun around day 4 (~Jan 4) -- published perihelion
        distance is ~0.9833 AU."""
        self.assertAlmostEqual(_earth_sun_distance_au(4), 0.98291, delta=0.0005)

    def test_matches_published_aphelion_distance(self):
        """Earth is farthest from the sun around day 186 (~Jul 5) -- published
        aphelion distance is ~1.0167 AU."""
        self.assertAlmostEqual(_earth_sun_distance_au(186), 1.01710, delta=0.0005)

    def test_always_within_orbital_bounds(self):
        """Every day of the year must fall within Earth's actual perihelion/aphelion
        bounds -- a sign error or misplaced factor would push values outside this."""
        for doy in range(1, 367):
            d = _earth_sun_distance_au(doy)
            self.assertTrue(0.9825 < d < 1.0175, f"day {doy}: {d} AU out of bounds")


class TestDatasetFrameToGeojsonPolygon(unittest.TestCase):
    def test_missing_or_non_four_vertex_frames_return_none(self):
        self.assertIsNone(_dataset_frame_to_geojson_polygon(None))
        self.assertIsNone(_dataset_frame_to_geojson_polygon({}))
        self.assertIsNone(_dataset_frame_to_geojson_polygon({"Vertex": []}))
        self.assertIsNone(_dataset_frame_to_geojson_polygon({"Vertex": {"FRAME_LON": 1}}))

    def test_malformed_vertices_are_skipped(self):
        frame = {
            "Vertex": [
                {"FRAME_LON": "1", "FRAME_LAT": "1", "FRAME_ROW": "0", "FRAME_COL": "0"},
                {"FRAME_LON": "2", "FRAME_LAT": "1", "FRAME_ROW": "0", "FRAME_COL": "1"},
                {"FRAME_LON": "2", "FRAME_LAT": "2", "FRAME_ROW": "1", "FRAME_COL": "1"},
                {"FRAME_LON": "bad", "FRAME_LAT": "2", "FRAME_ROW": "1", "FRAME_COL": "0"},
            ]
        }

        self.assertIsNone(_dataset_frame_to_geojson_polygon(frame))

    def test_vertices_are_reordered_into_a_closed_polygon_ring(self):
        frame = {
            "Vertex": [
                {"FRAME_LON": "2", "FRAME_LAT": "2", "FRAME_ROW": "10", "FRAME_COL": "10"},
                {"FRAME_LON": "1", "FRAME_LAT": "1", "FRAME_ROW": "0", "FRAME_COL": "0"},
                {"FRAME_LON": "1", "FRAME_LAT": "2", "FRAME_ROW": "10", "FRAME_COL": "0"},
                {"FRAME_LON": "2", "FRAME_LAT": "1", "FRAME_ROW": "0", "FRAME_COL": "10"},
            ]
        }

        self.assertEqual(
            _dataset_frame_to_geojson_polygon(frame),
            {
                "type": "Polygon",
                "coordinates": [[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]]],
            },
        )


if __name__ == "__main__":
    unittest.main()
