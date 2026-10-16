"""eoio.readers.gistda_theos.metadata - extractor class for GISTDA THEOS metadata."""

from __future__ import annotations

import datetime as dt
import math
import warnings
from pathlib import Path

from processor_tools.utils.dict_tools import get_value

from eoio.readers.footprint_utils import normalize_footprint
from eoio.readers.metadata import BaseMetadataExtractor
from eoio.utils.dict_tools import dict_merge


class GISTDATHEOSMetadataExtractorError(ValueError):
    pass


def _earth_sun_distance_au(day_of_year: int) -> float:
    """Earth-Sun distance (astronomical units) on *day_of_year*, via the Spencer
    (1971) Fourier-series approximation of Earth's orbital eccentricity correction
    factor -- unlike Landsat (``EARTH_SUN_DISTANCE`` in its own MTL) or RadCalNet
    (reported directly in its own site files), THEOS's METADATA.DIM has no such
    field, so this is derived instead of read.

    ``(d0/d)^2 = 1.000110 + 0.034221*cos(Γ) + 0.001280*sin(Γ) + 0.000719*cos(2Γ)
    + 0.000077*sin(2Γ)``, where ``Γ = 2π*(day_of_year - 1)/365`` is the day angle
    (radians) and ``d0 = 1`` AU. Solving for ``d`` gives the distance returned here.
    Reference: Spencer, J.W. (1971), "Fourier series representation of the
    position of the sun", Search, 2(5), 172 -- also given in Duffie & Beckman,
    *Solar Engineering of Thermal Processes*. Checked against published
    perihelion/aphelion distances (~0.9833 AU early January, ~1.0167 AU early
    July): this formula gives 0.98291 AU and 1.01710 AU respectively, within
    ~0.0004 AU (well under the ~0.1% error in d^2 this feeds into the reflectance
    formula, negligible next to the overall radiometric uncertainty).

    :param day_of_year: day of year (1-366) the product was acquired.
    :return: Earth-Sun distance, astronomical units.
    """
    gamma = 2 * math.pi * (day_of_year - 1) / 365
    eccentricity_correction = (
        1.000110
        + 0.034221 * math.cos(gamma)
        + 0.001280 * math.sin(gamma)
        + 0.000719 * math.cos(2 * gamma)
        + 0.000077 * math.sin(2 * gamma)
    )
    return math.sqrt(1.0 / eccentricity_correction)


def _solar_zenith_from_elevation(sun_elevation: str | float | None) -> str | None:
    """``90 - SUN_ELEVATION`` (degrees), or ``None`` if *sun_elevation* itself is
    missing. get_basic_metadata (this function's only caller) is on every read's
    critical path (attach_metadata calls it unconditionally), so a bare
    ``float(None)`` here would crash the whole read the moment a product's
    Scene_Source block happens to omit SUN_ELEVATION, instead of leaving this one
    observation-geometry field unset.

    :param sun_elevation: ``Scene_Source.SUN_ELEVATION`` value, or ``None``.
    """
    if sun_elevation is None:
        return None
    return str(90 - float(sun_elevation))


def _parse_imaging_datetime(date_txt: str | None, time_txt: str | None) -> str | None:
    """Acquisition time as a naive UTC ISO string, from METADATA.DIM's ``IMAGING_DATE`` (``YYYY-MM-DD``) and
    ``IMAGING_TIME`` (``HH:MM:SS[.ffffff]``, optionally with a trailing ``Z``). ``None`` if either is missing
    or unparseable -- with a warning for the latter, since a read that goes on without a time is easy to miss.
    A strict ``%f`` format would reject a time with no fractional seconds and so fail the whole read."""
    if not (date_txt and time_txt):
        return None
    text = f"{date_txt}T{str(time_txt).strip().rstrip('Z')}"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    warnings.warn(f"Could not parse THEOS acquisition time {text!r}; product_datetime will be empty.")
    return None


def _dataset_frame_to_geojson_polygon(dataset_frame: dict | None) -> dict | None:
    """THEOS's ``<Dataset_Frame>`` (parsed by xmltodict into ``product_geospatial_bounds``)
    lists its 4 corners under ``Vertex`` with ``FRAME_LON``/``FRAME_LAT`` keys, which
    ``eoio.readers.footprint_utils.dict_geometry_to_shapely`` doesn't recognise (it only
    checks ``Lon``/``lon``) -- so passed through as-is, ``normalize_footprint`` silently
    drops the geometry (logs "Could not parse dict geometry", returns ``None``).

    Even with the right keys, the ``Vertex`` list is in image-corner scan order (by
    ``FRAME_ROW``/``FRAME_COL`` -- e.g. top-left, top-right, bottom-left, bottom-right),
    not ring order: connecting it as listed self-intersects into a bowtie of near-zero
    area instead of the scene's real footprint. Re-deriving corner order from each
    vertex's own FRAME_ROW/FRAME_COL (rather than trusting the XML's listed order) gives
    a proper ring regardless of which order a given product happens to list them in.

    :param dataset_frame: ``product_geospatial_bounds`` -- the raw xmltodict
        ``Dataset_Frame`` dict, or ``None``.
    :return: a GeoJSON-like ``{"type": "Polygon", "coordinates": [...]}`` dict (which
        ``normalize_footprint`` already handles via ``shapely.geometry.shape``), or
        ``None`` if *dataset_frame* isn't the expected 4-vertex shape.
    """
    if not dataset_frame or "Vertex" not in dataset_frame:
        return None
    vertices = dataset_frame["Vertex"]
    if isinstance(vertices, dict):
        vertices = [vertices]

    corners = []
    for v in vertices:
        try:
            corners.append(
                {
                    "lon": float(v["FRAME_LON"]),
                    "lat": float(v["FRAME_LAT"]),
                    "row": float(v["FRAME_ROW"]),
                    "col": float(v["FRAME_COL"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if len(corners) != 4:
        return None

    min_row = min(c["row"] for c in corners)
    max_row = max(c["row"] for c in corners)
    min_col = min(c["col"] for c in corners)
    max_col = max(c["col"] for c in corners)

    def _nearest(row: float, col: float) -> dict:
        return min(corners, key=lambda c: (c["row"] - row) ** 2 + (c["col"] - col) ** 2)

    top_left = _nearest(min_row, min_col)
    top_right = _nearest(min_row, max_col)
    bottom_right = _nearest(max_row, max_col)
    bottom_left = _nearest(max_row, min_col)
    ring = [top_left, top_right, bottom_right, bottom_left, top_left]
    return {"type": "Polygon", "coordinates": [[[c["lon"], c["lat"]] for c in ring]]}


class GISTDATHEOSMetadataExtractor(BaseMetadataExtractor):
    """Metadata helper for GISTDA THEOS MS scene products."""

    def __init__(self, reader):
        super().__init__(reader)
        self.metadata_filepath = Path(reader.layout.metadata_file())
        self._metadata_dim: dict | None = None

    def read_metadata_dim(self) -> dict:
        """METADATA.DIM as a dict. Parsed once -- it is asked for per band, and again for the
        Earth-Sun distance and the basic metadata."""
        if self._metadata_dim is None:
            import xmltodict

            with open(self.metadata_filepath, encoding="utf-8") as metadata_file:
                metadata_dict = xmltodict.parse(metadata_file.read())

            self._metadata_dim = dict_merge([{}, metadata_dict])
        return self._metadata_dim

    def get_basic_metadata(self) -> dict:
        full_md = self.read_metadata_dim()

        product_name = Path(self.reader.path.name).name if self.reader.path else ""
        product_datetime = _parse_imaging_datetime(
            get_value(full_md, "IMAGING_DATE"), get_value(full_md, "IMAGING_TIME")
        )

        basic_md = {
            "collection_name": "THEOS",
            "product_name": product_name,
            "platform": get_value(full_md, "MISSION") or "THEOS",
            "instrument": (get_value(full_md, "INSTRUMENT") or "TOP")
            + "_"
            + str(get_value(full_md, "INSTRUMENT_INDEX") or "2"),
            "processing_level": f"Level_{get_value(full_md, 'PROCESSING_LEVEL')}",
            "satellite_id": get_value(full_md, "MISSION") or "THEOS",
            "constellation": "THEOS",
            "spatial_resolution": [15, 15, 15, 15],
            "geometry_ids": ["15m", "15m", "15m", "15m"],
            "product_geospatial_bounds": get_value(full_md, "Dataset_Frame"),
            "located_geometric_values": get_value(full_md, "Geoposition_Points"),
            "product_datetime": product_datetime,
            "description": "GISTDA THEOS multispectral scene product",
            "institution": "GISTDA",
            "keywords": ["gistda", "theos", "satellite", "remote sensing", "multispectral"],
            "source": "GISTDA THEOS satellite product",
            "platform_type": "satellite",
            "observation_geometry": {
                "VIEWING_ANGLE_ALONG_TRACK".lower(): get_value(full_md, "VIEWING_ANGLE_ALONG_TRACK"),
                "VIEWING_ANGLE_ACROSS_TRACK".lower(): get_value(full_md, "VIEWING_ANGLE_ACROSS_TRACK"),
                "viewing_zenith_angle": get_value(full_md, "SATELLITE_INCIDENCE_ANGLE"),
                "viewing_azimuth_angle": get_value(full_md, "SATELLITE_AZIMUTH_ANGLE"),
                "SOLAR_AZIMUTH_angle".lower(): get_value(full_md, "SUN_AZIMUTH"),
                "solar_zenith_angle": _solar_zenith_from_elevation(get_value(full_md, "SUN_ELEVATION")),
            },
            "cloud_area_fraction": None,
        }

        crs = get_value(full_md, "HORIZONTAL_CS_CODE") or "EPSG:4326"
        basic_md["geospatial_bounds_crs"] = crs
        geojson_bounds = _dataset_frame_to_geojson_polygon(basic_md["product_geospatial_bounds"])
        basic_md["footprint"] = normalize_footprint(
            geometry_input=geojson_bounds if geojson_bounds is not None else basic_md["product_geospatial_bounds"],
            crs_input=4326,
        )
        return basic_md

    def get_product_metadata(self) -> dict:
        # TODO update to be full metadata
        return self.get_basic_metadata()

    def get_variable_product_metadata(self, var: str) -> dict:
        var_md: dict = {}

        # TODO: update obs geom specific metadata
        if var == "observation_geometry":
            var_md = {
                "units": "degrees",
                # "spatial_resolution_units": "m",
                # "geometry_id": "15",
            }
            return var_md

        full_md = self.read_metadata_dim()

        band_parameters = get_value(full_md, "Band_Parameters") or []
        if isinstance(band_parameters, dict):  # xmltodict gives a lone child as a dict, not a one-item list
            band_parameters = [band_parameters]
        for band_dict in band_parameters:
            if var[-1] == band_dict["BAND_INDEX"]:
                gain = get_value(band_dict, "PHYSICAL_GAIN")
                bias = get_value(band_dict, "PHYSICAL_BIAS")
                break
        else:
            raise GISTDATHEOSMetadataExtractorError(
                f"No Band_Parameters entry found for variable {var!r} "
                f"(looked for BAND_INDEX == {var[-1]!r}) -- can't determine its gain/bias, "
                "so its DNs can't be converted to radiance."
            )

        return {
            "band_id": var,
            "band_name": var,
            "standard_name": "toa_radiance",
            "long_name": f"TOA radiance in band {var}",
            "units": "W/(m²·sr·μm)",
            "spatial_resolution": 15,
            "spatial_resolution_units": "m",
            "geometry_id": "15m",
            "ancillary_variables": [],
            "measurand": "radiance",
            "band_gain": gain,
            "band_bias": bias,
            "band_solar_irradiance": self.get_solar_irradiance(var),
            "earth_sun_distance": self.get_earth_sun_distance(),
            "band_central_wavelength": self.get_central_wavelength(var),
            "band_central_wavelength_units": "nm",
        }

    def get_variable_basic_metadata(self, var: str) -> dict:
        basic_var_md: dict = {}

        # aux metadata done within aux module so pass here
        # TODO: update obs geom specific metadata
        if var == "observation_geometry":
            var_md = {
                "units": "degrees",
                # "spatial_resolution_units": "m",
                # "geometry_id": "15",
            }
            return var_md

        basic_var_md = {
            "standard_name": "toa_radiance",
            "long_name": f"TOA radiance in band {var}",
            "units": "W/(m²·sr·μm)",
            "measurand": "radiance",
        }

        return basic_var_md

    def get_solar_irradiance(self, var: str) -> float:
        # Values provided in Spectral Resonse + ESUN xml file.
        E_sun = {"B1": 1980.359954, "B2": 1809.344616, "B3": 1535.632489, "B4": 1069.662891}

        return E_sun[var]

    def get_central_wavelength(self, var: str) -> float:
        """*var*'s band centre wavelength (nm) -- THEOS-1 MS payload spec: B1 (blue)
        483, B2 (green) 565, B3 (red) 659, B4 (NIR) 830. Used as the stack processor's
        ``dim_coord_attr`` (``product_metadata.band_central_wavelength``), matching
        S2/Landsat's own convention, so the ``band`` dimension coordinate carries real
        wavelengths (with band codes kept alongside as ``band_name``) instead of falling
        back to band codes as the sole coordinate.
        """
        central_wavelength_nm = {"B1": 483.0, "B2": 565.0, "B3": 659.0, "B4": 830.0}

        return central_wavelength_nm[var]

    def get_earth_sun_distance(self) -> float:
        """Earth-Sun distance (AU) on this product's own acquisition date -- see
        :py:func:`_earth_sun_distance_au`.

        :raises GISTDATHEOSMetadataExtractorError: ``IMAGING_DATE`` is missing or
            unparseable, since reflectance can't be correctly computed without it
            (unlike a missing solar zenith angle, which is tolerated -- see
            :py:func:`_solar_zenith_from_elevation` -- there's no equivalently
            honest "unset" value for a multiplicative distance-squared term).
        """
        date_txt = get_value(self.read_metadata_dim(), "IMAGING_DATE")
        if not date_txt:
            raise GISTDATHEOSMetadataExtractorError(
                "Scene_Source.IMAGING_DATE is missing -- can't derive the Earth-Sun distance "
                "needed to convert this product's radiance to TOA reflectance."
            )
        try:
            acquisition_date = dt.datetime.strptime(date_txt, "%Y-%m-%d").date()
        except ValueError as exc:
            raise GISTDATHEOSMetadataExtractorError(
                f"Scene_Source.IMAGING_DATE {date_txt!r} is not in the expected YYYY-MM-DD format."
            ) from exc
        return _earth_sun_distance_au(acquisition_date.timetuple().tm_yday)
