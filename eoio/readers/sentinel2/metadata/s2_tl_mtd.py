"""eoio.readers.sentinel2.metadata.s2_tl_mtd - reader for Sentinel-2 tile metadata."""

from typing import Optional
import datetime as dt
import numpy as np
from eoio.readers.xml import XMLReader

# Location of fields of interest within the tile metadata file
S2_TL_MTD_METADATA_PATHS: dict[str, str] = {
    # ------------------------------------------------------------------
    # General info
    # ------------------------------------------------------------------
    "tile_id": "./n1:General_Info/TILE_ID",
    "datastrip_id": "./n1:General_Info/DATASTRIP_ID",
    "downlink_priority": "./n1:General_Info/DOWNLINK_PRIORITY",
    "sensing_time": "./n1:General_Info/SENSING_TIME",
    "archiving_time": "./n1:General_Info/Archiving_Info/ARCHIVING_TIME",
    "archiving_centre": "./n1:General_Info/Archiving_Info/ARCHIVING_CENTRE",
    # ------------------------------------------------------------------
    # Tile geocoding
    # ------------------------------------------------------------------
    "horizontal_cs_name": "./n1:Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_NAME",
    "horizontal_cs_code": "./n1:Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_CODE",
    # Size per resolution
    "nrows_10": "./n1:Geometric_Info/Tile_Geocoding/Size[@resolution='10']/NROWS",
    "ncols_10": "./n1:Geometric_Info/Tile_Geocoding/Size[@resolution='10']/NCOLS",
    "nrows_20": "./n1:Geometric_Info/Tile_Geocoding/Size[@resolution='20']/NROWS",
    "ncols_20": "./n1:Geometric_Info/Tile_Geocoding/Size[@resolution='20']/NCOLS",
    "nrows_60": "./n1:Geometric_Info/Tile_Geocoding/Size[@resolution='60']/NROWS",
    "ncols_60": "./n1:Geometric_Info/Tile_Geocoding/Size[@resolution='60']/NCOLS",
    # Geoposition per resolution
    "ulx_10": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='10']/ULX",
    "uly_10": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='10']/ULY",
    "xdim_10": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='10']/XDIM",
    "ydim_10": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='10']/YDIM",
    "ulx_20": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='20']/ULX",
    "uly_20": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='20']/ULY",
    "xdim_20": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='20']/XDIM",
    "ydim_20": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='20']/YDIM",
    "ulx_60": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='60']/ULX",
    "uly_60": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='60']/ULY",
    "xdim_60": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='60']/XDIM",
    "ydim_60": "./n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='60']/YDIM",
    # ------------------------------------------------------------------
    # Sun angles (mean + grids)
    # ------------------------------------------------------------------
    "mean_sun_zenith": "./n1:Geometric_Info/Tile_Angles/Mean_Sun_Angle/ZENITH_ANGLE",
    "mean_sun_azimuth": "./n1:Geometric_Info/Tile_Angles/Mean_Sun_Angle/AZIMUTH_ANGLE",
    "sun_zenith_col_step": "./n1:Geometric_Info/Tile_Angles/Sun_Angles_Grid/Zenith/COL_STEP",
    "sun_zenith_row_step": "./n1:Geometric_Info/Tile_Angles/Sun_Angles_Grid/Zenith/ROW_STEP",
    "sun_azimuth_col_step": "./n1:Geometric_Info/Tile_Angles/Sun_Angles_Grid/Azimuth/COL_STEP",
    "sun_azimuth_row_step": "./n1:Geometric_Info/Tile_Angles/Sun_Angles_Grid/Azimuth/ROW_STEP",
    # NOTE: these select multiple <VALUES> row elements
    "sun_zenith_values_rows": "./n1:Geometric_Info/Tile_Angles/Sun_Angles_Grid/Zenith/Values_List/VALUES",
    "sun_azimuth_values_rows": "./n1:Geometric_Info/Tile_Angles/Sun_Angles_Grid/Azimuth/Values_List/VALUES",
    # ------------------------------------------------------------------
    # Quality indicators (tile-level)
    # ------------------------------------------------------------------
    "cloudy_pixel_percentage": ("./n1:Quality_Indicators_Info/Image_Content_QI/CLOUDY_PIXEL_PERCENTAGE"),
    "snow_pixel_percentage": ("./n1:Quality_Indicators_Info/Image_Content_QI/SNOW_PIXEL_PERCENTAGE"),
    "degraded_msi_data_percentage": ("./n1:Quality_Indicators_Info/Image_Content_QI/DEGRADED_MSI_DATA_PERCENTAGE"),
    "pvi_filename": "./n1:Quality_Indicators_Info/PVI_FILENAME",
    # Pixel-level masks live under Pixel_Level_QI as repeated MASK_FILENAME elements
    "mask_filenames": "./n1:Quality_Indicators_Info/Pixel_Level_QI/MASK_FILENAME",
}


class S2TLXMLReader(XMLReader):
    """
    Sentinel-2 tile metadata reader for ``MTD_TL.xml``.
    """

    metadata_paths = S2_TL_MTD_METADATA_PATHS

    # ------------------------------------------------------------------
    # Simple scalar accessors (thin wrappers)
    # ------------------------------------------------------------------

    def find_tile_id(self) -> str:
        """Return the tile identifier (TILE_ID)."""
        return self.find_value("tile_id")

    def find_datastrip_id(self) -> str:
        """Return the datastrip identifier (DATASTRIP_ID)."""
        return self.find_value("datastrip_id")

    def find_downlink_priority(self) -> str:
        """Return the downlink priority."""
        return self.find_value("downlink_priority")

    def find_sensing_time_str(self) -> str:
        """Return the sensing time string (SENSING_TIME)."""
        return self.find_value("sensing_time")

    def find_archiving_time_str(self) -> str:
        """Return the archiving time string (ARCHIVING_TIME)."""
        return self.find_value("archiving_time")

    def find_horizontal_cs_name(self) -> str:
        """Return the horizontal CRS name."""
        return self.find_value("horizontal_cs_name", split=None)

    def find_horizontal_cs_code(self) -> str:
        """Return the horizontal CRS code (e.g. EPSG:32630)."""
        return self.find_value("horizontal_cs_code")

    # ------------------------------------------------------------------
    # Light parsing helpers (keep simple)
    # ------------------------------------------------------------------

    def find_sensing_datetime(self) -> dt.datetime:
        """
        Return the sensing time as a timezone-aware UTC datetime.
        """
        s = self.find_sensing_time_str()
        try:
            return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)

    # ------------------------------------------------------------------
    # Tile sizes / geoposition (thin wrappers)
    # ------------------------------------------------------------------

    def find_tile_shape(self, resolution: int) -> tuple[int, int]:
        """
        Return (nrows, ncols) for the given resolution (10/20/60).
        """
        nrows = self.find_value(f"nrows_{resolution}")
        ncols = self.find_value(f"ncols_{resolution}")
        return int(nrows), int(ncols)

    def find_geoposition(self, resolution: int) -> dict[str, float]:
        """
        Return geoposition parameters for the given resolution (ULX/ULY/XDIM/YDIM).
        """
        return {
            "ulx": self.find_value(f"ulx_{resolution}"),
            "uly": self.find_value(f"uly_{resolution}"),
            "xdim": self.find_value(f"xdim_{resolution}"),
            "ydim": self.find_value(f"ydim_{resolution}"),
        }

    def build_xy_coords(self, resolution: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Build 1-D x and y coordinate vectors for the tile grid (pixel centres).
        """
        nrows, ncols = self.find_tile_shape(resolution)
        geo = self.find_geoposition(resolution)

        ulx, uly, xdim, ydim = geo["ulx"], geo["uly"], geo["xdim"], geo["ydim"]
        x = ulx + (np.arange(ncols, dtype=float) + 0.5) * xdim
        y = uly + (np.arange(nrows, dtype=float) + 0.5) * ydim
        return x, y

    # ------------------------------------------------------------------
    # Quality indicators (thin wrappers)
    # ------------------------------------------------------------------

    def find_cloudy_pixel_percentage(self) -> float:
        """
        Return the CLOUDY_PIXEL_PERCENTAGE value (0–100).
        """
        return self.find_value("cloudy_pixel_percentage")

    def find_snow_pixel_percentage(self) -> Optional[float]:
        """
        Return the SNOW_PIXEL_PERCENTAGE value (0–100).

        NB: Not defined in PSD < v15.0
        """
        return self.find_value("snow_pixel_percentage")

    def find_degraded_msi_data_percentage(self) -> float:
        """
        Return the DEGRADED_MSI_DATA_PERCENTAGE value (0–100).
        """
        return self.find_value("degraded_msi_data_percentage")

    def find_pvi_filename(self) -> str:
        """
        Return the preview image filename (PVI_FILENAME) relative to the SAFE root.
        """
        return self.find_value("pvi_filename")

    def find_mask_filenames(self) -> dict[tuple[str, int | None], str]:
        """
        Return mask filenames as a mapping keyed by (mask_type, band_id).

        Keys are:
        - (type, bandId) for masks that include bandId
        - (type, None) for masks without bandId (e.g. MSK_CLASSI)

        Values are the corresponding MASK_FILENAME strings.
        """
        elems = self.xml_root.findall(self.metadata_paths["mask_filenames"], self.xml_ns)

        out: dict[tuple[str, int | None], str] = {}
        for e in elems:
            mask_type = e.attrib.get("type") or ""
            band_raw = e.attrib.get("bandId")  # optional
            band_id = int(band_raw) if band_raw is not None else None
            out[(mask_type, band_id)] = (e.text or "").strip()
        return out

    # ------------------------------------------------------------------
    # Sun angles
    # ------------------------------------------------------------------

    def find_mean_sun_angles(self) -> tuple[float, float]:
        """
        Return (mean_sun_zenith_deg, mean_sun_azimuth_deg).
        """
        return self.find_value("mean_sun_zenith"), self.find_value("mean_sun_azimuth")

    def find_sun_angle_steps(self) -> tuple[float, float]:
        """
        Return (col_step_m, row_step_m) for the sun angles grids.
        """
        return self.find_value("sun_zenith_col_step"), self.find_value("sun_zenith_row_step")

    def find_sun_angle_grid(self, direction) -> np.ndarray:
        """
        Return the sun azimuth grid (degrees) as a 2-D array.
        """

        direction_norm = direction.strip().lower()

        return self._read_values_rows(self.metadata_paths[f"sun_{direction_norm}_values_rows"])

    # ------------------------------------------------------------------
    # Viewing incidence angles grids
    # ------------------------------------------------------------------

    def list_viewing_grids(self) -> list[tuple[int, int]]:
        """
        List available (bandId, detectorId) pairs for viewing incidence grids.
        """
        base = "./n1:Geometric_Info/Tile_Angles/Viewing_Incidence_Angles_Grids"
        elems = self.xml_root.findall(base, self.xml_ns)

        out: list[tuple[int, int]] = []
        for e in elems:
            out.append((int(e.attrib["bandId"]), int(e.attrib["detectorId"])))
        out.sort()
        return out

    def list_viewing_detectors(self, *, band_id: int) -> list[int]:
        """List detectorIds available for a given bandId."""
        return sorted({d for (b, d) in self.list_viewing_grids() if b == band_id})

    def find_viewing_angle_grid(self, *, band_id: int, direction: str) -> tuple[np.ndarray, list[int]]:
        """
        Return the viewing azimuth grid (degrees) for a given bandId and detectorId.
        """

        detector_ids = self.list_viewing_detectors(band_id=band_id)
        if not detector_ids:
            raise ValueError(f"No viewing incidence grids found for band_id={band_id}")

        ang_array = self._read_viewing_grid_stack(band_id=band_id, detector_ids=detector_ids, direction=direction)

        return ang_array, detector_ids

    def _read_viewing_grid_stack(self, *, band_id: int, detector_ids, direction: str) -> np.ndarray:
        """
        Read viewing incidence angle grids for a given band and stack across detectors.

        The returned array is ordered by detector ID defined by `detector_ids`

        :param band_id:
            Sentinel-2 ``bandId`` for which to read viewing incidence grids.
        :param direction:
            Grid direction name. Case-insensitive. Must be either ``"zenith"`` or
            ``"azimuth"`` (corresponding to the XML tags ``<Zenith>`` or ``<Azimuth>``).

        :returns:
            NumPy array of shape ``(ndetectors, nrows, ncols)`` containing the
            requested viewing angle grid for each detector.

        :raises ValueError:
            If no detectors are available for the requested band, or if ``direction``
            is not one of the supported values.
        """

        direction_norm = direction.strip().lower()
        if direction_norm == "zenith":
            direction_tag = "Zenith"
        elif direction_norm == "azimuth":
            direction_tag = "Azimuth"
        else:
            raise ValueError(f"direction must be 'zenith' or 'azimuth' (got {direction!r})")

        base = "./n1:Geometric_Info/Tile_Angles/"

        # First detector establishes grid shape
        xpath0 = (
            base
            + f"Viewing_Incidence_Angles_Grids[@bandId='{band_id}'][@detectorId='{detector_ids[0]}']/"
            + f"{direction_tag}/Values_List/VALUES"
        )
        grid0 = self._read_values_rows(xpath0)  # (nrows, ncols)

        out = np.empty(
            (len(detector_ids), grid0.shape[0], grid0.shape[1]),
            dtype=float,
        )
        out[0] = grid0

        for i, detector_id in enumerate(detector_ids[1:], start=1):
            xpath = (
                base
                + f"Viewing_Incidence_Angles_Grids[@bandId='{band_id}'][@detectorId='{detector_id}']/"
                + f"{direction_tag}/Values_List/VALUES"
            )
            out[i] = self._read_values_rows(xpath)

        return out

    def find_band_mean_viewing_grids(self, *, band_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mean_viewing_zenith, mean_viewing_azimuth) for a band averaged over detectors.
        """
        zen_stack, dets = self.find_viewing_angle_grid(band_id=band_id, direction="zenith")
        azi_stack, dets2 = self.find_viewing_angle_grid(band_id=band_id, direction="azimuth")

        # Defensive: make sure we averaged over the same detector set
        if dets != dets2:
            raise ValueError(f"Detector ids differ between zenith and azimuth for band {band_id}: {dets} vs {dets2}")

        # stacks are (ndetectors, nrows, ncols)
        return np.mean(zen_stack, axis=0), np.mean(azi_stack, axis=0)

    # ------------------------------------------------------------------
    # Internal grid parsing helper
    # ------------------------------------------------------------------

    def _read_values_rows(self, xpath: str) -> np.ndarray:
        """
        Parse a row-wise grid stored as multiple ``<VALUES>`` elements.
        """
        elems = self.xml_root.findall(xpath, self.xml_ns)
        rows: list[list[float]] = []
        for e in elems:
            text = (e.text or "").strip()
            if not text:
                continue
            toks = self._split_tokens(text)
            rows.append([float(t) for t in toks])
        return np.asarray(rows, dtype=float)


if __name__ == "__main__":
    pass
