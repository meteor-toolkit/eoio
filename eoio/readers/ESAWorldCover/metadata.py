from eoio.readers.metadata import BaseMetadataExtractor
from eoio.readers.footprint_utils import normalize_footprint

FLAG_VALUES = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    95,
    100,
]

FLAG_MEANINGS = [
    "tree_cover",
    "shrubland",
    "grassland",
    "cropland",
    "built_up",
    "bare_sparse_vegetation",
    "snow_and_ice",
    "permanent_water_bodies",
    "herbaceous_wetland",
    "mangroves",
    "moss_and_lichen",
]

FLAG_COLORS = [
    "#006400",
    "#ffbb22",
    "#ffff4c",
    "#f096ff",
    "#fa0000",
    "#b4b4b4",
    "#f0f0f0",
    "#0064c8",
    "#0096a0",
    "#00cf75",
    "#fae6a0",
]


class ESAWorldCoverMetadataExtractor(BaseMetadataExtractor):
    def __init__(self, reader, ds):
        super().__init__(reader)
        self.ds = ds.copy()
        self.subset = reader.config.subset

    def get_basic_metadata(self) -> dict:
        try:
            bounds = self.ds.rio.bounds()
        except Exception:
            bounds = ""
        try:
            crs = self.ds.rio.crs.to_epsg()
        except Exception:
            crs = ""

        return {
            "collection_name": "ESA WorldCover",
            "product_name": str(self.path.stem),
            "platform": "Sentinel-1, Sentinel-2",
            "processing_level": "L3",
            "spatial_resolution": 10,
            "geometry_ids": "10m",
            "product_bounds": bounds,
            "product_date": self._extract_year(),
            "description": (
                "ESA WorldCover global land cover map at 10 m spatial resolution "
                "derived from Sentinel-1 GRD and Sentinel-2 L2A observations."
            ),
            "eoio:reader": "ESAWorldCover",
            "eoio:subset": repr(self.subset),
            "history": "",  # include if there is any history previous to eoio reading
            "footprint": normalize_footprint(bounds, f'EPSG:{crs}'),
        }

    def get_product_metadata(self) -> dict:
        md = self.get_basic_metadata()
        # md['labels']= LABELS
        # md["labels_colours"] = landcover_colors
        return md

    def get_variable_basic_metadata(self, var):
        return {
            "units": "1",
            "long_name": "Land cover classification",
            "standard_name": "",
            "flag_values": FLAG_VALUES,
            "flag_meanings":FLAG_MEANINGS,
            "flag_colors": FLAG_COLORS

        }

    def _extract_year(self):
        parts = self.path.stem.split("_")
        for p in parts:
            if p.isdigit() and len(p) == 4:
                return p
        return ""
