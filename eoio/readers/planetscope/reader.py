"""eoio.readers.planetscope.reader - PlanetScope data reader implementation."""

from typing import Any, Dict, Optional
from pathlib import Path
import xarray as xr

from eoio.readers.base import BaseRasterReader
from eoio.readers.planetscope.layout import PlanetScopeLayout
from eoio.readers.planetscope.data_io import read_tif_into_dataset
from eoio.readers.planetscope.aux_data import read_aux
from eoio.readers.planetscope.metadata import PlanetScopeMetadataExtractor
from eoio.readers.planetscope.conventions import apply_conventions
from eoio.readers.subset.roi_subset import ROISubsetResolver, ResolvedROISubset
from eoio.deps import lazy_rasterio

from processor_tools.utils.dict_tools import get_value

# AUX_OPTIONS - Available auxiliary variable names
AUX_OPTIONS = []  # TODO - add

# TODO - add observation geometry angles

# MASK_OPTIONS - Available mask variable names
MASK_OPTIONS = []  # TODO - add


class PlanetScopeReaderError(ValueError):
    """
    Raised when a PlanetScope reader is coming across an input it can not handle.
    """

    pass


class PlanetScopeReader(BaseRasterReader):
    """
    Base File Reader for PlanetScope SuperDove data, containing 8 bands ("8b" in image tif filename).
    """

    default_read_params: Dict[str, Any] = {
        "save_extracted": False,
        "metadata_level": "all",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": False,
        "use_chunks": False,
        "chunks": None,
    }
    all_read_params = {
        "save_extracted": "True/False",
        "metadata_level": "all/basic/None",  # None | False disables metadata; True/'basic'/'full' etc enables
        "include_uncertainties": "True/False",
        "use_chunks": "True/False",
        "chunks": "None/{'x':X,'y':Y}",
    }

    meas_var_res: Dict[str, Any] = {
        "B1": 3,
        "B2": 3,
        "B3": 3,
        "B4": 3,
        "B5": 3,
        "B6": 3,
        "B7": 3,
        "B8": 3,
    }

    def __init__(
        self,
        path: Path | str,
        vars_sel: Optional[Dict[str, Any]] = None,
        subset: Optional[Dict[str, Any]] = None,
        read_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize PlanetScope reader.
        """

        self.path = Path(path)

        # Initialize the PlanetScope layout - i.e. the file structure of the PlanetScope product
        self.layout = PlanetScopeLayout(str(self.path))

        self.meas_def = {
            "none": [],
            "all": list(self.meas_var_res.keys()),
            "rgb": ["B6", "B4", "B2"],
        }
        self.aux_def = {
            "all": AUX_OPTIONS,
        }
        self.mask_def = {"all": MASK_OPTIONS}

        # initalise the metadata extractor
        self.mtd = PlanetScopeMetadataExtractor(self)

        # meas var res
        self.res = 3

        # run BaseRasterReader initializer
        super().__init__(path, vars_sel, subset, read_params)

    def resolve_subset(
        self,
        subset: Optional[Dict[str, Any]],
    ) -> Optional[ResolvedROISubset]:
        """
        Resolve the subset of the PlanetScope product.

        :param subset: Subset of the PlanetScope product.
        :return: Resolved ROI Subset.
        """

        if not subset:
            return None

        roi = subset.get("roi")
        if roi is None:
            return None

        roi_crs = subset.get("roi_crs")

        if not self.layout.metadata_files():
            # lazy import
            rio = lazy_rasterio()

            with rio.open(self.path) as src:
                image_crs = "EPSG:" + str(src.crs)
                pass
        else:
            image_crs = get_value(self.mtd.product_metadata, "geospatial_bounds_crs")

        if image_crs is None:
            raise ValueError("Cannot resolve ROI subset: product CRS not available.")

        # TODO add image_bounds to enable checking if ROI is within image bounds

        return ROISubsetResolver(
            roi=roi,
            roi_crs_epsg=roi_crs,
            image_crs_epsg=image_crs,
            image_bounds=None,
        ).run()

    def open_dataset(self) -> xr.Dataset:
        """
        Read the product and return an xarray.Dataset.
        """

        # unpack config
        meas = self.resolved_config.vars_sel.get("meas", None)
        aux = self.resolved_config.vars_sel.get("aux", None)
        roi_subset = self.resolved_config.subset
        rp = self.resolved_config.read_params
        mtd_level = rp.get("metadata_level", None)

        # initialize an empty xarray Dataset to populate
        ds = xr.Dataset()

        # Read image data if requested
        if meas:
            ds = read_tif_into_dataset(
                ds=ds,
                layout=self.layout,
                meas=meas,
                subset=roi_subset,
                mtd=self.mtd,
                use_chunks=rp.get("use_chunks", False),
                chunks=rp.get("chunks", None),
            )

        if mtd_level is True:
            mtd_level = "all"
        if isinstance(mtd_level, str) and mtd_level.lower() in ("all", "basic"):
            ds = self.mtd.attach_metadata(ds, level=mtd_level.lower())

        # add aux data if requested
        if aux:
            ds = read_aux(ds=ds, aux=aux, mtd=self.mtd)

        ds = apply_conventions(ds, layout=self.layout, roi_subset=roi_subset, config=self.resolved_config)

        return ds

    @staticmethod
    def get_extension() -> str:
        """
        Return the extension of the extracted PlanetScope product.

        Must be ".tif" as multiple Planetscope products are usually stored in the same directory, each with their corresponding .xml and .json files.
        """
        return ".tif"


if __name__ == "__main__":
    pass
