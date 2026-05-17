"""eoio.readers.planetscope.layout - helper for PlanetScope product layout parsing."""

from dataclasses import dataclass
from pathlib import Path
import glob
import os
from typing import List, Optional


class PlanetScopeLayoutError(ValueError):
    """
    Raised when a PlanetScope layout is missing expected files/structure.
    """

    pass


@dataclass(frozen=True)
class PlanetScopeLayout:
    """
    Layout helper for PlanetScope products.
    Solely responsible for filesystem/path logic. No heavy dependencies.

    Note:
    It is assumed (in conjunction with the reader-factory and processor-factory),
    that the path handed to the eoio reader is the image (tif) file path, incl the file name
    and that there might be more then one PlanetScope dataset in the same folder.
    """

    _default_img_str = "*AnalyticMS*"
    _default_mask_str = "*udm2*"
    _default_metadata_str1 = _default_img_str + ".xml"
    _default_metadata_str2 = "*.json"

    # main image tif path and filename handed over to layout class
    image_file: str

    def __post_init__(self) -> None:
        """
        Validate image path.
        """
        p = Path(self.image_file)
        if not p.exists():
            raise PlanetScopeLayoutError(f"The given path was not found: {self.image_file}")
        if not p.is_file():
            raise PlanetScopeLayoutError(f"The given path is not a file: {self.image_file}")

    def mask_file(self) -> str:
        """
        Return list of available mask file paths based on image file name and path (None if none found).
        """
        # extract path and first 4 groups of image file name (date_time_sat-id_product-level = id)
        dir_path, id = os.path.split(self.image_file)
        id = "_".join(id.split("_")[:4])

        # search for mask file in same folder starting with id and containing default mask string
        m_file = glob.glob(os.path.join(dir_path, id + self._default_mask_str))

        # if files are found store them, otherwise store None
        if len(m_file) == 1:
            mask_file = m_file[0]
        elif len(m_file) > 1:
            raise PlanetScopeLayoutError(f"More than one mask file found: {m_file}")
        else:
            raise PlanetScopeLayoutError(f"No corresponding mask file found for file: {self.image_file}")

        return mask_file

    def metadata_files(self) -> Optional[List[str]]:
        """
        Return list of available metadata file paths based on image file name and path (None if none found).
        """

        # extract path and first 4 groups of image file name (date_time_sat-id_product-level = id)
        dir_path, id = os.path.split(self.image_file)
        id = "_".join(id.split("_")[:4])

        # search for metadata files in same folder starting with id and containing default metadata strings
        m_files = glob.glob(os.path.join(dir_path, id + self._default_metadata_str1)) + glob.glob(
            os.path.join(dir_path, id + self._default_metadata_str2)
        )

        # if files are found store them, otherwise store None
        if len(m_files) > 0:
            return m_files
        else:
            return None
