"""eoio.readers.landsat.aux_data.aux_data - Aux data handler for Landsat."""

from __future__ import annotations
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from eoio.readers.landsat.reader import LandsatReader

from eoio.readers.landsat.aux_data.angles import read_angles_into_dataset
from eoio.utils.aux_read import warn_on_aux_failure


class LSAuxData:
    """
    Manager for retrieving and attaching auxiliary data to Landsat datasets.

    Only aux data for Level-1 products are angles.
    For Level-2 products, additional aux data will be added in future.
    """

    def __init__(self, reader: LandsatReader):
        """
        Initialise with the parent reader.

        :param reader: Active LandsatReader instance.
        """
        self.reader = reader
        self.layout = reader.layout

    def attach_aux(self, ds: xr.Dataset, aux: list[str]) -> xr.Dataset:
        """
        Attach requested auxiliary data to the dataset.

        :param ds: The dataset containing measurement variables.
        :return: Dataset with auxiliary variables added.
        """
        requested_aux = aux
        if not requested_aux:
            return ds

        roi_subset = self.reader.resolved_config.subset

        # Add angle variables first
        angle_vars = [var for var in requested_aux if var in self.layout.get_angle_files()]
        if angle_vars:
            with warn_on_aux_failure("Landsat angles"):
                ds = read_angles_into_dataset(
                    ds=ds,
                    layout=self.layout,
                    angle_vars=angle_vars,
                    subset=roi_subset,
                    mtd=self.reader.mtd,
                    use_chunks=self.reader.resolved_config.read_params["use_chunks"],
                    chunks=self.reader.resolved_config.read_params["chunks"],
                )

        # # Read remaining aux vars (e.g. QA bands) from files - only exist in L2 products
        # aux_vars = [
        #     var for var in requested_aux if var not in angle_vars
        # ]
        # if aux_vars:
        #     raise NotImplementedError("File-based aux data reading not yet implemented.")

        return ds


if __name__ == "__main__":
    pass
