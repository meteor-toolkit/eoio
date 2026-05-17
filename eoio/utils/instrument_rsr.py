# """eoio.utils.wavelengths - class to add relative spectral response information to datasets"""

# from copy import deepcopy
# from typing import Union

# import xarray as xr
# from pyspectral.rsr_reader import RelativeSpectralResponse

# __author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

# __all__ = ["InstrumentRSR"]


# class InstrumentRSR:
#     instrument_dict = {
#         "Sentinel-3A": ["OLCI", "SLSTR"],
#         "Sentinel-3B": ["OLCI", "SLSTR"],
#         "Sentinel-2A": ["MSI"],
#         "Sentinel-2B": ["MSI"],
#         "Landsat-8": ["OLI", "TIRS"],
#         "Landsat-9": ["OLI", "TIRS"],
#     }

#     def __init__(self, ds: xr.Dataset):
#         """
#         Initialise Instrument Relative Spectral Response class
#         """
#         self.platform = (
#             "Landsat-8" if ds.attrs["platform"] == "Landsat-9" else ds.attrs["platform"]
#         )
#         self.instrument = self.instrument_dict[ds.attrs["platform"]]
#         self.ds = ds

#     def add_info(self, options: Union[str, list]):
#         """
#         Add relative spectral response information to the attributes of a spectral band in a xr.Dataset

#         Options available::
#             "response", "wavelengths", "central_wavelength"

#         :param options: information to add to the xr.Dataset
#         """

#         if isinstance(options, list):
#             for i in options:
#                 self.add_info(i)
#         else:
#             for i in self.instrument:
#                 instr = RelativeSpectralResponse(self.platform, i)

#                 for v in self.ds:
#                     if v in instr.band_names:
#                         if "product_metadata" not in self.ds[v].attrs.keys():
#                             self.ds[v].attrs.update({"product_metadata": {}})
#                         if options != "central_wavelength":
#                             self.ds[v].attrs["product_metadata"].update(
#                                 {
#                                     "rsr_units": instr.unit,
#                                     options: list(instr.rsr[v]["det-1"][options]),
#                                 }
#                             )
#                         elif (
#                             "central_wavelength" not in self.ds[v].attrs.keys()
#                             and "band_central_wavelength" not in self.ds[v].attrs.keys()
#                         ):
#                             self.ds[v].attrs["product_metadata"].update(
#                                 {
#                                     "rsr_units": instr.unit,
#                                     options: instr.rsr[v]["det-1"][options],
#                                 }
#                             )
#                         if (
#                             "central_wavelength"
#                             in deepcopy(self.ds[v].attrs["product_metadata"]).keys()
#                         ):
#                             self.ds[v].attrs["band_central_wavelength"] = (
#                                 self.ds[v]
#                                 .attrs["product_metadata"]
#                                 .pop("central_wavelength")
#                             )
#                         if (
#                             "wavelength"
#                             in deepcopy(self.ds[v].attrs["product_metadata"]).keys()
#                         ):
#                             self.ds[v].attrs["product_metadata"].update(
#                                 {
#                                     "response_wavelengths": self.ds[v]
#                                     .attrs["product_metadata"]
#                                     .pop("wavelength")
#                                 }
#                             )


# if __name__ == "__main__":
#     pass
