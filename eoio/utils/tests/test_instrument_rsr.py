# """eoio.utils.tests.test_instrument_rsr - tests for eoio.utils.instrument_rsr"""

# import unittest
# import unittest.mock as mock

# import numpy as np
# import xarray as xr

# from eoio.utils.instrument_rsr import InstrumentRSR


# class TestInstrumentRSR(unittest.TestCase):
#     def test_init(self):
#         input_ds = xr.Dataset(attrs={"platform": "Sentinel-3A"})
#         instr = InstrumentRSR(input_ds)
#         xr.testing.assert_identical(instr.ds, input_ds)
#         self.assertEqual(instr.platform, "Sentinel-3A")
#         self.assertEqual(instr.instrument, ["OLCI", "SLSTR"])

#     @mock.patch("eoio.utils.instrument_rsr.RelativeSpectralResponse")
#     def test_add_info(self, mock_RSR):
#         mock_RSR().band_names = ["Oa01", "Oa02", "Oa03"]
#         mock_RSR().unit = "1e-6 m"
#         mock_RSR().rsr = {
#             "Oa01": {
#                 "det-1": {
#                     "response": np.array([1, 2, 3]),
#                     "central_wavelength": 20,
#                     "wavelength": np.array([2, 3, 4]),
#                 }
#             }
#         }
#         input_ds = xr.Dataset(
#             {"Oa01": xr.DataArray()}, attrs={"platform": "Sentinel-3A"}
#         )
#         instr = InstrumentRSR(input_ds)
#         instr.add_info(["response", "wavelength", "central_wavelength"])
#         self.assertEqual(
#             instr.ds.Oa01.attrs,
#             {
#                 "band_central_wavelength": 20,
#                 "product_metadata": {
#                     "response_wavelengths": [2, 3, 4],
#                     "response": [1, 2, 3],
#                     "rsr_units": "1e-6 m",
#                 },
#             },
#         )


# if __name__ == "__main__":
#     unittest.main()
