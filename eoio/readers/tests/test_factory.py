"""eoio.readers.tests.test_factory - tests for eoio.readers.factory"""

import unittest

from eoio.readers.sentinel2.reader import S2MSIReader

# from eoio.readers.sentinel2.reader import S2MSIL2AReader
from eoio.readers.factory import ReaderFactory
from eoio.readers.landsat.reader import LandsatReader
from eoio.readers.sentinel3_olci.reader import OLCIL1Reader

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


class TestReaderFactory(unittest.TestCase):
    def test_get_reader_None(self):
        path = "Invalid_product_path"
        reader = ReaderFactory()
        with self.assertRaises(ValueError):
            reader.get_reader(path)

    def test_get_reader_L89(self):
        path = [
            "LC08_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_01_TX",
            "LO08_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_01_TX",
            "LT08_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_01_TX",
            "LC09_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_02_TX",
            "LO09_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_02_TX",
            "LT09_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_02_TX",
        ]

        reader = ReaderFactory()
        for p in path:
            self.assertEqual(reader.get_reader(p), LandsatReader)

        for p in path:
            self.assertEqual(reader.get_reader(p + ".tar"), LandsatReader)

    def test_get_reader_OLIL1(self):
        path = "S3M_OL_1_TTTTTT_yyyymmddThhmmss_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_[instance ID]_GGG_[class ID].SEN3"
        reader = ReaderFactory()
        self.assertEqual(reader.get_reader(path), OLCIL1Reader)

        path = "S3M_OL_1_TTTTTT_yyyymmddThhmmss_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_[instance ID]_GGG_[class ID].zip"
        reader = ReaderFactory()
        self.assertEqual(reader.get_reader(path), OLCIL1Reader)

        path = "S3M_OL_1_TTTTTT_yyyymmddThhmmss_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_[instance ID]_GGG_[class ID]"
        reader = ReaderFactory()
        self.assertEqual(reader.get_reader(path), OLCIL1Reader)

    def test_get_reader_S2L1C(self):
        path = "S2_MSIL1C_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_YYYYMMDDHHMMSS.SAFE"
        reader = ReaderFactory()
        self.assertEqual(reader.get_reader(path), S2MSIReader)

        path = "S2_MSIL1C_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_YYYYMMDDHHMMSS.zip"
        reader = ReaderFactory()
        self.assertEqual(reader.get_reader(path), S2MSIReader)

        path = "S2_MSIL1C_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_YYYYMMDDHHMMSS"
        reader = ReaderFactory()
        self.assertEqual(reader.get_reader(path), S2MSIReader)

    # def test_get_reader_S2L2A(self):
    #     path = "S2_MSIL2A_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_YYYYMMDDHHMMSS.SAFE"
    #     reader = ReaderFactory()
    #     self.assertEqual(reader.get_reader(path), S2MSIL2AReader)

    #     path = "S2_MSIL2A_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_YYYYMMDDHHMMSS.zip"
    #     reader = ReaderFactory()
    #     self.assertEqual(reader.get_reader(path), S2MSIL2AReader)

    #     path = "S2_MSIL2A_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_YYYYMMDDHHMMSS"
    #     reader = ReaderFactory()
    #     self.assertEqual(reader.get_reader(path), S2MSIL2AReader)


if __name__ == "__main__":
    unittest.main()
