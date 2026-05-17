"""eoio.readers.factory - reader factory class"""

import re


__author__ = [
    "Sam Hunt <sam.hunt@npl.co.uk>",
    "Mattea Goalen <mattea.goalen@npl.co.uk>",
    "Maddie Stedman <maddie.stedman@npl.co.uk>",
]
__all__ = ["ReaderFactory"]


class ReaderFactory:
    """
    Reader Factory class
    """

    def __init__(self):
        pass

    @staticmethod
    def get_reader(path: str):
        """
        Return reader for product at path

        :param path: product path
        :returns: Reader object.
        """

        L89_pattern = re.compile(r"LC?O?T?08?9?_.*._0[1-2]_..*")  # Landsat 8/9 OLI and TIRS
        S3OL_pattern = re.compile(r"S3.?_OL_1.*")  # S3-OLCI
        S3SLSTR_pattern = re.compile(r"S3.?_SL_1.*")  # S3-SLSTR
        S2MSIL1C_pattern = re.compile(r"S2.?_MSIL1C_.*")  # S2-MSI L1c
        S2MSIL2A_pattern = re.compile(r"S2.?_MSIL2A_.*")  # S2-MSI L2a
        hypernets_L1Irr_pattern = re.compile(r"HYPERNETS.*L1.*IRR.*")
        hypernets_L2_pattern = re.compile(r"HYPERNETS.*L2.*")
        radcalnet_TOA_pattern = re.compile(r".*.v*.*.output")
        radcalnet_BOA_pattern = re.compile(r".*.v*.*.input")
        planetscope_pattern = re.compile(r".*AnalyticMS.*8b.*.tif\Z")
        airbus_pleiades_pattern = re.compile(r".IMG_PHR*.*")
        EMITL1B_pattern = re.compile(r"EMIT.?_L1B_.*.nc\Z")
        ERA5_pattern = re.compile(r"ERA5*.*")
        CAMS_pattern = re.compile(r"CAMS*.*")
        FLOX_pattern = re.compile(r".*.FLOX*.*")
        nc_pattern = re.compile(r".*.nc")
        # > Check if input product_path matches any known product regular expressions
        #   return parsingFactory as appropriate
        if re.search(S2MSIL1C_pattern, path):
            from eoio.readers.sentinel2.reader import S2MSIReader

            return S2MSIReader

        elif re.search(S2MSIL2A_pattern, path):
            from eoio.readers.sentinel2.reader import S2MSIReader

            return S2MSIReader

        elif re.search(L89_pattern, path):
            from eoio.readers.landsat.reader import LandsatReader

            return LandsatReader

        elif re.search(S3OL_pattern, path):
            from eoio.readers.sentinel3_olci.reader import OLCIL1Reader

            return OLCIL1Reader

        elif re.search(S3SLSTR_pattern, path):
            from eoio.readers.sentinel3_slstr.reader import SLSTRL1Reader

            return SLSTRL1Reader

        elif re.search(hypernets_L1Irr_pattern, path):
            from eoio.readers.hypernets.reader import HYPERNETSL1IrrReader

            return HYPERNETSL1IrrReader

        elif re.search(hypernets_L2_pattern, path):
            from eoio.readers.hypernets.reader import HYPERNETSL2RefReader

            return HYPERNETSL2RefReader

        elif re.search(planetscope_pattern, path):
            from eoio.readers.planetscope.reader import PlanetScopeReader

            return PlanetScopeReader

        elif re.search(airbus_pleiades_pattern, path):
            from eoio.readers.airbus_pleiades.reader import AirbusPleiadesReader

            return AirbusPleiadesReader

        elif re.search(radcalnet_TOA_pattern, path):
            from eoio.readers.radcalnet.reader import RadCalNetReader

            return RadCalNetReader

        elif re.search(radcalnet_BOA_pattern, path):
            from eoio.readers.radcalnet.reader import RadCalNetInputReader

            return RadCalNetInputReader

        elif re.search(EMITL1B_pattern, path):
            from eoio.readers.emit.reader import EMITL1BReader

            return EMITL1BReader

        elif re.search(ERA5_pattern, path):
            from eoio.readers.era5.reader import ERA5Reader

            return ERA5Reader
        elif re.search(CAMS_pattern, path) or re.search(FLOX_pattern, path) or re.search(nc_pattern, path):
            from eoio.readers.generic_netcdf.reader import NetCDFReader

            return NetCDFReader
        else:
            raise ValueError(f"Provided file given in path does not match any known product formats. Path: {path}.")


if __name__ == "__main__":
    pass
