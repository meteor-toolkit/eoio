import unittest
from unittest import mock
import xarray as xr
from eoio.readers.hypernets.metadata import HYPERNETSMetadataExtractor
from eoio.readers.hypernets.reader import HYPERNETSReader


class TestMetadata(unittest.TestCase):
    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_get_product_metadata(self, mock_open_dataset, mock_exists):
        ds = xr.Dataset()
        ds.attrs["foo"] = "bar"
        mock_open_dataset.return_value = ds
        reader = HYPERNETSReader("dummy_path")
        md = HYPERNETSMetadataExtractor(reader, ds).get_product_metadata()
        assert md["foo"] == "bar"

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_get_variable_product_metadata(self, mock_open_dataset, mock_exists):
        ds = xr.Dataset({"var": (["wavelength"], [1])})
        ds["var"].attrs["meta"] = "info"
        mock_open_dataset.return_value = ds
        reader = HYPERNETSReader("dummy_path")
        md = HYPERNETSMetadataExtractor(reader, ds).get_variable_product_metadata("var")
        assert md["meta"] == "info"

    @mock.patch("eoio.readers.base.os.path.exists", return_value=True)
    @mock.patch("eoio.readers.generic_netcdf.reader.xr.open_dataset")
    def test_get_basic_metadata(self, mock_open_dataset, mock_exists):
        ds = xr.Dataset()
        # initialize acquistion times consisting of multiple timestamps to ensure correct handling of time dimension in metadata extraction
        ds["acquisition_time"] = xr.DataArray([1772106654, 1772106655], dims=["time"])
        ds.attrs["site_latitude"] = 1.0
        ds.attrs["site_longitude"] = 2.0
        ds.attrs["site_id"] = "site"
        ds.attrs["product_name"] = "prod"
        ds.attrs["product_level"] = "L1"
        ds.attrs["product_date"] = "2020-01-01"
        reader = HYPERNETSReader("dummy_path")
        basic_md = HYPERNETSMetadataExtractor(reader, ds).get_basic_metadata()
        assert basic_md["collection_name"] == "site"
        assert basic_md["product_name"] == "prod"
        assert basic_md["platform"] == "HYPERNETS"


if __name__ == "__main__":
    unittest.main()
