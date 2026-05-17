"""eoio.utils.tests.test_read_utils - tests for eoio.utils.read_utils"""

import io
import unittest
from unittest.mock import MagicMock, patch

import xarray as xr

from eoio.utils.read_utils import (
    clean_and_convert_attrs,
    convert_to_netcdf_compatible,
    extract_file,
    extract_tarred_file,
    extract_zipped_file,
    revert_and_convert_attrs,
    revert_netcdf_compatible,
    setup_file,
)


def mock_interface_function(path, read_params=None, *args, **kwargs):
    with setup_file(path, read_params) as path:
        return path, read_params, True


class TestReadUtils(unittest.TestCase):
    @patch("eoio.utils.read_utils.os.remove")
    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.extract_file")
    def test_setup_file_saveTrue_notfileexists(self, mock_extract_file, mock_path_exists, mock_remove):
        mock_extract_file.return_value = "path_string", {"save_extracted": True}, True
        mock_path_exists.return_value = False

        test_path, test_read_params, test_extracted = mock_interface_function(
            "path_string", read_params={"save_extracted": True}
        )

        mock_extract_file.assert_called_once_with("path_string", {"save_extracted": True})
        self.assertDictEqual(test_read_params, {"save_extracted": True})
        self.assertEqual(test_path, "path_string")
        self.assertTrue(test_extracted)
        mock_remove.assert_not_called()

    @patch("eoio.utils.read_utils.os.remove")
    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.extract_file")
    def test_setup_file_saveTrue_fileexists(self, mock_extract_file, mock_path_exists, mock_remove):
        mock_extract_file.return_value = "path_string", {"save_extracted": True}, True
        mock_path_exists.return_value = True

        test_path, test_read_params, test_extracted = mock_interface_function(
            "path_string", read_params={"save_extracted": True}
        )

        mock_extract_file.assert_called_once_with("path_string", {"save_extracted": True})
        self.assertDictEqual(test_read_params, {"save_extracted": True})
        self.assertEqual(test_path, "path_string")
        self.assertTrue(test_extracted)
        mock_remove.assert_not_called()

    @patch("eoio.utils.read_utils.os.remove")
    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.extract_file")
    def test_setup_file_saveFalse_fileexists(self, mock_extract_file, mock_path_exists, mock_remove):
        mock_extract_file.return_value = "path_string", None, True
        mock_path_exists.return_value = True

        test_path, test_read_params, test_extracted = mock_interface_function("path_string")

        mock_extract_file.assert_called_once_with("path_string", None)
        self.assertEqual(test_path, "path_string")
        self.assertIsNone(test_read_params)
        self.assertTrue(test_extracted)
        mock_remove.assert_called_once_with("path_string")

    @patch("eoio.utils.read_utils.os.remove")
    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.extract_file")
    def test_setup_file_saveFalse_notfileexists(self, mock_extract_file, mock_path_exists, mock_remove):
        mock_extract_file.return_value = "path_string", None, True
        mock_path_exists.return_value = False

        test_path, test_read_params, test_extracted = mock_interface_function("path_string")

        mock_extract_file.assert_called_once_with("path_string", None)
        self.assertEqual(test_path, "path_string")
        self.assertIsNone(test_read_params)
        self.assertTrue(test_extracted)
        mock_remove.assert_not_called()

    @patch("eoio.readers.sentinel2.reader.S2MSIReader.get_extension")
    @patch("eoio.utils.read_utils.extract_tarred_file")
    @patch("eoio.utils.read_utils.extract_zipped_file")
    @patch("eoio.utils.read_utils.zipfile.is_zipfile")
    @patch("eoio.utils.read_utils.tarfile.is_tarfile")
    @patch("eoio.utils.read_utils.os.path.exists")
    def test_extract_file_s2(
        self,
        mock_path_exists,
        mock_is_tarfile,
        mock_is_zipfile,
        mock_extract_zipped_file,
        mock_extract_tarred_file,
        mock_get_extension,
    ):
        mock_path_exists.return_value = False
        mock_is_tarfile.return_value = False
        mock_is_zipfile.return_value = True
        mock_get_extension.return_value = ".SAFE"

        fn = "S2A_MSIL1C_20220101T034141_N0301_R061_T49TCF_20220101T053033"
        test_path, test_read_params, test_extracted = extract_file(fn)
        self.assertEqual(fn + ".SAFE", test_path)
        self.assertIsNone(test_read_params)
        self.assertTrue(test_extracted)
        mock_extract_zipped_file.assert_called_once_with(
            fn, "S2A_MSIL1C_20220101T034141_N0301_R061_T49TCF_20220101T053033.SAFE"
        )
        mock_extract_tarred_file.assert_not_called()

    @patch("eoio.readers.sentinel3_olci.reader.OLCIL1Reader.get_extension")
    @patch("eoio.utils.read_utils.extract_tarred_file")
    @patch("eoio.utils.read_utils.extract_zipped_file")
    @patch("eoio.utils.read_utils.zipfile.is_zipfile")
    @patch("eoio.utils.read_utils.tarfile.is_tarfile")
    @patch("eoio.utils.read_utils.os.path.exists")
    def test_extract_file_s3(
        self,
        mock_path_exists,
        mock_is_tarfile,
        mock_is_zipfile,
        mock_extract_zipped_file,
        mock_extract_tarred_file,
        mock_get_extension,
    ):
        mock_path_exists.return_value = False
        mock_is_tarfile.return_value = False
        mock_is_zipfile.return_value = True
        mock_get_extension.return_value = ".SEN3"

        test_path, test_read_params, test_extracted = extract_file(
            "S3A_OL_1_EFR____20100602T094537_20100602T094837_20150601T075458_0180_090_022______LN2_D_NT_001"
        )
        self.assertEqual(
            "S3A_OL_1_EFR____20100602T094537_20100602T094837_20150601T075458_0180_090_022______LN2_D_NT_001.SEN3",
            test_path,
        )
        mock_extract_zipped_file.assert_called_once_with(
            "S3A_OL_1_EFR____20100602T094537_20100602T094837_20150601T075458_0180_090_022______LN2_D_NT_001",
            "S3A_OL_1_EFR____20100602T094537_20100602T094837_20150601T075458_0180_090_022______LN2_D_NT_001.SEN3",
        )
        mock_extract_tarred_file.assert_not_called()
        self.assertIsNone(test_read_params)
        self.assertTrue(test_extracted)

    @patch("eoio.readers.factory.ReaderFactory.get_reader")
    @patch("eoio.utils.read_utils.extract_tarred_file")
    @patch("eoio.utils.read_utils.extract_zipped_file")
    @patch("eoio.utils.read_utils.zipfile.is_zipfile")
    @patch("eoio.utils.read_utils.tarfile.is_tarfile")
    @patch("eoio.utils.read_utils.os.path.exists")
    def test_extract_file_zip(
        self,
        mock_path_exists,
        mock_is_tarfile,
        mock_is_zipfile,
        mock_extract_zipped_file,
        mock_extract_tarred_file,
        mock_get_reader,
    ):
        mock_path_exists.return_value = False
        mock_is_tarfile.return_value = False
        mock_is_zipfile.return_value = True
        mock_get_reader.return_value.get_extension.return_value = ""

        test_path, test_read_params, test_extracted = extract_file("path_string")
        self.assertIsNone(test_read_params)
        self.assertEqual("path_string", test_path)
        self.assertTrue(test_extracted)
        mock_extract_zipped_file.assert_called_once_with("path_string", "path_string")
        mock_extract_tarred_file.assert_not_called()

    @patch("eoio.readers.factory.ReaderFactory.get_reader")
    @patch("eoio.utils.read_utils.extract_tarred_file")
    @patch("eoio.utils.read_utils.extract_zipped_file")
    @patch("eoio.utils.read_utils.zipfile.is_zipfile")
    @patch("eoio.utils.read_utils.tarfile.is_tarfile")
    @patch("eoio.utils.read_utils.os.path.exists")
    def test_extract_file_tar(
        self,
        mock_path_exists,
        mock_is_tarfile,
        mock_is_zipfile,
        mock_extract_zipped_file,
        mock_extract_tarred_file,
        mock_get_reader,
    ):
        mock_path_exists.return_value = False
        mock_is_tarfile.return_value = True
        mock_is_zipfile.return_value = False
        mock_get_reader.return_value.get_extension.return_value = ""

        test_path, test_read_params, test_extracted = extract_file("path_string")
        self.assertIsNone(test_read_params)
        self.assertEqual("path_string", test_path)
        self.assertTrue(test_extracted)
        mock_extract_tarred_file.assert_called_once_with("path_string", "path_string")
        mock_extract_zipped_file.assert_not_called()

    @patch("eoio.readers.factory.ReaderFactory.get_reader")
    @patch("eoio.utils.read_utils.extract_tarred_file")
    @patch("eoio.utils.read_utils.extract_zipped_file")
    @patch("eoio.utils.read_utils.zipfile.is_zipfile")
    @patch("eoio.utils.read_utils.tarfile.is_tarfile")
    @patch("eoio.utils.read_utils.os.path.exists")
    def test_extract_file_exists(
        self,
        mock_path_exists,
        mock_is_tarfile,
        mock_is_zipfile,
        mock_extract_zipped_file,
        mock_extract_tarred_file,
        mock_get_reader,
    ):
        mock_path_exists.return_value = True
        mock_is_tarfile.return_value = True
        mock_is_zipfile.return_value = True
        mock_get_reader.return_value.get_extension.return_value = ""

        test_path, test_read_params, test_extracted = extract_file("path_string")
        self.assertEqual("path_string", test_path)
        self.assertDictEqual(test_read_params, {"save_extracted": True})
        self.assertFalse(test_extracted)
        mock_extract_tarred_file.assert_not_called()
        mock_extract_zipped_file.assert_not_called()

    @patch("eoio.readers.factory.ReaderFactory.get_reader")
    @patch("eoio.utils.read_utils.extract_tarred_file")
    @patch("eoio.utils.read_utils.extract_zipped_file")
    @patch("eoio.utils.read_utils.zipfile.is_zipfile")
    @patch("eoio.utils.read_utils.tarfile.is_tarfile")
    @patch("eoio.utils.read_utils.os.path.exists")
    def test_extract_file_Error(
        self,
        mock_path_exists,
        mock_is_tarfile,
        mock_is_zipfile,
        mock_extract_zipped_file,
        mock_extract_tarred_file,
        mock_get_reader,
    ):
        mock_path_exists.return_value = False
        mock_is_tarfile.return_value = False
        mock_is_zipfile.return_value = False
        mock_get_reader.return_value.get_extension.return_value = ""

        with self.assertWarns(Warning):
            test_path, test_read_params, test_extracted = extract_file("path_string")

        self.assertEqual("path_string", test_path)
        self.assertIsNone(test_read_params)
        self.assertFalse(test_extracted)
        mock_extract_tarred_file.assert_not_called()
        mock_extract_zipped_file.assert_not_called()

    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.zipfile.ZipFile")
    def test_extract_zipped_file_ValidZip(self, mock_zipfile, mock_exists):
        # Mock path_extracted does not exist
        mock_exists.return_value = False

        # Mock zipfile behavior
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.infolist.return_value = ["file1.txt", "file2.txt"]

        # Run the function
        extract_zipped_file("fake_path.zip", "fake_extracted_path")

        # Assertions
        mock_zipfile.assert_called_once_with("fake_path.zip", "r")
        mock_zip_instance.infolist.assert_called_once()
        self.assertEqual(2, mock_zip_instance.extract.call_count)
        mock_zip_instance.close.assert_called_once()

    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.zipfile.ZipFile")
    def test_extract_zipped_file_ExtractedPathExists(self, mock_zipfile, mock_exists):
        # Mock path_extracted exists
        mock_exists.return_value = True

        extract_zipped_file("fake_path.zip", "fake_extracted_path")

        # Ensure zipfile.ZipFile is never opened
        mock_zipfile.assert_not_called()

    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.zipfile.ZipFile")
    def test_extract_zipped_file_PathExtractedNone(self, mock_zipfile, mock_exists):
        # Run the function with path_extracted=None
        extract_zipped_file("fake_path.zip", None)

        # Ensure zipfile.ZipFile is never opened
        mock_zipfile.assert_not_called()
        mock_exists.assert_not_called()

    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.zipfile.ZipFile")
    def test_extract_zipped_file_EmptyZip(self, mock_zipfile, mock_exists):
        # Mock path_extracted does not exist
        mock_exists.return_value = False

        # Mock zipfile behavior
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.infolist.return_value = []

        extract_zipped_file("fake_path.zip", "fake_extracted_path")

        # Assertions
        mock_zipfile.assert_called_once_with("fake_path.zip", "r")
        mock_zip_instance.infolist.assert_called_once()
        mock_zip_instance.extract.assert_not_called()
        mock_zip_instance.close.assert_called_once()

    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.os.makedirs")
    @patch("eoio.utils.read_utils.tarfile.open")
    def test_extract_tarred_file_with_path_extracted(self, mock_tarfile_open, mock_makedirs, mock_exists):
        # Mock os.path.exists to always return False
        mock_exists.return_value = False

        mock_tar = MagicMock()
        mock_tarfile_open.return_value = mock_tar

        path = "fake_path.tar"
        path_extracted = "fake_extracted_dir/"

        extract_tarred_file(path, path_extracted)

        mock_exists.assert_called_once_with(path_extracted)
        mock_makedirs.assert_called_once_with(path_extracted, exist_ok=True)
        mock_tarfile_open.assert_called_once_with(path)
        mock_tar.extractall.assert_called_once_with(path_extracted)
        mock_tar.close.assert_called_once()

    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.os.makedirs")
    @patch("eoio.utils.read_utils.tarfile.open")
    def test_extract_tarred_file_PathExtractedExists(self, mock_tarfile_open, mock_makedirs, mock_exists):
        # Mock os.path.exists to return True (path already exists)
        mock_exists.return_value = True

        path = "fake_path.tar"
        path_extracted = "fake_extracted_dir/"

        extract_tarred_file(path, path_extracted)

        mock_exists.assert_called_once_with(path_extracted)
        mock_makedirs.assert_not_called()
        mock_tarfile_open.assert_not_called()

    @patch("eoio.utils.read_utils.os.path.exists")
    @patch("eoio.utils.read_utils.os.makedirs")
    @patch("eoio.utils.read_utils.tarfile.open")
    def test_extract_tarred_file_PathExtractedNone(self, mock_tarfile_open, mock_makedirs, mock_exists):
        # Test paths
        path = "fake_path.tar"

        extract_tarred_file(path)

        mock_exists.assert_not_called()
        mock_makedirs.assert_not_called()
        mock_tarfile_open.assert_not_called()


class TestCleanAndConvertAttrs(unittest.TestCase):
    def test_convert_to_netcdf_compatible_simple_types(self):
        self.assertEqual(convert_to_netcdf_compatible("test string"), "test string")
        self.assertEqual(convert_to_netcdf_compatible(42), 42)
        self.assertEqual(convert_to_netcdf_compatible(3.14), 3.14)
        self.assertEqual(convert_to_netcdf_compatible(True), True)

    def test_convert_complex_number(self):
        self.assertEqual(convert_to_netcdf_compatible(3 + 4j), "3.0+4.0j")

    def test_convert_bytes(self):
        self.assertEqual(convert_to_netcdf_compatible(b"hello"), "hello")
        self.assertEqual(convert_to_netcdf_compatible(bytearray(b"world")), "world")

    def test_convert_nested_dict(self):
        nested_dict = {"key": {"subkey": "value"}}
        self.assertEqual(convert_to_netcdf_compatible(nested_dict), '{"key": {"subkey": "value"}}')

    def test_convert_set(self):
        self.assertEqual(convert_to_netcdf_compatible({1, 2, 3}), [1, 2, 3])

    def test_convert_generator(self):
        gen = (x for x in range(3))
        self.assertEqual(convert_to_netcdf_compatible(gen), [0, 1, 2])

    def test_clean_and_convert_attrs(self):
        # Create a sample dataset with global attributes and variable attributes
        ds = xr.Dataset(
            data_vars={"temperature": (("x", "y"), [[1, 2], [3, 4]])},
            coords={"x": [0, 1], "y": [0, 1]},
            attrs={
                "complex_attr": 3 + 4j,
                "nested_dict": {"key": "value"},
                "numeric_str": "42.0",
            },
        )
        # Set a variable attribute
        ds["temperature"].attrs["long_attr"] = "1234"

        cleaned_ds = clean_and_convert_attrs(ds)

        # Verify that global attributes are converted
        self.assertEqual(cleaned_ds.attrs["complex_attr"], "3.0+4.0j")
        self.assertEqual(cleaned_ds.attrs["nested_dict"], '{"key": "value"}')
        self.assertEqual(cleaned_ds.attrs["numeric_str"], "42.0")

        # Verify that variable attributes are converted
        self.assertEqual(cleaned_ds["temperature"].attrs["long_attr"], "1234")


class TestRevertAndConvertAttrs(unittest.TestCase):
    def test_revert_netcdf_compatible_simple_types(self):
        self.assertEqual(revert_netcdf_compatible("test string"), "test string")
        self.assertEqual(revert_netcdf_compatible(42), 42)
        self.assertEqual(revert_netcdf_compatible(3.14), 3.14)
        self.assertEqual(revert_netcdf_compatible(True), True)

    def test_revert_json_string(self):
        json_str = '{"key": {"subkey": "value"}}'
        self.assertEqual(revert_netcdf_compatible(json_str), {"key": {"subkey": "value"}})

    def test_revert_complex_string(self):
        complex_str = "3.0+4.0j"
        self.assertEqual(revert_netcdf_compatible(complex_str), 3 + 4j)

    def test_revert_numeric_string(self):
        self.assertEqual(revert_netcdf_compatible("42"), 42)
        self.assertEqual(revert_netcdf_compatible("3.14"), 3.14)

        # Ensure non-numeric strings aren't mistakenly converted
        self.assertEqual(revert_netcdf_compatible("/path/to/file.txt"), "/path/to/file.txt")
        self.assertEqual(revert_netcdf_compatible("version 1.2"), "version 1.2")

    def test_no_conversion_needed(self):
        self.assertEqual(revert_netcdf_compatible("plain text"), "plain text")
        self.assertEqual(revert_netcdf_compatible("/path/to/file"), "/path/to/file")

    def test_revert_and_convert_attrs(self):
        # Create a dataset with converted attributes
        ds = xr.Dataset(
            attrs={
                "complex_attr": "3.0+4.0j",
                "nested_dict": '{"key": "value"}',
                "numeric_str": "42.0",
            }
        )
        ds["temperature"] = (("x", "y"), [[1, 2], [3, 4]])
        ds["temperature"].attrs["long_attr"] = "1234"

        reverted_ds = revert_and_convert_attrs(ds)

        # Verify the attributes are reverted as expected
        self.assertEqual(reverted_ds.attrs["complex_attr"], 3 + 4j)
        self.assertEqual(reverted_ds.attrs["nested_dict"], {"key": "value"})
        self.assertEqual(reverted_ds.attrs["numeric_str"], 42.0)

        # Verify that variable attributes are reverted
        self.assertEqual(reverted_ds["temperature"].attrs["long_attr"], 1234)


class TestNetCDFSaveCompatibility(unittest.TestCase):
    def test_dataset_saves_to_netcdf(self):
        # Create a sample dataset with global and variable-specific attributes
        ds = xr.Dataset(
            data_vars={"temperature": (("x", "y"), [[1, 2], [3, 4]])},
            coords={"x": [0, 1], "y": [0, 1]},
            attrs={"complex_attr": "3.0+4.0j", "nested_dict": '{"key": "value"}'},
        )

        # Add a variable attribute
        ds["temperature"].attrs["long_attr"] = "1234"

        # Attempt to save the dataset to an in-memory file-like object
        try:
            with io.BytesIO() as buffer:
                ds.to_netcdf(buffer)
            success = True
        except Exception as e:
            print(f"Failed to save dataset to NetCDF: {e}")
            success = False

        # Assert that the dataset could be saved without any errors
        self.assertTrue(success, "The dataset should be able to save to NetCDF format")


if __name__ == "__main__":
    unittest.main()
