"""Utility functions for reading/writing products from files"""

import gc
import json
import os
import shutil
import tarfile
import warnings
import zipfile
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import numpy as np
import xarray as xr

from eoio.readers.factory import ReaderFactory

__author__ = "Maddie Stedman <maddie.stedman@npl.co.uk>"
__all__ = []


def _strip_archive_suffixes(path: str) -> str:
    """
    Strip archive suffixes from a path so nested archives like .tar.gz resolve to the product root.

    :param path: archive path.
    :returns: path without archive suffixes.
    """
    archive_suffixes = {".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz", ".tbz2", ".txz", ".zip"}
    root = path

    while True:
        root, suffix = os.path.splitext(root)
        if suffix.lower() not in archive_suffixes:
            return root + suffix


@contextmanager  # type: ignore[arg-type]
def setup_file(path: str, read_params: Optional[dict]) -> None:  # type: ignore[misc]
    """
    Context manager for setting up file for reading - extracts file if necessary and yields path to extracted file

    :param path: path to file
    :param read_params: dictionary of parameters for reading data
    """
    if (
        read_params
        and read_params.get("save_extracted", False) is False
        and read_params.get("use_chunks", False) is True
    ):
        warnings.warn(
            "Conflicting read_params: 'use_chunks' is True but 'save_extracted' is False. Chunked reading requires extracted files. Setting 'save_extracted' to True."
        )
        read_params["save_extracted"] = True
    extract_bool = False
    try:
        path, read_params, extract_bool = extract_file(path, read_params)
        yield path
    finally:
        if not read_params:
            read_params = {}
        if extract_bool and os.path.exists(path) and read_params.get("save_extracted", False) is False:
            gc.collect()
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)  # remove dir and all contains
                else:
                    os.remove(path)
            except Exception as e:
                print(f"{path} not removed due to error:", e)


def extract_file(path: str, read_params: Optional[dict] = None) -> Tuple[str, Optional[dict], bool]:
    """
    Extracts data from file indicated by path

    :param path: path to data
    :param read_params: dictionary of parameters for reading data

    :returns: Tuple of extracted path, updated read_params, and extraction flag.
    """
    extract_bool = False

    # retrieve file extension from reader
    reader_factory = ReaderFactory()
    reader = reader_factory.get_reader(path)
    extension = reader.get_extension()

    # set extracted filepath
    if os.path.splitext(path)[-1] == extension:
        path_extracted = path
    else:
        path_extracted = os.path.join(_strip_archive_suffixes(path) + extension)

    if not os.path.exists(path_extracted):  # check if already extracted
        if tarfile.is_tarfile(path) or zipfile.is_zipfile(path):
            extract_bool = True
            # check if file is type zip or tar and extract accordingly
            if tarfile.is_tarfile(path):
                extract_tarred_file(path, path_extracted)
            elif zipfile.is_zipfile(path):
                extract_zipped_file(path, path_extracted)
            print(f"{path_extracted} extracted.")
        else:
            warnings.warn(f"File type of {path} not supported for extraction.")
            return path, read_params, extract_bool
        print(f"{path_extracted} extracted.")
    else:
        if not read_params:
            read_params = {}
        if "save_extracted" in reader.default_read_params:
            read_params["save_extracted"] = True

    return path_extracted, read_params, extract_bool


def extract_zipped_file(path: str, path_extracted: Optional[str] = None) -> None:
    """
    Extracts data from zipped file indicated by path

    :param path: path to data
    :param path_extracted: path to extracted data
    """
    if path_extracted is not None:
        if not os.path.exists(path_extracted):
            with zipfile.ZipFile(path, "r") as zfile:
                fileinfos = zfile.infolist()
                for fileinfo in fileinfos:
                    zfile.extract(
                        fileinfo,
                        path=os.path.dirname(path_extracted),
                    )
                zfile.close()


def extract_tarred_file(path: str, path_extracted: Optional[str] = None) -> None:
    """
    Extracts data from tarred file indicated by path

    :param path: path to data
    :param path_extracted: path to extracted data
    """
    if path_extracted is not None:
        if not os.path.exists(path_extracted):
            os.makedirs(path_extracted, exist_ok=True)
            tar = tarfile.open(path)
            tar.extractall(path_extracted)
            tar.close()


def convert_to_netcdf_compatible(attr: Any) -> Any:
    """
    Converts an attribute to a NetCDF-compatible type.

    :param attr: The attribute to check and convert.
    :returns: A NetCDF-compatible version of the attribute.
    """

    # Check for simple compatible types
    if isinstance(attr, (str, int, float, np.ndarray, list, tuple, bool, type(None))):
        return attr

    # Handle complex numbers by converting to a string
    if isinstance(attr, complex):
        return f"{attr.real}+{attr.imag}j"

    # Convert bytes and bytearrays to strings
    if isinstance(attr, (bytes, bytearray)):
        return attr.decode("utf-8", errors="replace")

    # Handle sets by converting to a list
    if isinstance(attr, set):
        return list(attr)

    # Handle nested dictionaries by converting to a JSON string
    if isinstance(attr, dict):
        return json.dumps(attr)

    # Handle generators or iterators by converting to a list
    if hasattr(attr, "__iter__") and not isinstance(attr, (str, np.ndarray)):
        return list(attr)

    # Fallback: convert to a string as a last resort
    return str(attr)


def clean_and_convert_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Converts attributes in an xarray Dataset to NetCDF-compatible types.

    :param ds: The input xarray Dataset.
    :returns: Dataset with attributes converted to NetCDF-compatible types.
    """

    # Convert global attributes
    for key, value in ds.attrs.items():
        new_value = convert_to_netcdf_compatible(value)
        if new_value != value:
            print(f"Converting global attribute: {key} from {type(value).__name__} to {type(new_value).__name__}")
        ds.attrs[key] = new_value

    # Convert variable-specific attributes
    for var_name in ds.variables:
        for key, value in ds[var_name].attrs.items():
            new_value = convert_to_netcdf_compatible(value)
            if new_value != value:
                print(
                    f"Converting attribute in variable '{var_name}': {key} from {type(value).__name__} to {type(new_value).__name__}"
                )
            ds[var_name].attrs[key] = new_value

    return ds


def revert_netcdf_compatible(attr: Any) -> Any:
    """
    Reverts a NetCDF-compatible attribute back to its original type if possible.

    :param attr: The attribute to revert.

    :returns: The reverted version of the attribute.
    """

    # Try to convert JSON strings back to dictionaries or lists
    if isinstance(attr, str):
        try:
            # Check if the string is valid JSON and convert it
            loaded_json = json.loads(attr)
            # Only return if the parsed JSON is indeed a dict or list
            if isinstance(loaded_json, (dict, list)):
                return loaded_json
        except (json.JSONDecodeError, TypeError):
            # If it's not JSON, continue to the next check
            pass

        # Try to convert strings representing complex numbers back to complex
        try:
            if "+" in attr and "j" in attr:
                return complex(attr)
        except ValueError:
            pass

    # Try to convert numeric strings back to numbers
    if isinstance(attr, str):
        try:
            # Check if the attribute is numeric without leading/trailing non-numeric characters
            if attr.replace(".", "", 1).isdigit() or (attr.startswith("-") and attr[1:].replace(".", "", 1).isdigit()):
                if "." in attr:
                    return float(attr)
                else:
                    return int(attr)
        except ValueError:
            pass

    # Return the attribute unchanged if no conversion was done
    return attr


def revert_and_convert_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Converts attributes in an xarray Dataset back to their original types if possible.

    :param ds: The xarray Dataset to revert attributes in.
    :returns: Dataset with attributes reverted to their original types where possible.
    """

    # Revert global attributes
    for key, value in ds.attrs.items():
        new_value = revert_netcdf_compatible(value)
        if new_value != value:
            print(f"Reverting global attribute: {key} from {type(value).__name__} to {type(new_value).__name__}")
        ds.attrs[key] = new_value

    # Revert variable-specific attributes
    for var_name in ds.variables:
        for key, value in ds[var_name].attrs.items():
            new_value = revert_netcdf_compatible(value)
            if new_value != value:
                print(
                    f"Reverting attribute in variable '{var_name}': {key} from {type(value).__name__} to {type(new_value).__name__}"
                )
            ds[var_name].attrs[key] = new_value

    return ds


if __name__ == "__main__":
    pass
