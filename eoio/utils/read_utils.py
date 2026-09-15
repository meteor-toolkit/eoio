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

    If *path_extracted* already exists, its contents are checked against the archive's own
    member list (see :py:func:`_extraction_is_complete`) before it's trusted -- a directory
    that merely *exists* isn't proof a previous extraction actually finished: an interrupted
    extraction (killed process, OOM, disk full, ...) leaves exactly that (a real, existing,
    but incomplete directory), which a bare ``os.path.exists`` check can't tell apart from a
    genuinely complete one. An incomplete extraction is removed and re-extracted from scratch
    rather than trusted, so a partial local copy doesn't get silently, permanently treated as
    complete.

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

    # os.path.isfile(path) guards both calls below: tarfile.is_tarfile (unlike
    # zipfile.is_zipfile, which safely returns False) raises IsADirectoryError rather than
    # returning False when given a directory -- and path IS a directory on a very real call
    # pattern here: for a reader whose get_extension() is "" (e.g. Landsat), path_extracted
    # equals path itself once already extracted, and callers legitimately pass that same
    # already-resolved product directory back in as path on a later read of the same
    # product. The original (pre-completeness-check) code never hit this, since it only ever
    # inspected path inside the "not yet extracted" branch -- this restores that same safety
    # while still allowing the completeness check below to run whenever path really is an
    # archive file.
    is_archive = os.path.isfile(path) and (tarfile.is_tarfile(path) or zipfile.is_zipfile(path))

    if os.path.exists(path_extracted):  # check if already extracted
        if is_archive and not _extraction_is_complete(path, path_extracted):
            warnings.warn(
                f"{path_extracted!r} exists but is missing or has differently-sized files "
                f"compared to the archive {path!r} -- likely an interrupted previous "
                "extraction. Removing it and re-extracting."
            )
            _remove_incomplete_extraction(path, path_extracted)
        else:
            if not read_params:
                read_params = {}
            if "save_extracted" in reader.default_read_params:
                read_params["save_extracted"] = True
            return path_extracted, read_params, extract_bool

    if is_archive:
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

    return path_extracted, read_params, extract_bool


def _archive_members(path: str) -> list:
    """``(relative_path, size)`` for every regular-file member of the tar/zip archive at
    *path* -- directory entries excluded, since extracting any one file that lives under a
    given directory recreates that directory as a side effect, so a directory entry's own
    presence isn't independent evidence of anything. Used by
    :py:func:`_extraction_is_complete`/:py:func:`_remove_incomplete_extraction` to know
    exactly what a complete extraction should (and does) contain.

    Works for both plain and compressed tar (``tarfile.open`` auto-detects gzip/bz2/xz), which
    is the format Landsat products come as here -- a compressed ``.tar.gz`` has no central
    index the way zip does, so building this list means a full sequential pass over the
    archive's contents (decompressing along the way); see :py:func:`_extraction_is_complete`'s
    own docstring for what that costs in practice.

    :param path: archive path (tar/tar.gz/zip -- anything :py:mod:`tarfile`/:py:mod:`zipfile`
        recognise).
    :return: list of ``(member_relative_path, size_bytes)``; empty if *path* isn't a
        recognised tar/zip archive at all.
    """
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            return [(member.name, member.size) for member in tar.getmembers() if member.isfile()]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return [(info.filename, info.file_size) for info in zf.infolist() if not info.is_dir()]
    return []


def _extraction_root_for(path: str, path_extracted: str) -> str:
    """Where *path*'s own members actually land on disk once extracted -- not always
    *path_extracted* itself. :py:func:`extract_tarred_file` extracts a tar's members relative
    to *path_extracted* (created for exactly this purpose via ``os.makedirs``), but
    :py:func:`extract_zipped_file` extracts a zip's members relative to
    ``os.path.dirname(path_extracted)`` instead (*path_extracted* only ends up existing as a
    directory at all if the zip's own internal member paths happen to be prefixed with that
    name) -- so the two formats need different roots to check membership against.

    :param path: archive path, to tell which of the two extraction conventions above applies.
    :param path_extracted: the target :py:func:`extract_file` resolved for *path*.
    :return: the directory *path*'s own extracted members should be found under.
    """
    if tarfile.is_tarfile(path):
        return path_extracted
    return os.path.dirname(path_extracted)


def _extraction_is_complete(path: str, path_extracted: str) -> bool:
    """Whether every regular-file member of the archive at *path* is present, and the right
    size, under wherever it actually extracts to (see :py:func:`_extraction_root_for`) --
    catches an interrupted/partial extraction that left *path_extracted* existing but
    incomplete, which a bare ``os.path.exists`` check can't tell apart from a genuinely
    finished one.

    This re-reads the archive's own member list on every call (see
    :py:func:`_archive_members`) -- for a compressed tar (Landsat's own format here), that
    means a full decompression pass, not a cheap lookup, so this has a real per-call cost
    proportional to the archive's own (compressed) size. Fine for how this project uses it
    today (called once per product per read, not in a tight loop), but if that stops being
    true, a written-once, stat-cheap completion marker (checked instead of a full re-scan
    whenever it's present and still matches the archive's own size/mtime) would remove that
    cost for the common already-extracted case without changing the underlying safety check.

    :param path: archive path.
    :param path_extracted: the target :py:func:`extract_file` resolved for *path* -- must
        already exist as a path (this function doesn't check that itself).
    :return: ``True`` if every member is present at its expected size, ``False`` otherwise
        (including if *path* isn't a recognised archive at all, in which case there's nothing
        to verify against and completeness can't be claimed).
    """
    members = _archive_members(path)
    if not members:
        return False
    root = _extraction_root_for(path, path_extracted)
    for member_name, size in members:
        member_path = os.path.join(root, member_name)
        if not os.path.isfile(member_path):
            return False
        if os.path.getsize(member_path) != size:
            return False
    return True


def _remove_incomplete_extraction(path: str, path_extracted: str) -> None:
    """Delete exactly the files the archive at *path* defines as members, wherever they
    actually landed (see :py:func:`_extraction_root_for`) -- not a blanket removal of
    *path_extracted* (safe for tar, whose members land inside it, but not for zip, whose
    members land in its *parent* directory alongside whatever else happens to be there) or of
    ``os.path.dirname(path_extracted)`` (would risk deleting unrelated sibling files/products
    that happen to share that directory).

    For tar, *path_extracted* itself is also pruned back to nothing afterward if it's now
    empty (recursively -- any subdirectories the removed files left behind too): otherwise
    :py:func:`extract_tarred_file`'s own ``if not os.path.exists(path_extracted)`` guard would
    see that (now-empty) directory still there and skip the re-extraction that's meant to
    follow this, silently leaving nothing behind at all. Zip's *path_extracted* is never
    pruned this way, since for zip it's :py:func:`_extraction_root_for`'s parent-directory
    case that matters, not *path_extracted* itself -- pruning it would only remove an
    incidental (and for zip, not even reliably created) empty directory, not anything
    :py:func:`extract_zipped_file` actually checks before extracting.

    :param path: archive path.
    :param path_extracted: the target :py:func:`extract_file` resolved for *path*.
    """
    root = _extraction_root_for(path, path_extracted)
    for member_name, _ in _archive_members(path):
        member_path = os.path.join(root, member_name)
        try:
            if os.path.isfile(member_path):
                os.remove(member_path)
        except OSError as e:
            print(f"{member_path} not removed due to error:", e)

    if root == path_extracted:
        _prune_empty_dirs(path_extracted)


def _prune_empty_dirs(root: str) -> None:
    """Remove *root* and any of its subdirectories left empty (bottom-up, so a directory only
    counts as empty once everything under it has already been removed) -- scoped strictly to
    *root* itself, never anything above it, so this can't reach outside the one directory
    :py:func:`_remove_incomplete_extraction` is cleaning up.

    :param root: directory to prune; a no-op if it doesn't exist or isn't a directory.
    """
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


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
