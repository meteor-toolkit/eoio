"""EMIT reader layout helpers.

This module contains a small helper class used by the EMIT readers to
normalise and validate on-disk product paths for radiance (RAD) and
observation (OBS) products.

The project expects a RAD product path as the canonical path used by
readers; when given an OBS path the layout will convert it to the
corresponding RAD path (by replacing ``OBS`` with ``RAD``).

Only light-weight filesystem checks are performed here; more extensive
validation happens in the reader implementations (see
``eoio/readers/emit``).

Module API
----------
EmitLayout
    Small helper for formatting and validating EMIT product paths.

"""

from __future__ import annotations
import os


class EmitLayout:
    """Provide filesystem-aware path helpers for EMIT products.

    Parameters
    ----------
    path : str
        Input path to an EMIT product. May contain either ``RAD`` or
        ``OBS`` as the product type token. The constructor will
        normalise the path to a ``RAD`` path and validate that both the
        RAD and corresponding OBS paths exist on disk.

    Attributes
    ----------
    path : str
        Normalised path (guaranteed to contain ``RAD``).

    Raises
    ------
    ValueError
        If the provided path does not contain a recognised token or if
        the expected files/directories cannot be found.
    """

    def __init__(self, path: str):
        self.path = self.format_path(str(path))
        self.check_path(self.path)

    @staticmethod
    def format_path(path):
        """Normalise an EMIT product path to the RAD product.

        Parameters
        ----------
        path : str
            Input path that should contain either the substring ``'RAD'``
            or ``'OBS'`` indicating the product type.

        Returns
        -------
        str
            The path containing ``'RAD'``. If the input already contains
            ``'RAD'`` it is returned unchanged; if it contains ``'OBS'``
            the returned value is a string where the first occurrence of
            ``'OBS'`` is replaced with ``'RAD'``.

        Raises
        ------
        ValueError
            If the input path contains neither ``'RAD'`` nor ``'OBS'``.

        Notes
        -----
        This is a simple string-based normalisation used by the EMIT
        readers in this package; it performs no filesystem access.
        """
        if "RAD" in path:
            return path
        elif "OBS" in path:
            return path.replace("OBS", "RAD")
        else:
            raise ValueError("invalid path for EMIT reader, must contain 'RAD' or 'OBS'")

    @staticmethod
    def check_path(path):
        """Verify that both RAD and OBS product paths exist on disk.

        Parameters
        ----------
        path : str
            Path expected to contain ``'RAD'``. The method will also
            check for the corresponding OBS path by substituting
            ``'RAD'`` -> ``'OBS'``.

        Raises
        ------
        ValueError
            If either the RAD path or the derived OBS path does not
            exist on the filesystem.

        Notes
        -----
        This helper intentionally raises ``ValueError`` for missing
        files to keep behaviour simple for callers; callers that need
        different error handling should call ``os.path.exists``
        themselves.
        """
        if os.path.exists(path) and os.path.exists(path.replace("RAD", "OBS")):
            return None
        else:
            raise ValueError(f"{path} cannot be found")
