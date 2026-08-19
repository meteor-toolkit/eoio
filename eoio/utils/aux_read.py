"""Shared helpers for resilient auxiliary data reading across readers."""

import warnings
from contextlib import contextmanager
from typing import Iterator

__all__ = ["warn_on_aux_failure"]


@contextmanager
def warn_on_aux_failure(description: str) -> Iterator[None]:
    """Run a block that reads one auxiliary data source, turning any exception it
    raises into a warning instead of letting it propagate.

    Auxiliary data (atmospheric correction inputs, angles, ancillary masks, etc.) is
    often unavailable or malformed for an individual product even when the core
    measurement data reads fine -- e.g. a missing or corrupt CAMS GRIB file for a
    Sentinel-2 SAFE product. Aborting the whole read in that situation is overly
    strict: the caller loses data (bands, angles, other aux sources) that read
    successfully. Wrap each independent aux source in this context manager so a
    failure there is reported and skipped, and the read continues with whatever aux
    data (and non-aux data) it does have.

    :param description: short human-readable label for what's being read, used in
        the warning message -- e.g. ``"Sentinel-2 CAMS meteo"``.
    """
    try:
        yield
    except Exception as e:
        warnings.warn(
            f"Failed to read {description} auxiliary data ({e!r}); skipping it and continuing with the rest of the read."
        )
