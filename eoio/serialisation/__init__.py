"""eoio.serialisation - attribute (de)serialisation shared by readers and writers

Readers populate ``attrs`` with rich Python objects that netCDF and Zarr cannot store.
This package converts them on the way out and restores them on the way back:

    >>> from eoio.serialisation import encode_attrs, decode_attrs
    >>> encode_attrs(ds).to_netcdf("out.nc")           # doctest: +SKIP
    >>> ds = decode_attrs(xr.open_dataset("out.nc"))   # doctest: +SKIP

Support for a new attribute type is added by registering a codec - see
:func:`eoio.serialisation.codecs.register_codec` - rather than by modifying this package.
"""

from eoio.serialisation.attrs import decode_attrs, encode_attrs
from eoio.serialisation.codecs import (
    MARKER_ATTR,
    AttrCodec,
    get_codec,
    is_serialisable,
    list_codecs,
    register_codec,
    select_codec,
    unregister_codec,
)

__all__ = [
    "MARKER_ATTR",
    "AttrCodec",
    "decode_attrs",
    "encode_attrs",
    "get_codec",
    "is_serialisable",
    "list_codecs",
    "register_codec",
    "select_codec",
    "unregister_codec",
]
