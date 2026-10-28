"""eoio.writers - dataset writers

Writers serialise an ``xarray`` ``Dataset`` or ``DataTree`` to a file format. They are
selected explicitly by name rather than inferred from the output path::

    from eoio.writers import WriterFactory

    writer_cls = WriterFactory.get_writer("netcdf")
    writer_cls().write(ds, "output.nc")

See ``docs/content/developer_guide/writers/writers.rst`` for how to add one.
"""

from eoio.writers.base import BaseWriter, WritableObject, WriteCapability, WriterConfig
from eoio.writers.factory import WriterFactory, register_writer, unregister_writer

__all__ = [
    "BaseWriter",
    "WritableObject",
    "WriteCapability",
    "WriterConfig",
    "WriterFactory",
    "register_writer",
    "unregister_writer",
]
