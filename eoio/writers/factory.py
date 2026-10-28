"""eoio.writers.factory - writer factory and registry

Writers are resolved **by name**, not by inspecting a path::

    WriterFactory.get_writer("netcdf")   # -> NetCDFWriter
    WriterFactory.list_writers()         # -> ["netcdf", "zarr", ...]

This is deliberately unlike :class:`eoio.readers.factory.ReaderFactory`, which matches
the product path against a table of regular expressions. The asymmetry is intentional:
on read, the format is a fact about the input that has to be discovered, whereas on
write it is the user's choice. Inferring it from the output extension is rejected
because it silently guesses, and because it does not extend to directory-style products
such as ``.SAFE``.

Built-in writers are imported lazily, so the cost of an optional backend is only paid
when that writer is actually used. Anything defined outside the package registers with
:func:`register_writer`, which is a decorator in the same style as
``@register_processor``::

    @register_writer("myformat")
    class MyWriter(BaseWriter):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Type, overload

if TYPE_CHECKING:
    from eoio.writers.base import BaseWriter

__all__ = ["WriterFactory", "register_writer", "unregister_writer"]


#: Built-in writers, as name -> (module path, class name). Imported on first use.
_BUILTIN_WRITERS: Dict[str, Tuple[str, str]] = {
    "netcdf": ("eoio.writers.netcdf.writer", "NetCDFWriter"),
    "zarr": ("eoio.writers.zarr.writer", "ZarrWriter"),
}

#: Writers registered at runtime, as name -> class.
_REGISTERED_WRITERS: Dict[str, Type[BaseWriter]] = {}


def _is_taken(name: str, cls: Type[BaseWriter]) -> bool:
    """
    Report whether ``name`` is already claimed by a writer other than ``cls``.

    A built-in writer module registers itself with the decorator when it is lazily
    imported, so finding its own name in the built-in table is expected and must not be
    treated as a clash. Anyone else claiming a built-in name still has to say so
    explicitly. Comparing the class identity rather than just the name keeps this
    independent of whether the built-in has been imported yet.

    :param name: Name being registered.
    :param cls: Class being registered under it.
    :returns: ``True`` when the name belongs to something else.
    """
    if name in _REGISTERED_WRITERS:
        return _REGISTERED_WRITERS[name] is not cls

    builtin = _BUILTIN_WRITERS.get(name)
    if builtin is None:
        return False
    return (cls.__module__, cls.__name__) != builtin


@overload
def register_writer(name: str, writer_cls: Type[BaseWriter], overwrite: bool = ...) -> Type[BaseWriter]: ...


@overload
def register_writer(
    name: str, writer_cls: None = ..., overwrite: bool = ...
) -> Callable[[Type[BaseWriter]], Type[BaseWriter]]: ...


def register_writer(
    name: str,
    writer_cls: Type[BaseWriter] | None = None,
    overwrite: bool = False,
) -> Type[BaseWriter] | Callable[[Type[BaseWriter]], Type[BaseWriter]]:
    """
    Register a writer class under a name.

    Used as a decorator, matching :func:`eoio.processors.registry.register_processor`::

        @register_writer("netcdf")
        class NetCDFWriter(BaseWriter):
            ...

    Or called directly, which is what you want for a class you do not own::

        register_writer("geotiff", SomeThirdPartyWriter)

    Either way the class gets its ``name`` set, so a registered writer always knows what
    it is called - which is what ends up in the provenance a write records.

    :param name: Name used to select the writer, e.g. ``"netcdf"``.
    :param writer_cls: Writer class. Omit it to use this as a decorator.
    :param overwrite: Whether replacing an already-registered name is permitted.
    :returns: The registered class, or the decorator that will register it.
    :raises KeyError: If the name is taken and ``overwrite`` is False.
    """

    def decorator(cls: Type[BaseWriter]) -> Type[BaseWriter]:
        if not overwrite and _is_taken(name, cls):
            raise KeyError(f"Writer {name!r} is already registered. Pass overwrite=True to replace it.")

        cls.name = name
        _REGISTERED_WRITERS[name] = cls
        return cls

    if writer_cls is not None:
        return decorator(writer_cls)
    return decorator


def unregister_writer(name: str) -> Type[BaseWriter]:
    """
    Remove a writer registered with :func:`register_writer`.

    :param name: Registered writer name.
    :returns: The removed class.
    :raises KeyError: If no writer was registered at runtime under that name.
    """
    try:
        return _REGISTERED_WRITERS.pop(name)
    except KeyError:
        raise KeyError(f"No runtime-registered writer named {name!r}.") from None


class WriterFactory:
    """
    Writer Factory class
    """

    def __init__(self):
        pass

    @staticmethod
    def get_writer(name: str) -> Type[BaseWriter]:
        """
        Return the writer registered under ``name``.

        :param name: Writer name, e.g. ``"netcdf"``.
        :returns: Writer class.
        :raises ValueError: If no writer is registered under that name.
        """
        if name in _REGISTERED_WRITERS:
            return _REGISTERED_WRITERS[name]

        if name in _BUILTIN_WRITERS:
            module_path, class_name = _BUILTIN_WRITERS[name]
            from importlib import import_module

            module = import_module(module_path)
            return getattr(module, class_name)

        available = WriterFactory.list_writers()
        raise ValueError(
            f"Unknown writer {name!r}. Available writers: {available if available else '(none registered)'}."
        )

    @staticmethod
    def list_writers() -> List[str]:
        """
        List the names of all available writers.

        :returns: Sorted writer names.
        """
        return sorted({*_BUILTIN_WRITERS, *_REGISTERED_WRITERS})


if __name__ == "__main__":
    pass
