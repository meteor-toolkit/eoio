"""eoio.processors.units.drivers.registry - registry for unit conversion drivers"""

from __future__ import annotations
from typing import Dict, Type
from processor_tools import BaseProcessor

PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessor]] = {}  # example {"units.convert"}


def register_processor(name: str):
    """
    Decorator to register a processor under a stable name.

    :param name:
        Processor name used for selection (e.g. "units.convert").
    """

    def decorator(
        cls: Type[BaseProcessor],
    ) -> Type[BaseProcessor]:
        if name in PROCESSOR_REGISTRY:
            raise KeyError(f"Processor {name!r} already registered (existing: {PROCESSOR_REGISTRY[name].__name__}).")
        cls.name = name
        PROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


if __name__ == "__main__":
    pass
