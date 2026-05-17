"""eoio.processors.units.drivers.registry - registry for unit conversion drivers"""

from __future__ import annotations
from typing import Dict, Type
from eoio.processors.units.drivers.base import BaseUnitConversionDriver

UNIT_DRIVER_REGISTRY: Dict[str, Type[BaseUnitConversionDriver]] = {}


def register_unit_driver(name: str):
    """
    Decorator to register a unit conversion driver under a stable name.

    :param name:
        Driver name used for selection (e.g. "sentinel2", "landsat_c2").
    """

    def decorator(
        cls: Type[BaseUnitConversionDriver],
    ) -> Type[BaseUnitConversionDriver]:
        if name in UNIT_DRIVER_REGISTRY:
            raise KeyError(
                f"Unit driver {name!r} already registered (existing: {UNIT_DRIVER_REGISTRY[name].__name__})."
            )
        cls.name = name
        UNIT_DRIVER_REGISTRY[name] = cls
        return cls

    return decorator


if __name__ == "__main__":
    pass
