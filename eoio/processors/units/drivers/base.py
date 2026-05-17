"""eoio.processors.units.drivers.base - base class for unit conversion drivers"""

from abc import ABC, abstractmethod
from typing import Set, Tuple, Sequence, Mapping, Any, Optional
import xarray as xr


class BaseUnitConversionDriver(ABC):
    """
    Abstract base class for unit conversion drivers.

    Subclasses implement dataset-specific conversion logic (e.g. Landsat,
    Sentinel-2) while exposing a common interface to the eoio processor layer.

    A driver must:
    - declare whether it can handle a dataset via :meth:`matches`
    - declare which conversions it supports via :meth:`supported`
    - perform the conversion via :meth:`convert`
    """

    # Identifier for debugging / error messages
    name: str = "base"

    @abstractmethod
    def matches(self, ds, context) -> bool:
        """
        Determine whether this driver can handle the given dataset.

        This method should be cheap and rely on unambiguous indicators,
        such as normalised eoio metadata (e.g. ``ds.attrs["eoio:product"]``)
        or values provided in the processing context.

        :param ds:
            Input dataset.
        :param context:
            Processing context provided by eoio (reader metadata,
            resolved config, etc.).
        :return:
            ``True`` if this driver can convert units for this dataset,
            otherwise ``False``.
        """

        raise NotImplementedError

    @abstractmethod
    def supported(self) -> Set[Tuple[str, str]]:
        """
        Return the set of supported unit conversion pairs.

        Each entry is a ``(from_unit, to_unit)`` tuple using the public
        eoio unit vocabulary (e.g. ``("reflectance", "radiance")``).

        :return:
            Set of supported ``(from_unit, to_unit)`` pairs.
        """

        raise NotImplementedError

    @abstractmethod
    def convert(
        self,
        ds: xr.Dataset,
        to: str,
        var_names: Optional[Sequence[str]],
        *,
        context: Mapping[str, Any],
    ) -> xr.Dataset:
        """
        Convert variables in the dataset to the requested target units.

        Implementations are responsible for:
        - determining the source units of each variable
        - reading any required metadata (from ``ds`` or ``context``)
        - applying the correct conversion formula
        - updating variable ``units`` attributes consistently

        This method must not mutate the input dataset in-place unless
        explicitly documented; returning a modified copy is preferred.

        :param ds:
            Input dataset.
        :param to:
            Target unit name (e.g. ``"radiance"``).
        :param var_names:
            Names of variables to convert. If ``None``, the driver should
            determine a sensible default (e.g. measurement variables only).
        :param context:
            Processing context provided by eoio (reader metadata,
            resolved config, etc.).
        :return:
            Dataset with converted variables.
        """
        raise NotImplementedError


if __name__ == "__main__":
    pass
