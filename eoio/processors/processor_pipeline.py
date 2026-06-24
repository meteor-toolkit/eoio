"""eoio.processors.processor_pipeline

Returns
-------
xarray.Dataset
    Processed xarray Dataset.

Raises
------
ProcessorPipelineError
    Raised when invalid processor specified.
    Raised when error occurred in the specified processors during the run.

"""

from typing import Any, Dict, List, Tuple, Union
from eoio.processors.registry import PROCESSOR_REGISTRY
import xarray as xr
from processor_tools import Context
import logging

logger = logging.getLogger(__name__)


class ProcessorPipelineError(ValueError):
    """
    Raised when a processor is invalid or fails during execution.
    """


class ProcessorPipeline:
    """
    Sequential processor pipeline that applies registered EOIO processors
    to an xarray.Dataset.

    The pipeline performs the following steps:
    - validates that the requested processors exist in the registry
    - instantiates them with their respective parameters
    - runs each processor in sequence

    Parameters
    --------------------

    :param processor_params:
        Either a dict or a list of single-key dicts mapping processor names to
        their parameter dictionaries. The list form allows the same processor
        to appear more than once with different parameters.

        The on_missing parameter can be used to specify how to handle errors for each processor, supported values are:
        - "error" (default): processor pipeline is stopped returning most recent successful processed dataset
        - "skip": processor pipeline omits processor when an error occurs

        Keys must exist in `PROCESSOR_REGISTRY`.

    :param context:
        Processing context (reader info, metadata view, logger, etc.).

    Example
    -------
    # Dict form (each processor once):
    processor_params = {
        "processor_name_0": {param_0:..., param_1:...., "on_missing":"skip"},
        "processor_name_1": {param_0:..., param_1:...., "on_missing":"error"}
    }

    # List form (allows repeated processors):
    processor_params = [
        {"interpolate": {param_0:..., "on_missing": "skip"}},
        {"interpolate": {param_0:..., "on_missing": "error"}},
    ]
    """

    def __init__(
        self,
        processor_params: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]],
        context: Dict[str, Any] | Context,
    ) -> None:

        self.steps: List[Tuple[str, Dict[str, Any]]] = self._normalise(processor_params)
        self.context: Dict[str, Any] = context
        self.processors: list[Any] = []
        self.registry_lc = {name.lower(): name for name in PROCESSOR_REGISTRY}

        self._validate_processor_names()
        self._instantiate_processors()

    @staticmethod
    def _normalise(
        processor_params: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Normalise dict or list-of-single-key-dicts to an ordered list of (name, params) tuples."""
        if isinstance(processor_params, dict):
            return [(name, params) for name, params in processor_params.items()]
        steps = []
        for item in processor_params:
            if not isinstance(item, dict) or len(item) != 1:
                raise ProcessorPipelineError(
                    "Each entry in a list-form processors definition must be a single-key dict, "
                    f'e.g. {{"interpolate": {{...}}}}. Got: {item!r}'
                )
            name, params = next(iter(item.items()))
            steps.append((name, params))
        return steps

    def _validate_processor_names(self) -> None:
        """Ensure all referenced processors exist in the registry."""

        for processor_name, _ in self.steps:
            if processor_name.lower() not in self.registry_lc:
                raise ProcessorPipelineError(
                    f"Unknown processor: {processor_name}, available processors are {sorted(PROCESSOR_REGISTRY.keys())}"
                )

    def _instantiate_processors(self) -> None:
        """Create processor instances with parameters."""

        for processor_name, params in self.steps:
            registry_name = self.registry_lc[processor_name.lower()]
            cls = PROCESSOR_REGISTRY[registry_name]
            processor_instance = cls(params, self.context)
            self.processors.append(processor_instance)

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Run all processors sequentially.

        :param ds:
            Input dataset.
        :return:
            The processed xarray dataset.
        :raises ProcessorPipelineError:
            If a processor fails and its `on_missing` is `"error"`.
        """

        logger.info(msg="Starting processor pipeline...")
        current_ds = ds
        for i, (processor, (step_name, _)) in enumerate(zip(self.processors, self.steps)):
            try:
                logger.info(f"Running processor {i + 1}: {step_name}")
                current_ds = processor.run(current_ds)
            except Exception as e:
                params = getattr(processor, "params", {})  # check if params exists

                on_missing = str(object=params.get("on_missing", "error")).lower()  # "error" | "skip"
                logger.error(msg=f"Error in processor {step_name}: {e}")

                if on_missing == "skip":
                    logger.warning(msg=f"Skipping processor {step_name} (index {i}) due to error: {e}")
                    # Continue without altering current_ds
                    continue

                # Default behaviour: raise a pipeline error
                raise ProcessorPipelineError(f"Error in {step_name}: {e}") from e

        logger.info("Pipeline completed successfully.")
        return current_ds
