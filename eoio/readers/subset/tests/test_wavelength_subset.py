"""
ResolvedWavelengthSubset: Initialization and attribute checks.
validate_inputs decorator:

Valid data and params.
Invalid cases: empty data, missing keys, bad tolerance type.


WavelengthSubsetResolver methods:

get_min, get_max, get_nearest (with and without tolerance).
get_min_max.
run() returns correct ResolvedWavelengthSubset.
"""

import unittest
import numpy as np
from types import SimpleNamespace

from eoio.readers.subset.wavelength_subset import (
    WavelengthSubsetResolver,
)
from eoio.readers.subset.base_subset import ResolvedVariableSubset, VariableSubsetError


class TestResolvedWavelengthSubset(unittest.TestCase):
    def test_dataclass_initialization(self):
        subset = ResolvedVariableSubset(
            variable=450,
            variable_indices=[0, 1, 2],
            variable_dimensions=("wavelength",),
            method="nearest",
        )
        self.assertEqual(subset.variable, 450)
        self.assertEqual(subset.variable_indices, [0, 1, 2])
        self.assertEqual(subset.method, "nearest")


class TestValidateInputs(unittest.TestCase):
    def setUp(self):
        self.mock_data = SimpleNamespace(
            size=5,
            values=np.array([100, 200, 300, 400, 500]),
            dims=("wavelength",),
        )

    def test_valid_inputs(self):
        params = {"min": 200}
        resolver = WavelengthSubsetResolver(self.mock_data, params)
        self.assertIsInstance(resolver, WavelengthSubsetResolver)

    def test_empty_data_raises_error(self):
        bad_data = SimpleNamespace(size=0, values=np.array([]), dims=("wavelength",))
        with self.assertRaises(VariableSubsetError):
            WavelengthSubsetResolver(bad_data, {"min": 200})

    def test_multi_dimensional_size_raises_error(self):
        # Simulate object exposing a tuple size
        bad_data = SimpleNamespace(
            size=(2, 7),
            values=np.array([[100, 200], [300, 400]]),
            dims=("wavelength", "band"),
        )
        with self.assertRaises(VariableSubsetError):
            WavelengthSubsetResolver(bad_data, {"min": 200})

    def test_missing_valid_keys_raises_error(self):
        with self.assertRaises(VariableSubsetError):
            WavelengthSubsetResolver(self.mock_data, {"invalid": 123})

    def test_invalid_tolerance_type_raises_error(self):
        with self.assertRaises(VariableSubsetError):
            WavelengthSubsetResolver(self.mock_data, {"nearest": 300, "tolerance": "bad"})


class TestWavelengthSubsetResolverWithRun(unittest.TestCase):
    def setUp(self):
        self.values = np.array([100, 200, 300, 400, 500])
        self.mock_data = SimpleNamespace(
            size=5,
            values=self.values,
            dims=("wavelength",),
        )

    # ---- min
    def test_run_min(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"min": 300})
        result = resolver.run()
        self.assertEqual(result.method, "min")
        np.testing.assert_array_equal(np.array(result.variable_indices), np.array([2, 3, 4]))

    # ---- max
    def test_run_max(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"max": 300})
        result = resolver.run()
        self.assertEqual(result.method, "max")
        np.testing.assert_array_equal(np.array(result.variable_indices), np.array([0, 1, 2]))

    # ---- nearest (no tolerance)
    def test_run_nearest_without_tolerance(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"nearest": 320})
        result = resolver.run()
        self.assertEqual(result.method, "nearest")
        np.testing.assert_array_equal(np.array(result.variable_indices), np.array([2]))

    # ---- nearest (with tolerance)
    def test_run_nearest_with_tolerance(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"nearest": 320, "tolerance": 100})
        result = resolver.run()
        self.assertEqual(result.method, "nearest")
        np.testing.assert_array_equal(np.array(result.variable_indices), np.array([2, 3]))

    # ---- nearest (tolerance too small -> RasterSubsetError)
    def test_run_nearest_outside_tolerance_raises_error(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"nearest": 50, "tolerance": 10})
        with self.assertRaises(VariableSubsetError):
            resolver.run()

    # ---- min_max (range intersection)
    def test_run_min_max(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"min": 200, "max": 400})
        result = resolver.run()
        self.assertEqual(result.method, "min_max")
        self.assertEqual(result.variable, (200, 400))
        np.testing.assert_array_equal(np.array(result.variable_indices), np.array([1, 2, 3]))

    # ---- min_max (empty intersection when min > max)
    def test_run_min_max_empty_range(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"min": 400, "max": 200})
        result = resolver.run()
        self.assertEqual(result.method, "min_max")
        self.assertEqual(result.variable, (400, 200))
        # No values can be >= 400 and <= 200 simultaneously -> empty
        self.assertEqual(len(result.variable_indices), 0)

    # ---- smoke test: run returns dataclass with metadata
    def test_run_returns_resolved_subset(self):
        resolver = WavelengthSubsetResolver(self.mock_data, {"nearest": 300})
        result = resolver.run()
        self.assertIsInstance(result, ResolvedVariableSubset)
        self.assertEqual(result.method, "nearest")
        self.assertEqual(result.variable, 300)
        self.assertEqual(result.variable_dimensions, ("wavelength",))


if __name__ == "__main__":
    unittest.main()
