"""eoio.processors.tests.test_base - tests for eoio.processors.base"""

import unittest

import xarray as xr

from eoio.processors.base import BaseProcessor

__author__ = ["Mattea Goalen <mattea.goalen@npl.co.uk>"]
__all__ = []


# dummy helper class to test BaseProcessor class
class DummyProcessorDict(BaseProcessor):
    name = "test_processor_dict"

    _default_process_params = {"r_key1": None, "r_key2": None}

    @property
    def all_options(self):
        """
        Return dictionary or list of available processing parameter
        """
        if self._all_options is None:
            self._all_options = {"r_key1": None, "r_key2": None}
        return self._all_options

    def process_dataset(self):
        """
        Process data and metadata and return an xarray.Dataset
        """


class DummyProcessorDictNoAbstractMethods(BaseProcessor):
    name = "test_processor_dict"

    _default_process_params = {"r_key1": None, "r_key2": None}

    @property
    def all_options(self):
        raise NotImplementedError

    def process_dataset(self):
        raise NotImplementedError


class DummyProcessorList(BaseProcessor):
    name = "test_processor_list"

    _default_process_params = ["test_1", "test_2"]

    @property
    def all_options(self):
        """
        Return dictionary or list of available processing parameter
        """
        if self._all_options is None:
            self._all_options = [True, "test_1", "test_2"]
        return self._all_options

    def process_dataset(self):
        """
        Process data and metadata and return an xarray.Dataset
        """
        raise NotImplementedError


class TestBaseProcessor(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test___init__abstractmethods(self):
        process_params = {"test_processor_dict": {"r_key2": "basic"}}
        input_ds = xr.Dataset()
        with self.assertRaises(TypeError):
            BaseProcessor(input_ds, process_params=process_params)

    def test___init__set_process_params_dict(self):
        process_params = {"test_processor_dict": {"r_key2": "basic"}}
        input_ds = xr.Dataset()
        dummy = DummyProcessorDict(input_ds, process_params=process_params)
        self.assertEqual(list(dummy.process_params.values()), [None, "basic"], "")

    def test___init__bad_process_params_dict(self):
        process_params = {"test_processor_dict": {"r_key2": "basic", "r_key": "initial"}}
        input_ds = xr.Dataset()
        with self.assertRaises(ValueError):
            DummyProcessorDict(input_ds, process_params=process_params)

    def test___init__no_all_options_process_params_dict(self):
        process_params = {"test_processor_dict": {"r_key2": "basic"}}
        input_ds = xr.Dataset()
        dummy = DummyProcessorDictNoAbstractMethods(input_ds, process_params=process_params)
        with self.assertRaises(NotImplementedError):
            dummy.all_options

    def test___init__no_process_dataset_dict(self):
        process_params = {"test_processor_dict": {"r_key2": "basic"}}
        input_ds = xr.Dataset()
        dummy = DummyProcessorDictNoAbstractMethods(input_ds, process_params=process_params)
        with self.assertRaises(NotImplementedError):
            dummy.process_dataset()

    def test___init__true_process_params_list(self):
        process_params = {"test_processor_list": True}
        input_ds = xr.Dataset()
        dummy = DummyProcessorList(input_ds, process_params=process_params)
        self.assertEqual(dummy.process_params, ["test_1", "test_2"])

    def test___init__str_process_params_list(self):
        process_params = {"test_processor_list": "test_1"}
        input_ds = xr.Dataset()
        dummy = DummyProcessorList(input_ds, process_params=process_params)
        self.assertEqual(dummy.process_params, "test_1")

    def test___init__false_process_params_list(self):
        process_params = {"test_processor_list": False}
        input_ds = xr.Dataset()
        with self.assertRaises(ValueError):
            DummyProcessorList(input_ds, process_params=process_params)

    def test___init__no_process_dataset_list(self):
        process_params = {"test_processor_list": "test_1"}
        input_ds = xr.Dataset()
        dummy = DummyProcessorList(input_ds, process_params=process_params)
        with self.assertRaises(NotImplementedError):
            dummy.process_dataset()

    def test__set_keys_invalid_params_type(self):
        process_params = {"test_processor_list": {"r_key2": "basic", "r_key": "initial"}}
        input_ds = xr.Dataset()
        with self.assertRaises(ValueError):
            DummyProcessorList(input_ds, process_params=process_params)


if __name__ == "__main__":
    unittest.main()
