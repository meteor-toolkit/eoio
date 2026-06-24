import unittest
from unittest.mock import patch, MagicMock
import xarray as xr

import eoio.processors.processor_pipeline as process_pipeline


class TestProcessorPipeline(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset(
            {
                "B02": xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("y", "x")),
                "B03": xr.DataArray([[10.0, 20.0], [30.0, 40.0]], dims=("y", "x")),
            }
        )
        self.context = {"ctx": True}

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    def test_unknown_processor_raises(self):
        params = {"does.not.exist": {"foo": "bar"}}

        with patch.dict(f"{process_pipeline.__name__}.PROCESSOR_REGISTRY", {}, clear=True):
            with self.assertRaises(process_pipeline.ProcessorPipelineError):
                process_pipeline.ProcessorPipeline(params, self.context)

    # -------------------------------------------------------------------------
    # LIST FORM
    # -------------------------------------------------------------------------

    def test_list_form_allows_duplicates(self):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.run.return_value = self.ds
        mock_cls.return_value = mock_inst

        params = [
            {"proc": {"x": 1}},
            {"proc": {"x": 2}},
        ]

        with patch.dict(
            f"{process_pipeline.__name__}.PROCESSOR_REGISTRY",
            {"proc": mock_cls},
            clear=True,
        ):
            pipeline = process_pipeline.ProcessorPipeline(params, self.context)
            pipeline.run(self.ds)

        self.assertEqual(mock_cls.call_count, 2)

    def test_invalid_list_form_raises(self):
        params = [{"a": {}}, {"b": {}, "c": {}}]  # invalid: second item has 2 keys

        with self.assertRaises(process_pipeline.ProcessorPipelineError):
            process_pipeline.ProcessorPipeline(params, self.context)

    # -------------------------------------------------------------------------
    # INSTANTIATION
    # -------------------------------------------------------------------------

    def test_processor_instantiation_with_params_and_context(self):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        mock_inst.run.return_value = self.ds

        params = {"units.convert": {"to": "radiance"}}

        with patch.dict(
            f"{process_pipeline.__name__}.PROCESSOR_REGISTRY",
            {"units.convert": mock_cls},
            clear=True,
        ):
            pipeline = process_pipeline.ProcessorPipeline(params, self.context)
            out = pipeline.run(self.ds)

        mock_cls.assert_called_once_with({"to": "radiance"}, self.context)
        mock_inst.run.assert_called_once_with(self.ds)
        self.assertIs(out, self.ds)

    # -------------------------------------------------------------------------
    # PARAMETER FORWARDING
    # -------------------------------------------------------------------------

    def test_all_params_forwarded_unchanged(self):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.run.return_value = self.ds
        mock_cls.return_value = mock_inst

        params = {"math.add": {"a": 10, "b": [1, 2]}}

        with patch.dict(
            f"{process_pipeline.__name__}.PROCESSOR_REGISTRY",
            {"math.add": mock_cls},
            clear=True,
        ):
            pipeline = process_pipeline.ProcessorPipeline(params, self.context)
            pipeline.run(self.ds)

        mock_cls.assert_called_once_with({"a": 10, "b": [1, 2]}, self.context)

    # -------------------------------------------------------------------------
    # CHAINING
    # -------------------------------------------------------------------------

    def test_processors_run_in_sequence(self):
        class Add:
            def __init__(self, params, context):
                self.var = params["var"]
                self.value = params["value"]
                self.params = params

            def run(self, ds):
                out = ds.copy()
                out[self.var] = out[self.var] + self.value
                return out

        class Ren:
            def __init__(self, params, context):
                self.old = params["old"]
                self.new = params["new"]
                self.params = params

            def run(self, ds):
                return ds.rename({self.old: self.new})

        reg = {"add": Add, "rename": Ren}

        params = [
            {"add": {"var": "B02", "value": 10}},
            {"add": {"var": "B02", "value": 5}},
            {"rename": {"old": "B02", "new": "B06"}},
        ]

        with patch.dict(f"{process_pipeline.__name__}.PROCESSOR_REGISTRY", reg, clear=True):
            pipeline = process_pipeline.ProcessorPipeline(params, self.context)
            out = pipeline.run(self.ds)

        xr.testing.assert_allclose(out["B06"], self.ds["B02"] + 15)
        self.assertIn("B03", out)

    # -------------------------------------------------------------------------
    # DATA FLOW (MOCKED)
    # -------------------------------------------------------------------------

    def test_data_flows_from_processor_to_processor(self):
        cls1 = MagicMock()
        cls2 = MagicMock()

        inst1 = MagicMock()
        inst2 = MagicMock()

        cls1.return_value = inst1
        cls2.return_value = inst2

        mid = self.ds.assign_attrs(stage="mid")
        final = self.ds.assign_attrs(stage="final")

        inst1.run.return_value = mid
        inst2.run.return_value = final

        params = {"p1": {"x": 1}, "p2": {"y": 2}}

        with patch.dict(
            f"{process_pipeline.__name__}.PROCESSOR_REGISTRY",
            {"p1": cls1, "p2": cls2},
            clear=True,
        ):
            pipeline = process_pipeline.ProcessorPipeline(params, self.context)
            out = pipeline.run(self.ds)

        cls1.assert_called_once_with({"x": 1}, self.context)
        cls2.assert_called_once_with({"y": 2}, self.context)

        inst1.run.assert_called_once_with(self.ds)
        inst2.run.assert_called_once_with(mid)
        self.assertIs(out, final)

    # -------------------------------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------------------------------

    def test_processor_exception_is_wrapped(self):

        class Processor:
            def __init__(self, params, context):
                self.params = params
                self.context = context

            def run(self, ds):
                raise ValueError("ProcessorError")

        params = {"processor": {}}

        with patch.dict(
            f"{process_pipeline.__name__}.PROCESSOR_REGISTRY",
            {"processor": Processor},
            clear=True,
        ):
            pipeline = process_pipeline.ProcessorPipeline(params, self.context)

            with self.assertRaises(process_pipeline.ProcessorPipelineError) as ctx:
                pipeline.run(self.ds)

        exc = ctx.exception
        self.assertIn("Error in processor: ProcessorError", str(exc))
        self.assertIsInstance(exc.__cause__, ValueError)

    # -------------------------------------------------------------------------
    # ERROR SKIP LOGIC
    # -------------------------------------------------------------------------

    def test_on_missing_skip_skips_processor(self):
        class Failing:
            def __init__(self, params, context):
                self.params = params

            def run(self, ds):
                raise RuntimeError("fail")

        class OK:
            def __init__(self, params, context):
                self.params = params

            def run(self, ds):
                return ds.assign_attrs(success=True)

        reg = {"bad": Failing, "good": OK}
        params = {
            "bad": {"on_missing": "skip"},
            "good": {},
        }

        with patch.dict(f"{process_pipeline.__name__}.PROCESSOR_REGISTRY", reg, clear=True):
            pipeline = process_pipeline.ProcessorPipeline(params, self.context)
            out = pipeline.run(self.ds)

        self.assertEqual(out.attrs["success"], True)

    # -------------------------------------------------------------------------
    # CONSTRUCTOR ERRORS
    # -------------------------------------------------------------------------

    def test_missing_required_constructor_param_raises(self):

        class Needs:
            def __init__(self, params, context):
                _ = params["required"]  # missing key

            def run(self, ds):
                return ds

        params = {"needs": {}}

        with patch.dict(
            f"{process_pipeline.__name__}.PROCESSOR_REGISTRY",
            {"needs": Needs},
            clear=True,
        ):
            with self.assertRaises(KeyError):
                process_pipeline.ProcessorPipeline(params, self.context)

    def test_custom_validation_errors_propagate(self):
        class Valid:
            def __init__(self, params, context):
                if params["x"] not in (1, 2):
                    raise ValueError("invalid x")

        params = {"v": {"x": 99}}

        with patch.dict(
            f"{process_pipeline.__name__}.PROCESSOR_REGISTRY",
            {"v": Valid},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                process_pipeline.ProcessorPipeline(params, self.context)


if __name__ == "__main__":
    unittest.main()
