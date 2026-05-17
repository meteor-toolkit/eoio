"""Unit tests for eoio.processors.s2_rut.processor."""

import unittest
from unittest.mock import patch

import xarray as xr

from eoio.processors.s2_rut.processor import S2RUTConfig, S2Rut


class TestS2RutInit(unittest.TestCase):
    @patch("eoio.processors.s2_rut.processor.BaseProcessor.__init__", return_value=None)
    @patch.object(S2Rut, "_parse_params")
    def test_init_uses_empty_params_when_none(self, mock_parse_params, _mock_base_init):
        parsed = S2RUTConfig(data_vars=["B02"], group_unc=True)
        mock_parse_params.return_value = parsed

        proc = S2Rut(params=None, context={"reader": "dummy"})

        mock_parse_params.assert_called_once_with({})
        self.assertIs(proc.rut_config, parsed)

    @patch("eoio.processors.s2_rut.processor.BaseProcessor.__init__", return_value=None)
    @patch.object(S2Rut, "_parse_params")
    def test_init_forwards_params_to_parse(self, mock_parse_params, _mock_base_init):
        params = {"data_vars": ["B03"], "group_unc": False}
        parsed = S2RUTConfig(data_vars=["B03"], group_unc=False)
        mock_parse_params.return_value = parsed

        proc = S2Rut(params=params)

        mock_parse_params.assert_called_once_with(params)
        self.assertIs(proc.rut_config, parsed)


class TestS2RutParseParams(unittest.TestCase):
    def test_parse_params_defaults(self):
        proc = S2Rut()

        config = proc._parse_params({})

        self.assertEqual(config, S2RUTConfig())

    def test_parse_params_reads_fields_and_normalizes_on_missing(self):
        proc = S2Rut()
        params = {
            "data_vars": ["B02", "B03"],
            "group_unc": False,
            "subset_unc": ["u_rel"],
            "on_missing": "SKIP",
        }

        config = proc._parse_params(params)

        self.assertEqual(config.data_vars, ["B02", "B03"])
        self.assertFalse(config.group_unc)
        self.assertEqual(config.subset_unc, ["u_rel"])
        self.assertEqual(config.on_missing, "skip")

    def test_parse_params_invalid_on_missing_raises(self):
        proc = S2Rut()

        with self.assertRaises(ValueError):
            proc._parse_params({"on_missing": "invalid"})


class TestS2RutRun(unittest.TestCase):
    def test_run_raises_type_error_for_non_dataset(self):
        proc = S2Rut(params={"data_vars": ["B02"], "group_unc": True})

        with patch("s2_rut_python.interface.S2RUTTool") as mock_tool_cls:
            with self.assertRaises(TypeError):
                proc.run("not a dataset")

        mock_tool_cls.assert_not_called()

    @patch("s2_rut_python.interface.S2RUTTool")
    def test_run_calls_rut_tool_with_config(self, mock_tool_cls):
        proc = S2Rut(params={"data_vars": ["B02"], "group_unc": False})
        ds_in = xr.Dataset({"B02": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
        ds_after_rut = xr.Dataset({"B02_unc": ("x", [0.1, 0.2])}, coords={"x": [0, 1]})
        ds_after_provenance = ds_after_rut.copy(deep=True)

        tool = mock_tool_cls.return_value
        tool.run.return_value = ds_after_rut

        with patch.object(proc, "_record_provenance", return_value=ds_after_provenance) as mock_record:
            result = proc.run(ds_in)

        mock_tool_cls.assert_called_once_with()
        tool.run.assert_called_once_with(
            ds_in,
            data_vars=["B02"],
            group_unc=False,
        )
        mock_record.assert_called_once_with(ds_after_rut)
        self.assertIs(result, ds_after_provenance)


class TestS2RutRecordProvenance(unittest.TestCase):
    def test_record_provenance_appends_to_existing_steps_list(self):
        proc = S2Rut(params={"data_vars": ["B02"]})
        ds = xr.Dataset({"v": ("x", [1])}, coords={"x": [0]})
        ds.attrs["eoio:processing_steps"] = [{"processor": "existing"}]

        out = proc._record_provenance(ds)

        self.assertIs(out, ds)
        steps = out.attrs["eoio:processing_steps"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0], {"processor": "existing"})
        self.assertEqual(steps[-1]["processor"], "s2_rut")
        self.assertEqual(steps[-1]["rut_config"], proc.rut_config)

    def test_record_provenance_wraps_non_list_steps(self):
        proc = S2Rut(params={})
        ds = xr.Dataset({"v": ("x", [1])}, coords={"x": [0]})
        ds.attrs["eoio:processing_steps"] = "legacy-step"

        out = proc._record_provenance(ds)

        steps = out.attrs["eoio:processing_steps"]
        self.assertIsInstance(steps, list)
        self.assertEqual(steps[0], "legacy-step")
        self.assertEqual(steps[-1]["processor"], "s2_rut")


if __name__ == "__main__":
    unittest.main()
