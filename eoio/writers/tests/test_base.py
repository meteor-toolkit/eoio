"""Tests for eoio.writers.base"""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eoio.serialisation import decode_attrs
from eoio.serialisation.codecs import MARKER_ATTR
from eoio.writers.base import BaseWriter, WriteCapability


class DummyWriter(BaseWriter):
    """Minimal writer that records what it was handed instead of touching a format."""

    name = "dummy"

    default_write_params = {
        **BaseWriter.default_write_params,
        "flavour": "plain",
    }
    all_write_params = {
        **BaseWriter.all_write_params,
        "flavour": "plain/fancy",
    }

    def __init__(self, write_params=None):
        super().__init__(write_params=write_params)
        self.written = None
        self.written_path = None

    def write_dataset(self, obj, path):
        self.written = obj
        self.written_path = path
        Path(path).write_text("dummy")

    @staticmethod
    def get_extension():
        return ".dummy"


class PickyWriter(DummyWriter):
    """Writer that refuses anything without a platform attribute."""

    name = "picky"

    def can_write(self, obj):
        base = super().can_write(obj)
        if not base.ok:
            return base
        if "platform" not in obj.attrs:
            return WriteCapability(False, "requires a 'platform' attribute; none found")
        return WriteCapability(True)


class NonConformingWriter(DummyWriter):
    """Writer whose conformance check always complains, standing in for #205."""

    name = "nonconforming"

    def check_conformance(self, obj):
        return ["missing required attribute 'Conventions'"]


@pytest.fixture
def dataset():
    ds = xr.Dataset(
        data_vars={"B02": (("y", "x"), np.arange(4, dtype="float32").reshape(2, 2))},
        coords={"x": ("x", [0, 1]), "y": ("y", [0, 1])},
        attrs={
            "title": "test product",
            "product_metadata": {"datatake": {"id": "GS2A"}},
            "acquired": dt.datetime(2023, 7, 2, 10, 56, 19),
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_defaults_are_applied():
    writer = DummyWriter()
    assert writer.config.write_params["encode_attrs"] is True
    assert writer.config.write_params["flavour"] == "plain"


def test_provided_params_override_defaults():
    writer = DummyWriter(write_params={"flavour": "fancy"})
    assert writer.config.write_params["flavour"] == "fancy"
    assert writer.config.write_params["encode_attrs"] is True


def test_unknown_param_raises_matching_the_reader_behaviour():
    with pytest.raises(ValueError, match="Unknown writing parameter"):
        DummyWriter(write_params={"not_a_param": 1})


def test_all_options_reports_declared_params():
    options = DummyWriter().all_options
    assert options["write_params"]["flavour"] == "plain/fancy"
    assert "on_nonconformance" in options["write_params"]


def test_get_extension():
    assert DummyWriter.get_extension() == ".dummy"


def test_base_writer_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseWriter()


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


def test_generic_writer_accepts_dataset_and_datatree(dataset):
    writer = DummyWriter()
    assert writer.can_write(dataset).ok
    assert writer.can_write(xr.DataTree()).ok


@pytest.mark.parametrize("obj", ["a string", 42, None, {"not": "xarray"}])
def test_non_xarray_objects_are_refused(obj):
    capability = DummyWriter().can_write(obj)
    assert not capability.ok
    assert "Dataset or DataTree" in capability.reason


def test_write_raises_with_the_capability_reason(tmp_path, dataset):
    writer = PickyWriter()
    with pytest.raises(TypeError, match="requires a 'platform' attribute"):
        writer.write(dataset, tmp_path / "out.dummy")


def test_write_proceeds_when_capability_is_satisfied(tmp_path, dataset):
    dataset.attrs["platform"] = "Sentinel-2A"
    assert PickyWriter().write(dataset, tmp_path / "out.dummy")


def test_capability_is_truthy():
    assert WriteCapability(True)
    assert not WriteCapability(False, "nope")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_write_returns_the_path(tmp_path, dataset):
    target = tmp_path / "out.dummy"
    assert DummyWriter().write(dataset, target) == str(target)


def test_write_creates_parent_directories(tmp_path, dataset):
    target = tmp_path / "nested" / "deeper" / "out.dummy"
    DummyWriter().write(dataset, target)
    assert target.exists()


def test_parent_creation_can_be_disabled(tmp_path, dataset):
    target = tmp_path / "nested" / "out.dummy"
    with pytest.raises(FileNotFoundError):
        DummyWriter(write_params={"create_parents": False}).write(dataset, target)


def test_attrs_are_encoded_before_write_dataset(tmp_path, dataset):
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    handed_over = writer.written.attrs
    assert MARKER_ATTR in handed_over
    assert "product_metadata.datatake.id" in handed_over
    assert isinstance(handed_over["acquired"], str)


def test_encoding_can_be_disabled(tmp_path, dataset):
    writer = DummyWriter(write_params={"encode_attrs": False})
    writer.write(dataset, tmp_path / "out.dummy")

    assert MARKER_ATTR not in writer.written.attrs
    assert isinstance(writer.written.attrs["acquired"], dt.datetime)


def test_write_does_not_mutate_the_input(tmp_path, dataset):
    before = dict(dataset.attrs)
    DummyWriter().write(dataset, tmp_path / "out.dummy")

    assert dataset.attrs == before
    assert "eoio:processing_steps" not in dataset.attrs
    assert isinstance(dataset.attrs["acquired"], dt.datetime)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_history_is_appended(tmp_path, dataset):
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    history = writer.written.attrs["history"]
    assert "written to out.dummy" in history
    assert "dummy" in history


def test_existing_history_is_preserved(tmp_path, dataset):
    dataset.attrs["history"] = "2023-01-01: read in using eoio"
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    history = writer.written.attrs["history"]
    assert history.startswith("2023-01-01: read in using eoio")
    assert "written to out.dummy" in history


def test_processing_step_is_recorded_in_read_multi_shape(tmp_path, dataset):
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    steps = json.loads(writer.written.attrs["eoio:processing_steps"])
    assert steps[-1] == {"processor": "write", "writer": "dummy", "path": "out.dummy"}


def test_existing_processing_steps_are_extended(tmp_path, dataset):
    dataset.attrs["eoio:processing_steps"] = [{"processor": "read_multi", "n_files": 3}]
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    steps = json.loads(writer.written.attrs["eoio:processing_steps"])
    assert len(steps) == 2
    assert steps[0]["processor"] == "read_multi"
    assert steps[1]["processor"] == "write"


def test_json_string_processing_steps_from_an_undecoded_file_are_extended(tmp_path, dataset):
    # An object opened without decode_attrs carries the steps as their encoded JSON string.
    dataset.attrs["eoio:processing_steps"] = json.dumps([{"processor": "read_multi", "n_files": 3}])
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    steps = json.loads(writer.written.attrs["eoio:processing_steps"])
    assert [s["processor"] for s in steps] == ["read_multi", "write"]


@pytest.mark.parametrize("existing", ["not json", "42", '"quoted"'])
def test_unparseable_processing_steps_are_kept_as_one_step(tmp_path, dataset, existing):
    dataset.attrs["eoio:processing_steps"] = existing
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    steps = json.loads(writer.written.attrs["eoio:processing_steps"])
    assert steps[0] == existing
    assert steps[-1]["processor"] == "write"


def test_provenance_can_be_disabled(tmp_path, dataset):
    writer = DummyWriter(write_params={"update_history": False})
    writer.write(dataset, tmp_path / "out.dummy")

    assert "history" not in writer.written.attrs
    assert "eoio:processing_steps" not in writer.written.attrs


def test_provenance_survives_encoding_and_decoding(tmp_path, dataset):
    # Provenance is added before encoding, so it must be readable back out again.
    writer = DummyWriter()
    writer.write(dataset, tmp_path / "out.dummy")

    restored = decode_attrs(writer.written)
    assert restored.attrs["eoio:processing_steps"][-1]["writer"] == "dummy"


# ---------------------------------------------------------------------------
# Conformance policy
# ---------------------------------------------------------------------------


def test_nonconformance_warns_by_default(tmp_path, dataset):
    with pytest.warns(UserWarning, match="does not conform"):
        NonConformingWriter().write(dataset, tmp_path / "out.dummy")


def test_nonconformance_can_raise(tmp_path, dataset):
    writer = NonConformingWriter(write_params={"on_nonconformance": "error"})
    with pytest.raises(ValueError, match="Conventions"):
        writer.write(dataset, tmp_path / "out.dummy")


def test_nonconformance_can_be_ignored(tmp_path, dataset, recwarn):
    writer = NonConformingWriter(write_params={"on_nonconformance": "ignore"})
    writer.write(dataset, tmp_path / "out.dummy")
    assert not [w for w in recwarn if "does not conform" in str(w.message)]


def test_unknown_nonconformance_policy_raises(tmp_path, dataset):
    writer = DummyWriter(write_params={"on_nonconformance": "explode"})
    with pytest.raises(ValueError, match="on_nonconformance"):
        writer.write(dataset, tmp_path / "out.dummy")


def test_conforming_dataset_does_not_warn(tmp_path, dataset, recwarn):
    DummyWriter().write(dataset, tmp_path / "out.dummy")
    assert not [w for w in recwarn if "does not conform" in str(w.message)]


# ---------------------------------------------------------------------------
# DataTree
# ---------------------------------------------------------------------------


def test_datatree_is_accepted_and_gets_provenance(tmp_path):
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"campaign": {"name": "test"}}),
            "/s2": xr.Dataset({"v": ("x", [1, 2])}),
        }
    )
    writer = DummyWriter()
    writer.write(tree, tmp_path / "tree.dummy")

    assert "written to tree.dummy" in writer.written.attrs["history"]
    assert "campaign.name" in writer.written.attrs
