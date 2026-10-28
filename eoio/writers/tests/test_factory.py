"""Tests for eoio.writers.factory"""

import pytest
import xarray as xr

from eoio.writers import factory
from eoio.writers.factory import (
    _BUILTIN_WRITERS,
    WriterFactory,
    register_writer,
    unregister_writer,
)
from eoio.writers.tests.test_base import DummyWriter


@pytest.fixture
def registered():
    """Register the dummy writer and clean up afterwards."""
    register_writer("dummy", DummyWriter)
    yield DummyWriter
    unregister_writer("dummy")


def test_get_writer_returns_the_registered_class(registered):
    assert WriterFactory.get_writer("dummy") is DummyWriter


def test_register_writer_works_as_a_decorator():
    # The form that matches @register_processor, and the one a new writer will use.
    @register_writer("decorated")
    class DecoratedWriter(DummyWriter):
        pass

    try:
        assert WriterFactory.get_writer("decorated") is DecoratedWriter
        assert DecoratedWriter.name == "decorated"
    finally:
        unregister_writer("decorated")


def test_decorator_returns_the_class_itself():
    @register_writer("still_usable")
    class DecoratedWriter(DummyWriter):
        pass

    try:
        # The decorator must not replace the class with something else.
        assert isinstance(DecoratedWriter(), DummyWriter)
        assert DecoratedWriter.get_extension() == ".dummy"
    finally:
        unregister_writer("still_usable")


def test_decorator_form_rejects_a_duplicate_name(registered):
    with pytest.raises(KeyError, match="already registered"):

        @register_writer("dummy")
        class Clashing(DummyWriter):
            pass


def test_decorator_form_honours_overwrite(registered):
    @register_writer("dummy", overwrite=True)
    class Replacement(DummyWriter):
        pass

    assert WriterFactory.get_writer("dummy") is Replacement


def test_register_writer_sets_the_name(registered):
    assert DummyWriter.name == "dummy"


def test_list_writers_includes_registered(registered):
    assert "dummy" in WriterFactory.list_writers()


def test_unknown_writer_raises_and_lists_what_is_available(registered):
    with pytest.raises(ValueError, match="Unknown writer 'nope'"):
        WriterFactory.get_writer("nope")

    with pytest.raises(ValueError, match="dummy"):
        WriterFactory.get_writer("nope")


def test_unknown_writer_says_so_when_nothing_is_registered(monkeypatch):
    monkeypatch.setattr(factory, "_BUILTIN_WRITERS", {})
    monkeypatch.setattr(factory, "_REGISTERED_WRITERS", {})

    with pytest.raises(ValueError, match=r"\(none registered\)"):
        WriterFactory.get_writer("nope")


def test_registering_a_different_class_under_a_taken_name_raises(registered):
    class Impostor(DummyWriter):
        pass

    with pytest.raises(KeyError, match="already registered"):
        register_writer("dummy", Impostor)


def test_overwrite_replaces_the_registration(registered):
    class OtherWriter(DummyWriter):
        pass

    register_writer("dummy", OtherWriter, overwrite=True)
    assert WriterFactory.get_writer("dummy") is OtherWriter


def test_unregister_removes_it():
    register_writer("temp", DummyWriter)
    unregister_writer("temp")
    assert "temp" not in WriterFactory.list_writers()


def test_unregistering_an_unknown_writer_raises():
    with pytest.raises(KeyError, match="No runtime-registered writer"):
        unregister_writer("never_registered")


def test_writer_is_selected_by_name_not_by_path(registered):
    # The asymmetry with ReaderFactory is deliberate: nothing about the output path
    # influences which writer is returned.
    assert WriterFactory.get_writer("dummy") is DummyWriter
    with pytest.raises(ValueError):
        WriterFactory.get_writer("output.dummy")


@pytest.fixture
def fake_builtin():
    """Point the built-in table at the dummy writer, as a real built-in would."""
    _BUILTIN_WRITERS["fake"] = (DummyWriter.__module__, DummyWriter.__name__)
    yield
    _BUILTIN_WRITERS.pop("fake", None)
    try:
        unregister_writer("fake")
    except KeyError:
        pass


def test_a_builtin_may_register_itself(fake_builtin):
    # A built-in module runs its own @register_writer when lazily imported; finding its
    # own name in the built-in table must not count as a clash.
    register_writer("fake", DummyWriter)
    assert WriterFactory.get_writer("fake") is DummyWriter


def test_another_class_cannot_claim_a_builtin_name(fake_builtin):
    class Impostor(DummyWriter):
        pass

    with pytest.raises(KeyError, match="already registered"):
        register_writer("fake", Impostor)


def test_a_builtin_name_can_be_claimed_with_overwrite(fake_builtin):
    class Replacement(DummyWriter):
        pass

    register_writer("fake", Replacement, overwrite=True)
    assert WriterFactory.get_writer("fake") is Replacement


def test_re_registering_the_same_class_is_allowed(registered):
    # Idempotent, so a module being reloaded does not explode.
    register_writer("dummy", DummyWriter)
    assert WriterFactory.get_writer("dummy") is DummyWriter


def test_builtin_table_is_lazy():
    # Entries are (module, class) pairs resolved on use, so importing the factory does
    # not import any format backend.
    for name, entry in _BUILTIN_WRITERS.items():
        assert isinstance(entry, tuple) and len(entry) == 2, name
        assert isinstance(entry[0], str) and isinstance(entry[1], str), name


def test_end_to_end_through_the_factory(tmp_path, registered):
    writer_cls = WriterFactory.get_writer("dummy")
    target = writer_cls().write(xr.Dataset(attrs={"title": "x"}), tmp_path / "out.dummy")
    assert target.endswith("out.dummy")
