"""Tests for eoio.writers.zarr.writer"""

import datetime as dt
import importlib.util

import numpy as np
import pytest
import xarray as xr

from eoio.serialisation import decode_attrs
from eoio.writers import WriterFactory
from eoio.writers.zarr.writer import ZarrWriter

# zarr is a declared dependency, but skip rather than fail in an environment that has
# not been reinstalled since it was added.
pytest.importorskip("zarr")


def _store_size(path) -> int:
    """Total size of every file in a zarr store."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@pytest.fixture
def dataset():
    ds = xr.Dataset(
        data_vars={"B02": (("y", "x"), np.arange(400, dtype="float32").reshape(20, 20))},
        coords={"x": ("x", np.arange(20)), "y": ("y", np.arange(20))},
        attrs={
            "title": "Sentinel-2A MSI Level-1C",
            "product_version": "03.01",
            "acquired": dt.datetime(2023, 7, 2, 10, 56, 19),
            "product_metadata": {"datatake": {"id": "GS2A", "orbit": 65}},
        },
    )
    ds["B02"].attrs = {"units": "1", "measurand": "toa_reflectance"}
    return ds


def time_step(t: int, **attrs) -> xr.Dataset:
    """One time step of a growing stack."""
    return xr.Dataset(
        {"v": ("time", np.array([t], dtype="float32"))},
        coords={"time": [t]},
        attrs={"title": "stack", **attrs},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_resolvable_through_the_factory():
    assert WriterFactory.get_writer("zarr") is ZarrWriter


def test_registered_name_and_extension():
    assert ZarrWriter.name == "zarr"
    assert ZarrWriter.get_extension() == ".zarr"


def test_listed_as_available():
    assert "zarr" in WriterFactory.list_writers()


def test_accepts_dataset_and_datatree(dataset):
    writer = ZarrWriter()
    assert writer.can_write(dataset).ok
    assert writer.can_write(xr.DataTree()).ok


def test_unknown_write_param_raises():
    with pytest.raises(ValueError, match="Unknown writing parameter"):
        ZarrWriter(write_params={"appenddim": "time"})


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_data_round_trips(tmp_path, dataset):
    path = ZarrWriter().write(dataset, tmp_path / "out.zarr")

    with xr.open_zarr(path) as back:
        xr.testing.assert_allclose(back["B02"], dataset["B02"])


def test_attrs_round_trip(tmp_path, dataset):
    path = ZarrWriter().write(dataset, tmp_path / "out.zarr")
    back = decode_attrs(xr.open_zarr(path))

    assert back.attrs["product_version"] == "03.01"
    assert back.attrs["acquired"] == dt.datetime(2023, 7, 2, 10, 56, 19)
    assert back.attrs["product_metadata"] == {"datatake": {"id": "GS2A", "orbit": 65}}
    assert back["B02"].attrs["measurand"] == "toa_reflectance"


def test_dataset_that_could_not_be_written_unaided(tmp_path, dataset):
    with pytest.raises(TypeError):
        dataset.to_zarr(tmp_path / "raw.zarr")

    assert ZarrWriter().write(dataset, tmp_path / "out.zarr")


def test_provenance_is_in_the_store(tmp_path, dataset):
    path = ZarrWriter().write(dataset, tmp_path / "out.zarr")
    back = decode_attrs(xr.open_zarr(path))

    assert "written to out.zarr" in back.attrs["history"]
    assert back.attrs["eoio:processing_steps"][-1] == {
        "processor": "write",
        "writer": "zarr",
        "path": "out.zarr",
    }


def test_read_multi_style_attrs_round_trip(tmp_path):
    steps = [{"processor": "read_multi", "concat_dim": "time", "n_files": 3}]
    ds = xr.Dataset(
        {"reflectance": (("time", "band"), np.zeros((3, 2), dtype="float32"))},
        coords={"time": [0, 1, 2], "band": ["B02", "B03"]},
        attrs={"eoio:processing_steps": steps},
    )

    path = ZarrWriter().write(ds, tmp_path / "multi.zarr")
    back = decode_attrs(xr.open_zarr(path))

    assert back.attrs["eoio:processing_steps"][0]["processor"] == "read_multi"


# ---------------------------------------------------------------------------
# Appending - the growing time series case
# ---------------------------------------------------------------------------


def test_append_grows_the_store(tmp_path):
    store = tmp_path / "stack.zarr"
    ZarrWriter().write(time_step(0), store)

    writer = ZarrWriter(write_params={"append_dim": "time"})
    writer.write(time_step(1), store)
    writer.write(time_step(2), store)

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["v"].values, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(back["time"].values, [0, 1, 2])


def test_the_same_call_works_for_the_first_and_later_steps(tmp_path):
    # The point of treating append_dim as first class: no special case for step one.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})

    for t in range(4):
        writer.write(time_step(t), store)

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["v"].values, [0.0, 1.0, 2.0, 3.0])


def test_append_to_a_missing_store_creates_it(tmp_path):
    store = tmp_path / "new.zarr"
    ZarrWriter(write_params={"append_dim": "time"}).write(time_step(0), store)

    assert store.exists()
    with xr.open_zarr(store) as back:
        assert back.sizes["time"] == 1


def test_append_mode_is_resolved_from_the_store(tmp_path):
    writer = ZarrWriter(write_params={"append_dim": "time"})
    store = tmp_path / "stack.zarr"

    assert writer._resolve_mode(store) == ("w", None)  # store absent -> create

    writer.write(time_step(0), store)
    assert writer._resolve_mode(store) == ("a", "time")  # store present -> append


def test_mode_applies_only_when_not_appending(tmp_path, dataset):
    writer = ZarrWriter(write_params={"mode": "w-"})
    assert writer._resolve_mode(tmp_path / "absent.zarr") == ("w-", None)


def test_encoding_is_dropped_when_appending(tmp_path):
    # zarr rejects encoding for a variable that already exists, so an append must not
    # carry the encoding that created the store.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time", "encoding": {"v": {"chunks": (1,)}}})

    writer.write(time_step(0), store)  # creates, encoding applies
    writer.write(time_step(1), store)  # appends, encoding must be dropped

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["v"].values, [0.0, 1.0])


def test_resolve_encoding_returns_none_when_appending():
    writer = ZarrWriter(write_params={"encoding": {"v": {"chunks": (1,)}}})
    assert writer._resolve_encoding("time") is None
    assert writer._resolve_encoding(None) == {"v": {"chunks": (1,)}}


def test_mode_alongside_append_dim_is_refused(tmp_path):
    # Contradictory: appending needs mode="a", and xarray rejects the combination
    # outright. Silently picking one of the two would do something unasked for.
    writer = ZarrWriter(write_params={"append_dim": "time", "mode": "w"})

    with pytest.raises(ValueError, match="contradicts append_dim"):
        writer.write(time_step(0), tmp_path / "stack.zarr")


def test_default_mode_is_not_treated_as_a_request(tmp_path):
    # "w" is the default, so it must not trip the contradiction check.
    writer = ZarrWriter(write_params={"append_dim": "time"})
    assert writer.config.write_params["mode"] == "w"

    writer.write(time_step(0), tmp_path / "stack.zarr")  # must not raise


def test_mode_a_is_accepted_alongside_append_dim(tmp_path):
    writer = ZarrWriter(write_params={"append_dim": "time", "mode": "a"})
    writer.write(time_step(0), tmp_path / "stack.zarr")

    with xr.open_zarr(tmp_path / "stack.zarr") as back:
        assert back.sizes["time"] == 1


# ---------------------------------------------------------------------------
# Appending - coordinates the store already holds
# ---------------------------------------------------------------------------


def test_rerunning_a_write_does_not_silently_duplicate(tmp_path):
    # Re-running after a partial failure is routine, and the artifact is permanent, so
    # duplicate time steps must not be the default outcome.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})
    for t in range(3):
        writer.write(time_step(t), store)

    with pytest.raises(ValueError, match="already in stack.zarr"):
        writer.write(time_step(1), store)

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["time"].values, [0, 1, 2])


def test_on_overlap_skip_writes_only_what_is_new(tmp_path):
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time", "on_overlap": "skip"})
    for t in range(3):
        writer.write(time_step(t), store)

    # A batch straddling the boundary contributes only its new half.
    batch = xr.concat([time_step(2), time_step(3)], dim="time")
    writer.write(batch, store)

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["time"].values, [0, 1, 2, 3])


def test_on_overlap_skip_makes_a_rerun_a_no_op(tmp_path):
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time", "on_overlap": "skip"})

    for _ in range(2):  # the whole run, twice over
        for t in range(3):
            writer.write(time_step(t), store)

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["time"].values, [0, 1, 2])
        np.testing.assert_array_equal(back["v"].values, [0.0, 1.0, 2.0])


def test_on_overlap_allow_appends_regardless(tmp_path):
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time", "on_overlap": "allow"})
    writer.write(time_step(0), store)
    writer.write(time_step(0), store)

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["time"].values, [0, 0])


def test_unknown_on_overlap_raises(tmp_path):
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time", "on_overlap": "ignore"})
    writer.write(time_step(0), store)

    with pytest.raises(ValueError, match="on_overlap must be"):
        writer.write(time_step(1), store)


def test_a_dimension_without_a_coordinate_is_appended_unchecked(tmp_path):
    # No coordinate means no identity to compare, so the policy has nothing to act on.
    store = tmp_path / "nocoord.zarr"
    writer = ZarrWriter(write_params={"append_dim": "n"})
    plain = xr.Dataset({"v": ("n", np.zeros(3, dtype="float32"))}, attrs={"title": "t"})

    writer.write(plain, store)
    writer.write(plain, store)

    with xr.open_zarr(store) as back:
        assert back.sizes["n"] == 6


def test_datetime_coordinates_are_compared_correctly(tmp_path):
    # The realistic case: a time axis, which round trips through CF encoding.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})

    def day(d):
        return xr.Dataset(
            {"v": ("time", np.array([float(d)], dtype="float32"))},
            coords={"time": [np.datetime64("2023-07-02") + np.timedelta64(d, "D")]},
            attrs={"title": "stack"},
        )

    writer.write(day(0), store)
    writer.write(day(1), store)

    with pytest.raises(ValueError, match="already in stack.zarr"):
        writer.write(day(1), store)


# ---------------------------------------------------------------------------
# Appending - attributes the store already holds
# ---------------------------------------------------------------------------


def test_history_accumulates_across_appends(tmp_path):
    # Zarr replaces group attrs wholesale, so without merging a stack would record only
    # its most recent append.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})
    for t in range(3):
        writer.write(time_step(t), store)

    back = decode_attrs(xr.open_zarr(store))
    assert len(back.attrs["history"].splitlines()) == 3


def test_history_lines_are_not_repeated(tmp_path):
    # Every scene carries the same reader preamble; repeating it once per append would
    # make the history of a long stack unreadable.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})
    for t in range(3):
        writer.write(time_step(t, history="2023-01-01T00:00:00: read by eoio"), store)

    back = decode_attrs(xr.open_zarr(store))
    lines = back.attrs["history"].splitlines()
    assert sum("read by eoio" in line for line in lines) == 1
    assert sum("written to" in line for line in lines) == 3


def test_processing_steps_accumulate_across_appends(tmp_path):
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})

    for t in range(3):
        step = time_step(t)
        step.attrs["eoio:processing_steps"] = [{"processor": "read", "file": f"scene_{t}.nc"}]
        writer.write(step, store)

    back = decode_attrs(xr.open_zarr(store))
    steps = back.attrs["eoio:processing_steps"]

    # Each distinct read is kept; the identical write step is recorded once.
    assert [s["file"] for s in steps if s["processor"] == "read"] == [
        "scene_0.nc",
        "scene_1.nc",
        "scene_2.nc",
    ]
    assert sum(s["processor"] == "write" for s in steps) == 1


def test_store_attrs_survive_an_append(tmp_path):
    # Metadata set when the store was created must not vanish because a later scene
    # happened not to carry it.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})

    writer.write(time_step(0, instrument="MSI", campaign="A"), store)
    writer.write(time_step(1), store)

    back = decode_attrs(xr.open_zarr(store))
    assert back.attrs["instrument"] == "MSI"
    assert back.attrs["campaign"] == "A"


def test_conflicting_attr_keeps_the_store_value_and_warns(tmp_path):
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})
    writer.write(time_step(0, campaign="A"), store)

    with pytest.warns(UserWarning, match="Keeping the store's value for campaign"):
        writer.write(time_step(1, campaign="B"), store)

    back = decode_attrs(xr.open_zarr(store))
    assert back.attrs["campaign"] == "A"


def test_new_attrs_from_an_appended_object_are_added(tmp_path):
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})

    writer.write(time_step(0), store)
    writer.write(time_step(1, processing_level="L2A"), store)

    back = decode_attrs(xr.open_zarr(store))
    assert back.attrs["processing_level"] == "L2A"


def test_merged_attrs_still_decode(tmp_path):
    # The encoding marker has to describe the merged attributes, not just the last
    # object's, or a merged-in value comes back as its encoded form.
    store = tmp_path / "stack.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})

    writer.write(time_step(0, acquired=dt.datetime(2023, 7, 2, 10, 56, 19)), store)
    writer.write(time_step(1, footprint={"type": "bbox", "coords": [0, 1]}), store)

    back = decode_attrs(xr.open_zarr(store))
    assert back.attrs["acquired"] == dt.datetime(2023, 7, 2, 10, 56, 19)
    assert back.attrs["footprint"] == {"type": "bbox", "coords": [0, 1]}


def test_a_fresh_write_does_not_merge_with_what_was_there(tmp_path):
    # Merging is an append-only concern: overwriting a store must not inherit its attrs.
    store = tmp_path / "stack.zarr"
    ZarrWriter().write(time_step(0, campaign="A"), store)
    ZarrWriter().write(time_step(1), store)

    back = decode_attrs(xr.open_zarr(store))
    assert "campaign" not in back.attrs


# ---------------------------------------------------------------------------
# Chunking and compression
# ---------------------------------------------------------------------------


def test_chunks_are_applied(tmp_path, dataset):
    path = ZarrWriter(write_params={"chunks": {"x": 5, "y": 5}}).write(dataset, tmp_path / "out.zarr")

    with xr.open_zarr(path) as back:
        assert back["B02"].encoding["chunks"] == (5, 5)


def test_chunks_default_to_leaving_the_object_alone(tmp_path, dataset):
    path = ZarrWriter().write(dataset, tmp_path / "out.zarr")

    with xr.open_zarr(path) as back:
        assert back["B02"].encoding["chunks"] == (20, 20)


def test_zarr_compresses_by_default(tmp_path):
    # Unlike netCDF, no compression defaults are needed here.
    big = xr.Dataset({"v": ("x", np.zeros(200_000, dtype="float32"))}, attrs={"title": "t"})
    ZarrWriter().write(big, tmp_path / "c.zarr")

    assert _store_size(tmp_path / "c.zarr") < 200_000 * 4 / 10


def test_compression_can_be_turned_off_through_encoding(tmp_path):
    big = xr.Dataset({"v": ("x", np.zeros(200_000, dtype="float32"))}, attrs={"title": "t"})
    ZarrWriter().write(big, tmp_path / "c.zarr")
    ZarrWriter(write_params={"encoding": {"v": {"compressors": None}}}).write(big, tmp_path / "u.zarr")

    assert _store_size(tmp_path / "u.zarr") > _store_size(tmp_path / "c.zarr")


# ---------------------------------------------------------------------------
# Store options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_zarr_format_is_honoured(tmp_path, dataset, zarr_format):
    path = ZarrWriter(write_params={"zarr_format": zarr_format}).write(dataset, tmp_path / f"v{zarr_format}.zarr")

    with xr.open_zarr(path) as back:
        xr.testing.assert_allclose(back["B02"], dataset["B02"])

    # Format 2 writes .zgroup metadata; format 3 writes zarr.json.
    marker = ".zgroup" if zarr_format == 2 else "zarr.json"
    assert (tmp_path / f"v{zarr_format}.zarr" / marker).exists()


def test_consolidated_can_be_disabled(tmp_path, dataset):
    path = ZarrWriter(write_params={"consolidated": False}).write(dataset, tmp_path / "out.zarr")

    # Read it back the same way, or xarray warns about falling back to unconsolidated.
    with xr.open_zarr(path, consolidated=False) as back:
        xr.testing.assert_allclose(back["B02"], dataset["B02"])


def test_append_checks_overlap_against_a_store_written_without_consolidated_metadata(tmp_path):
    # The store's state used to be read with the *write* setting, so a mismatch made it
    # unreadable, the overlap check silently disappeared, and the duplicates were appended.
    store = tmp_path / "out.zarr"
    ZarrWriter(write_params={"consolidated": False}).write(time_step(0), store)

    with pytest.raises(ValueError, match="already in"):
        ZarrWriter(write_params={"append_dim": "time"}).write(time_step(0), store)

    with xr.open_zarr(store, consolidated=False) as back:
        assert back.sizes["time"] == 1


def test_append_to_an_unreadable_store_raises_rather_than_skipping_the_check(tmp_path):
    store = tmp_path / "out.zarr"
    ZarrWriter().write(time_step(0), store)
    for metadata in store.glob("zarr.json"):
        metadata.write_text("{ not json")

    with pytest.raises(ValueError, match="Cannot read the existing store"):
        ZarrWriter(write_params={"append_dim": "time", "on_overlap": "allow"}).write(time_step(1), store)


def test_a_missing_zarr_package_points_at_the_extra(tmp_path, dataset, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "zarr" else real(name, *a, **k)
    )

    with pytest.raises(ImportError, match=r"pip install eoio\[zarr\]"):
        ZarrWriter().write(dataset, tmp_path / "out.zarr")


def test_appending_a_variable_the_store_lacks_is_refused_and_leaves_the_store_readable(tmp_path):
    store = tmp_path / "out.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})
    writer.write(time_step(0), store)

    extra = time_step(1)
    extra["extra"] = ("time", np.array([5.0]))

    with pytest.raises(ValueError, match="not in the store: extra"):
        writer.write(extra, store)

    with xr.open_zarr(store) as back:
        assert back.sizes["time"] == 1
    writer.write(time_step(1), store)
    with xr.open_zarr(store) as back:
        assert back.sizes["time"] == 2


def test_appending_without_a_variable_the_store_has_is_refused(tmp_path):
    store = tmp_path / "out.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})
    writer.write(time_step(0), store)

    with pytest.raises(ValueError, match="in the store but not in the appended object: v"):
        writer.write(time_step(1).rename({"v": "w"}).drop_vars("w"), store)


def test_a_variable_that_does_not_vary_along_the_append_dim_may_be_added(tmp_path):
    store = tmp_path / "out.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})
    writer.write(time_step(0), store)

    step = time_step(1)
    step["static"] = ((), 3.0)
    writer.write(step, store)

    with xr.open_zarr(store) as back:
        assert back.sizes["time"] == 2
        assert float(back["static"]) == 3.0


def test_fill_value_in_attrs_can_be_appended(tmp_path):
    # Readers such as RadCalNet's leave _FillValue in attrs; xarray refuses it on append.
    store = tmp_path / "out.zarr"
    writer = ZarrWriter(write_params={"append_dim": "time"})

    for t in range(3):
        step = time_step(t)
        step["v"].attrs["_FillValue"] = np.float32(-999.0)
        writer.write(step, store)

    with xr.open_zarr(store, decode_cf=False) as back:
        np.testing.assert_array_equal(back["v"].values, [0.0, 1.0, 2.0])
        assert back["v"].attrs["_FillValue"] == -999.0


def test_moving_the_fill_value_does_not_modify_the_input(tmp_path):
    step = time_step(0)
    step["v"].attrs["_FillValue"] = np.float32(-999.0)

    ZarrWriter().write(step, tmp_path / "out.zarr")

    assert step["v"].attrs["_FillValue"] == -999.0
    assert "_FillValue" not in step["v"].encoding


def test_sub_day_time_steps_survive_being_appended_one_at_a_time(tmp_path):
    # xarray alone picks "days since ..." from a single first value and then stores a
    # half-hour later step as 30 days, with no error.
    store = tmp_path / "out.zarr"
    times = np.array(["2022-01-01T17:00", "2022-01-01T17:30", "2022-01-01T18:00"], dtype="datetime64[us]")
    ds = xr.Dataset({"v": ("time", [1.0, 2.0, 3.0])}, coords={"time": times})
    writer = ZarrWriter(write_params={"append_dim": "time"})

    for i in range(3):
        writer.write(ds.isel(time=slice(i, i + 1)), store)

    with xr.open_zarr(store) as back:
        np.testing.assert_array_equal(back["time"].values, times.astype("datetime64[ns]"))


def test_a_repeated_sub_day_step_is_still_caught_as_an_overlap(tmp_path):
    store = tmp_path / "out.zarr"
    times = np.array(["2022-01-01T17:00", "2022-01-01T17:30"], dtype="datetime64[us]")
    ds = xr.Dataset({"v": ("time", [1.0, 2.0])}, coords={"time": times})
    writer = ZarrWriter(write_params={"append_dim": "time"})
    writer.write(ds.isel(time=slice(0, 1)), store)
    writer.write(ds.isel(time=slice(1, 2)), store)

    with pytest.raises(ValueError, match="already in"):
        writer.write(ds.isel(time=slice(1, 2)), store)


def test_explicit_time_encoding_is_not_overridden(tmp_path):
    store = tmp_path / "out.zarr"
    ds = xr.Dataset({"v": ("time", [1.0])}, coords={"time": np.array(["2022-01-01"], dtype="datetime64[ns]")})
    enc = {"time": {"units": "days since 2000-01-01", "dtype": "int64"}}

    ZarrWriter(write_params={"append_dim": "time", "encoding": enc}).write(ds, store)

    with xr.open_zarr(store, decode_times=False) as raw:
        assert raw["time"].attrs["units"] == "days since 2000-01-01"


def test_appending_into_a_group_that_is_not_in_the_store_yet_is_a_first_write(tmp_path):
    store = tmp_path / "out.zarr"
    ZarrWriter().write(time_step(0), store)

    ZarrWriter(write_params={"append_dim": "time", "group": "new"}).write(time_step(0), store)

    with xr.open_zarr(store, group="new") as back:
        assert back.sizes["time"] == 1


def test_group_is_honoured(tmp_path, dataset):
    path = ZarrWriter(write_params={"group": "measurements"}).write(dataset, tmp_path / "out.zarr")

    with xr.open_zarr(path, group="measurements") as back:
        assert "B02" in back


# ---------------------------------------------------------------------------
# DataTree
# ---------------------------------------------------------------------------


@pytest.fixture
def tree():
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"campaign": {"name": "test", "year": 2023}}),
            "/s2": xr.Dataset({"v": ("x", np.arange(10.0))}, attrs={"acquired": dt.datetime(2023, 7, 2, 10, 0)}),
            "/l8": xr.Dataset({"w": ("y", np.arange(5.0))}),
        }
    )


def test_datatree_writes_every_node(tmp_path, tree):
    path = ZarrWriter().write(tree, tmp_path / "tree.zarr")

    with xr.open_datatree(path, engine="zarr") as back:
        assert sorted(node.path for node in back.subtree) == ["/", "/l8", "/s2"]
        xr.testing.assert_allclose(back["s2"]["v"], tree["s2"]["v"])


def test_datatree_attrs_round_trip(tmp_path, tree):
    path = ZarrWriter().write(tree, tmp_path / "tree.zarr")
    back = decode_attrs(xr.open_datatree(path, engine="zarr"))

    assert back.attrs["campaign"] == {"name": "test", "year": 2023}
    assert back["s2"].attrs["acquired"] == dt.datetime(2023, 7, 2, 10, 0)


@pytest.mark.parametrize("params", [{"append_dim": "x"}, {"group": "g"}, {"append_dim": "x", "group": "g"}])
def test_datatree_with_append_dim_or_group_is_refused_before_anything_is_written(tmp_path, tree, params):
    # xarray implements neither for a tree. Refusing on the first write matters: a first
    # append is an ordinary write, so it would succeed and only the second would fail.
    writer = ZarrWriter(write_params=params)
    store = tmp_path / "tree.zarr"

    assert not writer.can_write(tree).ok
    with pytest.raises(TypeError, match="DataTree"):
        writer.write(tree, store)
    assert not store.exists()


def test_append_dim_and_group_are_still_accepted_for_a_dataset(tmp_path, dataset):
    assert ZarrWriter(write_params={"append_dim": "time", "group": "g"}).can_write(dataset).ok
