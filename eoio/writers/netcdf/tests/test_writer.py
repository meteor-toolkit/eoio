"""Tests for eoio.writers.netcdf.writer"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eoio.serialisation import decode_attrs
from eoio.writers import WriterFactory
from eoio.writers.netcdf.writer import NetCDFWriter, _default_engine


@pytest.fixture
def dataset():
    """A dataset shaped like reader output, with attrs netCDF cannot store natively."""
    ds = xr.Dataset(
        data_vars={
            "B02": (("y", "x"), np.arange(400, dtype="float32").reshape(20, 20)),
            "quality": ((), 1.0),
        },
        coords={"x": ("x", np.arange(20)), "y": ("y", np.arange(20))},
        attrs={
            "title": "Sentinel-2A MSI Level-1C",
            "product_version": "03.01",
            "acquired": dt.datetime(2023, 7, 2, 10, 56, 19),
            "product_metadata": {"datatake": {"id": "GS2A", "orbit": 65}},
        },
    )
    ds["B02"].attrs = {"units": "1", "measurand": "toa_reflectance"}
    ds["x"].attrs = {"standard_name": "projection_x_coordinate"}
    return ds


@pytest.fixture
def big_dataset():
    """Large enough that compression is measurable."""
    return xr.Dataset(
        {"v": ("x", np.zeros(200_000, dtype="float32"))},
        coords={"x": np.arange(200_000)},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_resolvable_through_the_factory():
    assert WriterFactory.get_writer("netcdf") is NetCDFWriter


def test_registered_name():
    assert NetCDFWriter.name == "netcdf"


def test_listed_as_available():
    assert "netcdf" in WriterFactory.list_writers()


def test_extension():
    assert NetCDFWriter.get_extension() == ".nc"


def test_accepts_dataset_and_datatree(dataset):
    writer = NetCDFWriter()
    assert writer.can_write(dataset).ok
    assert writer.can_write(xr.DataTree()).ok


def test_options_are_discoverable():
    options = NetCDFWriter().all_options["write_params"]
    assert "engine" in options
    assert "complevel" in options


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_data_round_trips(tmp_path, dataset):
    path = NetCDFWriter().write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        xr.testing.assert_allclose(back["B02"], dataset["B02"])


def test_attrs_round_trip(tmp_path, dataset):
    path = NetCDFWriter().write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as opened:
        back = decode_attrs(opened)

    assert back.attrs["title"] == "Sentinel-2A MSI Level-1C"
    assert back.attrs["product_version"] == "03.01"
    assert back.attrs["acquired"] == dt.datetime(2023, 7, 2, 10, 56, 19)
    assert back.attrs["product_metadata"] == {"datatake": {"id": "GS2A", "orbit": 65}}


def test_variable_and_coord_attrs_round_trip(tmp_path, dataset):
    path = NetCDFWriter().write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert back["B02"].attrs["measurand"] == "toa_reflectance"
        assert back["x"].attrs["standard_name"] == "projection_x_coordinate"


def test_dataset_that_could_not_be_written_unaided(tmp_path, dataset):
    # Without the writer's attribute encoding, this dataset cannot be written at all.
    with pytest.raises(TypeError):
        dataset.to_netcdf(tmp_path / "raw.nc")

    assert NetCDFWriter().write(dataset, tmp_path / "out.nc")


def test_read_multi_style_attrs_round_trip(tmp_path):
    steps = [{"processor": "read_multi", "concat_dim": "time", "n_files": 3}]
    ds = xr.Dataset(
        {"reflectance": (("time", "band"), np.zeros((3, 2), dtype="float32"))},
        coords={"time": [0, 1, 2], "band": ["B02", "B03"]},
        attrs={"eoio:processing_steps": steps, "product_name": ["a.SAFE", "b.SAFE", "c.SAFE"]},
    )

    path = NetCDFWriter().write(ds, tmp_path / "multi.nc")
    with xr.open_dataset(path) as opened:
        back = decode_attrs(opened)

    assert back.attrs["eoio:processing_steps"][0]["processor"] == "read_multi"
    assert back.attrs["product_name"] == ["a.SAFE", "b.SAFE", "c.SAFE"]


def test_provenance_is_in_the_written_file(tmp_path, dataset):
    path = NetCDFWriter().write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as opened:
        back = decode_attrs(opened)

    assert "written to out.nc" in back.attrs["history"]
    assert back.attrs["eoio:processing_steps"][-1] == {
        "processor": "write",
        "writer": "netcdf",
        "path": "out.nc",
    }


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def test_compression_is_applied_by_default(tmp_path, big_dataset):
    path = NetCDFWriter().write(big_dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert back["v"].encoding["zlib"] is True
        assert back["v"].encoding["complevel"] == 4


def test_small_variables_are_not_compressed_by_default(tmp_path, dataset):
    # 400 float32 values: compressing costs more in chunk metadata than it can save.
    path = NetCDFWriter().write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert not back["B02"].encoding.get("zlib", False)


def test_compress_min_bytes_sets_the_cutoff(tmp_path, dataset):
    size = dataset["B02"].nbytes
    at = NetCDFWriter(write_params={"compress_min_bytes": size}).write(dataset, tmp_path / "at.nc")
    above = NetCDFWriter(write_params={"compress_min_bytes": size + 1}).write(dataset, tmp_path / "above.nc")

    with xr.open_dataset(at) as back:
        assert back["B02"].encoding["zlib"] is True
    with xr.open_dataset(above) as back:
        assert not back["B02"].encoding.get("zlib", False)


def test_many_small_variables_do_not_make_the_file_larger(tmp_path):
    # Shaped like RadCalNet: many short series. Compressing each used to add ~2 kB apiece.
    ds = xr.Dataset({f"v{i}": ("x", np.linspace(0, 1, 13) * i) for i in range(20)}, coords={"x": np.arange(13)})
    compressed = NetCDFWriter().write(ds, tmp_path / "c.nc")
    plain = NetCDFWriter(write_params={"compress": False}).write(ds, tmp_path / "p.nc")

    assert (tmp_path / "c.nc").stat().st_size <= (tmp_path / "p.nc").stat().st_size
    assert compressed and plain


def test_compression_mechanics_on_a_small_dataset(tmp_path, dataset):
    path = NetCDFWriter(write_params={"compress_min_bytes": 0}).write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert back["B02"].encoding["zlib"] is True
        assert back["B02"].encoding["complevel"] == 4


def test_compression_actually_shrinks_the_file(tmp_path, big_dataset):
    compressed = NetCDFWriter().write(big_dataset, tmp_path / "c.nc")
    plain = NetCDFWriter(write_params={"compress": False}).write(big_dataset, tmp_path / "p.nc")

    assert (tmp_path / "c.nc").stat().st_size < (tmp_path / "p.nc").stat().st_size / 10
    assert compressed and plain


def test_compression_can_be_disabled(tmp_path, dataset):
    path = NetCDFWriter(write_params={"compress": False}).write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert not back["B02"].encoding.get("zlib", False)


def test_complevel_is_honoured(tmp_path, dataset):
    path = NetCDFWriter(write_params={"complevel": 9, "compress_min_bytes": 0}).write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert back["B02"].encoding["complevel"] == 9


def test_scalars_are_not_compressed(tmp_path, dataset):
    path = NetCDFWriter(write_params={"compress_min_bytes": 0}).write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert not back["quality"].encoding.get("zlib", False)


def test_explicit_encoding_replaces_the_defaults(tmp_path, dataset):
    writer = NetCDFWriter(write_params={"encoding": {"B02": {"zlib": True, "complevel": 1}}})
    path = writer.write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert back["B02"].encoding["complevel"] == 1
        # Only what the caller asked for is applied.
        assert not back["x"].encoding.get("zlib", False)


def test_coordinates_are_compressed_too(tmp_path, dataset):
    # 2-D geolocation coordinates are among the largest arrays in some products.
    path = NetCDFWriter(write_params={"compress_min_bytes": 0}).write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert back["x"].encoding["zlib"] is True


@pytest.mark.parametrize(
    "params",
    [
        {"format": "NETCDF3_CLASSIC", "engine": "scipy"},
        {"engine": "scipy", "format": None},
    ],
)
def test_no_compression_defaults_for_formats_that_cannot_express_them(dataset, params):
    writer = NetCDFWriter(write_params=params)
    assert writer._resolve_encoding(dataset) is None


def test_netcdf3_write_succeeds(tmp_path):
    # The defaults must not make an otherwise-valid NETCDF3 write fail.
    ds = xr.Dataset({"v": ("x", np.arange(10.0))}, coords={"x": np.arange(10)}, attrs={"title": "t"})
    writer = NetCDFWriter(write_params={"format": "NETCDF3_CLASSIC", "engine": "scipy"})
    path = writer.write(ds, tmp_path / "v3.nc")

    with xr.open_dataset(path) as back:
        xr.testing.assert_allclose(back["v"], ds["v"])


def test_h5netcdf_engine_is_compressed(tmp_path, big_dataset):
    pytest.importorskip("h5netcdf")
    plain = NetCDFWriter(write_params={"engine": "h5netcdf", "compress": False}).write(big_dataset, tmp_path / "p.nc")
    packed = NetCDFWriter(write_params={"engine": "h5netcdf"}).write(big_dataset, tmp_path / "c.nc")

    assert Path(packed).stat().st_size < Path(plain).stat().st_size


def test_default_engine_falls_back_to_h5netcdf_when_netcdf4_cannot_import(monkeypatch):
    pytest.importorskip("h5netcdf")
    monkeypatch.setitem(sys.modules, "netCDF4", None)  # makes `import netCDF4` raise ImportError
    _default_engine.cache_clear()
    try:
        assert _default_engine() == "h5netcdf"
        assert NetCDFWriter()._engine() == "h5netcdf"
    finally:
        _default_engine.cache_clear()


def test_requested_engine_is_never_overridden(monkeypatch):
    monkeypatch.setitem(sys.modules, "netCDF4", None)
    _default_engine.cache_clear()
    try:
        assert NetCDFWriter(write_params={"engine": "scipy", "format": "NETCDF3_CLASSIC"})._engine() == "scipy"
    finally:
        _default_engine.cache_clear()


def test_other_formats_are_left_to_xarray_when_no_engine_requested():
    assert NetCDFWriter(write_params={"format": "NETCDF3_CLASSIC"})._engine() is None


# ---------------------------------------------------------------------------
# Write parameters
# ---------------------------------------------------------------------------


def test_group_is_honoured(tmp_path, dataset):
    path = NetCDFWriter(write_params={"group": "measurements"}).write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path, group="measurements") as back:
        assert "B02" in back


def test_unlimited_dims_is_honoured(tmp_path, dataset):
    path = NetCDFWriter(write_params={"unlimited_dims": ["y"]}).write(dataset, tmp_path / "out.nc")

    with xr.open_dataset(path) as back:
        assert back["B02"].sizes["y"] == 20


def test_unknown_write_param_raises():
    with pytest.raises(ValueError, match="Unknown writing parameter"):
        NetCDFWriter(write_params={"compresion": True})


# ---------------------------------------------------------------------------
# DataTree
# ---------------------------------------------------------------------------


@pytest.fixture
def tree():
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"campaign": {"name": "test", "year": 2023}}),
            "/s2": xr.Dataset(
                {"v": ("x", np.arange(100.0))},
                coords={"x": np.arange(100)},
                attrs={"acquired": dt.datetime(2023, 7, 2, 10, 0)},
            ),
            "/l8": xr.Dataset({"w": ("y", np.arange(50.0))}, coords={"y": np.arange(50)}),
        }
    )


def test_datatree_writes_every_node(tmp_path, tree):
    path = NetCDFWriter().write(tree, tmp_path / "tree.nc")

    with xr.open_datatree(path) as back:
        assert sorted(node.path for node in back.subtree) == ["/", "/l8", "/s2"]
        xr.testing.assert_allclose(back["s2"]["v"], tree["s2"]["v"])


def test_datatree_attrs_round_trip(tmp_path, tree):
    path = NetCDFWriter().write(tree, tmp_path / "tree.nc")

    with xr.open_datatree(path) as opened:
        back = decode_attrs(opened)

    assert back.attrs["campaign"] == {"name": "test", "year": 2023}
    assert back["s2"].attrs["acquired"] == dt.datetime(2023, 7, 2, 10, 0)


def test_datatree_nodes_are_compressed(tmp_path, tree):
    path = NetCDFWriter(write_params={"compress_min_bytes": 0}).write(tree, tmp_path / "tree.nc")

    with xr.open_datatree(path) as back:
        assert back["s2"]["v"].encoding["zlib"] is True
        assert back["l8"]["w"].encoding["zlib"] is True


def test_empty_datatree_node_does_not_break_encoding(tmp_path):
    # The root here owns no variables, so it contributes no encoding entries.
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "/child": xr.Dataset({"v": ("x", [1.0, 2.0])})})
    path = NetCDFWriter().write(tree, tmp_path / "tree.nc")

    with xr.open_datatree(path) as back:
        assert "v" in back["child"]


def test_datatree_unlimited_dims_list_applies_to_every_group_containing_them(tmp_path, tree):
    path = NetCDFWriter(write_params={"unlimited_dims": ["x"]}).write(tree, tmp_path / "tree.nc")

    with xr.open_datatree(path) as back:
        assert back["s2"].to_dataset().encoding["unlimited_dims"] == {"x"}
        assert not back["l8"].to_dataset().encoding.get("unlimited_dims")


def test_datatree_unlimited_dims_accepts_a_single_name(tmp_path, tree):
    path = NetCDFWriter(write_params={"unlimited_dims": "y"}).write(tree, tmp_path / "tree.nc")

    with xr.open_datatree(path) as back:
        assert back["l8"].to_dataset().encoding["unlimited_dims"] == {"y"}


def test_datatree_unlimited_dims_mapping_is_passed_through(tmp_path, tree):
    path = NetCDFWriter(write_params={"unlimited_dims": {"/s2": ["x"]}}).write(tree, tmp_path / "tree.nc")

    with xr.open_datatree(path) as back:
        assert back["s2"].to_dataset().encoding["unlimited_dims"] == {"x"}


def test_group_is_refused_for_a_datatree_with_a_reason(tmp_path, tree):
    writer = NetCDFWriter(write_params={"group": "measurements"})

    assert not writer.can_write(tree).ok
    with pytest.raises(TypeError, match="root group"):
        writer.write(tree, tmp_path / "tree.nc")
