"""Tests for eoio.processors.add_lat_lon.processor"""

import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import numpy as np
import xarray as xr

from eoio.processors.add_lat_lon.processor import AddLatLon


def _make_mock_pyproj():
    mock_pyproj = MagicMock()

    # CRS.from_epsg should return something (value not used beyond being passed)
    mock_pyproj.CRS.from_epsg.return_value = "EPSG:4326"

    # Transformer.from_crs returns an object with a transform method
    transformer = MagicMock()

    # transformer.transform should return (lons, lats) given (x, y)
    def transform(x, y):
        # we return x, y so tests can easily assert shapes/values
        return x, y

    transformer.transform.side_effect = transform
    mock_pyproj.Transformer.from_crs.return_value = transformer

    return mock_pyproj


class TestAddLatLonProcessor(unittest.TestCase):
    def test_parse_params_invalid_on_missing_raises(self):
        with self.assertRaises(ValueError):
            AddLatLon(params={"on_missing": "bad"})

    def test_format_geometry_id_returns_list_when_none(self):
        inst = AddLatLon(params={})
        ds = xr.Dataset(coords={"x_10m": [1, 2], "y_10m": [3, 4], "x_60m": [1], "y_60m": [2]})
        geoms = inst._format_geometry_id(ds, None)
        self.assertIsInstance(geoms, list)

    def test_format_geometry_id_contains_expected(self):
        inst = AddLatLon(params={})
        ds = xr.Dataset(coords={"x_10m": [1, 2], "y_10m": [3, 4], "x_60m": [1], "y_60m": [2]})
        geoms = inst._format_geometry_id(ds, None)
        self.assertIn("10m", geoms)

    def test_format_geometry_id_invalid_raises(self):
        inst = AddLatLon(params={})
        ds = xr.Dataset(coords={"x_10m": [1], "y_10m": [2]})
        with self.assertRaises(ValueError):
            inst._format_geometry_id(ds, ["DOES_NOT_EXIST"])

    def test_run_adds_lat_lon_coords_presence(self):
        # Build a small dataset with x_10m and y_10m coords and a data variable
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([0.0, 1.0])
        vals = np.zeros((y.size, x.size))
        ds = xr.Dataset({"val": (("y_10m", "x_10m"), vals)}, coords={"x_10m": x, "y_10m": y})

        inst = AddLatLon(params={"geometry_id": ["10m"]})

        # patch xr.Dataset to provide a rio accessor for the duration of the call
        with patch.object(
            xr.Dataset,
            "rio",
            property(lambda self: SimpleNamespace(crs="EPSG:3857")),
            create=True,
        ):
            with patch(
                "eoio.processors.add_lat_lon.processor.lazy_pyproj",
                return_value=_make_mock_pyproj(),
            ):
                out = inst.run(ds)

        # coordinates should have been added
        self.assertIn("latitude_10m", out.coords)

    def test_run_adds_lat_lon_coords_shape(self):
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([0.0, 1.0])
        vals = np.zeros((y.size, x.size))
        ds = xr.Dataset({"val": (("y_10m", "x_10m"), vals)}, coords={"x_10m": x, "y_10m": y})
        inst = AddLatLon(params={"geometry_id": ["10m"]})
        # patch xr.Dataset to provide a rio accessor for the duration of the call
        with patch.object(
            xr.Dataset,
            "rio",
            property(lambda self: SimpleNamespace(crs="EPSG:3857")),
            create=True,
        ):
            with patch(
                "eoio.processors.add_lat_lon.processor.lazy_pyproj",
                return_value=_make_mock_pyproj(),
            ):
                out = inst.run(ds)

        self.assertEqual(out.coords["latitude_10m"].shape, (y.size, x.size))

    def test_run_adds_lat_lon_for_each_geometry_when_none(self):
        x10 = np.array([1.0, 2.0])
        y10 = np.array([0.0])
        x60 = np.array([5.0])
        y60 = np.array([6.0, 7.0])

        ds = xr.Dataset(
            {"a": (("y_10m", "x_10m"), np.zeros((y10.size, x10.size)))},
            coords={"x_10m": x10, "y_10m": y10, "x_60m": x60, "y_60m": y60},
        )
        inst = AddLatLon(params={})
        with patch.object(
            xr.Dataset,
            "rio",
            property(lambda self: SimpleNamespace(crs="EPSG:3857")),
            create=True,
        ):
            with patch(
                "eoio.processors.add_lat_lon.processor.lazy_pyproj",
                return_value=_make_mock_pyproj(),
            ):
                out = inst.run(ds)

        self.assertIn("latitude_10m", out.coords)

    def test_run_raises_on_non_dataset(self):
        inst = AddLatLon(params={})
        with self.assertRaises(TypeError):
            inst.run("not a dataset")

    def test_record_provenance_appends_geometry_ids(self):
        x = np.array([1.0])
        y = np.array([2.0])
        ds = xr.Dataset(
            {"a": (("y_10m", "x_10m"), np.zeros((1, 1)))},
            coords={"x_10m": x, "y_10m": y},
        )
        inst = AddLatLon(params={"geometry_id": ["10m"]})
        with patch.object(
            xr.Dataset,
            "rio",
            property(lambda self: SimpleNamespace(crs="EPSG:3857")),
            create=True,
        ):
            with patch(
                "eoio.processors.add_lat_lon.processor.lazy_pyproj",
                return_value=_make_mock_pyproj(),
            ):
                out = inst.run(ds)

        steps = out.attrs.get("eoio:processing_steps")
        self.assertIsNotNone(steps)
        self.assertIsInstance(steps, list)
        last = steps[-1]
        self.assertEqual(last.get("processor"), "add_lat_lon")
        self.assertIn("10m", last.get("geometry_ids"))


if __name__ == "__main__":
    unittest.main()
