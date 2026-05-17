"""eoio.utils.tests.test_crs_utils - test for eoio.utils.crs_utils"""

import unittest
import unittest.mock as mock

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

from eoio.utils.crs_utils import coord_bounds, convert_xy, get_nearest_lon_lat_coords, get_xy_arrays, y_on_line

__author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

__all__ = []


class TestCRSUtils(unittest.TestCase):
    def test_get_nearest_lon_lat_coords_point(self):
        input_lon = np.array([[1, 1.5, 2, 2.5], [2, 2.5, 3, 3.5], [3, 3.5, 4, 4.5], [4, 4.5, 5, 5.5]])
        input_lat = np.array([[4, 3, 2, 1], [4.5, 3.5, 2.5, 1.5], [5, 4, 3, 2], [5.5, 4.5, 3.5, 2.5]])
        input_coords = [1, 4]
        output_coords = [[0, 0]]

        self.assertEqual(
            get_nearest_lon_lat_coords(input_lon, input_lat, input_coords),
            output_coords,
        )

    def test_get_nearest_lon_lat_coords_list(self):
        input_lon = np.array([[1, 1.5, 2, 2.5], [2, 2.5, 3, 3.5], [3, 3.5, 4, 4.5], [4, 4.5, 5, 5.5]])
        input_lat = np.array([[4, 3, 2, 1], [4.5, 3.5, 2.5, 1.5], [5, 4, 3, 2], [5.5, 4.5, 3.5, 2.5]])
        input_coords = [[1, 4], [2, 2]]
        output_coords = [[0, 0], [2, 0]]

        self.assertEqual(
            get_nearest_lon_lat_coords(input_lon, input_lat, input_coords),
            output_coords,
        )

    def test_get_nearest_lon_lat_coords_nested_list(self):
        input_lon = np.array([[1, 1.5, 2, 2.5], [2, 2.5, 3, 3.5], [3, 3.5, 4, 4.5], [4, 4.5, 5, 5.5]])
        input_lat = np.array([[4, 3, 2, 1], [4.5, 3.5, 2.5, 1.5], [5, 4, 3, 2], [5.5, 4.5, 3.5, 2.5]])
        input_coords = [[[1, 4], [2, 2]]]
        output_coords = [[0, 0], [2, 0]]

        self.assertEqual(
            get_nearest_lon_lat_coords(input_lon, input_lat, input_coords),
            output_coords,
        )

    def test_get_nearest_lon_lat_coords_in_between(self):
        input_lon = np.array([[1, 1.5, 2, 2.5], [2, 2.5, 3, 3.5], [3, 3.5, 4, 4.5], [4, 4.5, 5, 5.5]])
        input_lat = np.array([[4, 3, 2, 1], [4.5, 3.5, 2.5, 1.5], [5, 4, 3, 2], [5.5, 4.5, 3.5, 2.5]])
        input_coords = [1.75, 3]
        output_coords = [[1, 0]]

        self.assertEqual(
            get_nearest_lon_lat_coords(input_lon, input_lat, input_coords),
            output_coords,
        )

    def test_get_nearest_lon_lat_coords_middle(self):
        input_lon = np.array([[1, 1.5, 2, 2.5], [2, 2.5, 3, 3.5], [3, 3.5, 4, 4.5], [4, 4.5, 5, 5.5]])
        input_lat = np.array([[4, 3, 2, 1], [4.5, 3.5, 2.5, 1.5], [5, 4, 3, 2], [5.5, 4.5, 3.5, 2.5]])
        input_coords = [1.75, 3.75]
        output_coords = [[0.5, 0.5]]

        self.assertEqual(
            get_nearest_lon_lat_coords(input_lon, input_lat, input_coords),
            output_coords,
        )

    def test_convert_xy(self):
        x_input_1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        y_input_1 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

        x_input_2 = np.array([1, 2, 3, 4])
        y_input_2 = np.array([1, 2, 3, 4])

        with mock.patch(
            "pyproj.Transformer.from_crs",
            return_value=Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True),
        ):
            np.testing.assert_array_equal(
                convert_xy((x_input_1, y_input_1), "input_format")[0],
                x_input_1,
            )
            np.testing.assert_array_equal(
                convert_xy((x_input_1, y_input_1), "input_format")[1],
                y_input_1,
            )

            np.testing.assert_array_equal(
                convert_xy((x_input_2, y_input_2), "input_format")[0],
                x_input_1,
            )
            np.testing.assert_array_equal(
                convert_xy((x_input_2, y_input_2), "input_format")[1],
                y_input_1,
            )

    @mock.patch(
        "eoio.utils.crs_utils.convert_xy",
        side_effect=[[[1], [3]], [[2], [3]], [[2], [4]], [[1], [4]]],
    )
    def test_coord_bounds(self, convert_xy_mock):
        output = {
            "EPSG:4326": Polygon([[1.0, 3.0], [2.0, 3.0], [2.0, 4.0], [1.0, 4.0]]),
            "epsg": Polygon(
                [
                    [664200.0, 3311400.0],
                    [890100.0, 3311400.0],
                    [890100.0, 3081000.0],
                    [664200.0, 3081000.0],
                ]
            ),
        }
        self.assertEqual(
            coord_bounds(30, -30, 664200.0, 3311400.0, 7531, 7681, {"crs": "crs", "epsg": "epsg"}),
            output,
        )

    def test_get_xy_arrays(self):
        output_x = np.array([0, 1, 2, 3, 4])
        output_y = np.array([0, -1, -2, -3, -4])

        output = get_xy_arrays(1, -1, 0, 0, 5, 5)

        np.testing.assert_array_equal(output[0], output_x)
        np.testing.assert_array_equal(output[1], output_y)

    def test_y_on_line_xy(self):
        first_point = (0, 0)
        second_point = (2, 2)
        x = 1
        self.assertEqual(y_on_line(first_point, second_point, x), 1)

    def test_y_on_line(self):
        first_point = (0, 0)
        second_point = (10, 2)
        x = 5
        self.assertEqual(y_on_line(first_point, second_point, x), 1)


if __name__ == "__main__":
    unittest.main()
