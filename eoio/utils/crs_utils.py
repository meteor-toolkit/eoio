"""eoio.utils.crs_utils - coordinate reference system utility functions"""

from typing import Tuple, Union

import numpy as np
import scipy
from pyproj import Transformer
from shapely.geometry import Polygon

__author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

__all__ = [
    "get_nearest_lon_lat_coords",
    "convert_xy",
    "coord_bounds",
    "get_xy_arrays",
    "y_on_line",
]


# todo - unit testing
def get_nearest_lon_lat_coords(lon_array, lat_array, coords) -> list:
    """
    Return the new roi clipping coordinates for input longitude and latitude grids using nearest neighbour algorithms

    (function currently rounds down to the first closest index, so treats values on either edge of the roi differently if
    consists of duplicated values)

    :param lon_array: input longitude array
    :param lat_array: input latitude array
    :param coords: coordinates in longitude and latitude for which to find the new x, y coordinates for
    :returns: List of coordinates in x and y that satisfy the input.
    """
    # check input longitude and latitude
    if not lon_array.shape == lat_array.shape:
        raise ValueError(
            """
        Longitude and latitude coordinate grids are different shapes, '{}' and '{}' respectively. Please check inputs.""".format(
                lon_array.shape, lat_array.shape
            )
        )

    # get the grid shape (used to reshape the output index)
    geo_shape = lon_array.shape

    # check input coords
    if not isinstance(coords, list):
        raise ValueError("Please input coordinates in list format.")
    else:
        if not all([isinstance(i, (list, tuple)) for i in coords]):
            if len(coords) == 2 and all([isinstance(i, (float, int)) for i in coords]):
                coords = [coords]
            else:
                raise ValueError("Please input coordinates in nested list format.")
        elif all([all([isinstance(j, list) for j in i]) for i in coords]):
            coords = coords[0]

    # get lon lat grid to use to make the kd-tree
    lon_lat_grid = np.c_[lon_array.ravel(), lat_array.ravel()]

    # make the tree
    tree = scipy.spatial.cKDTree(lon_lat_grid)

    # create new empty coords to fill
    new_coords = []

    # query the grid
    for i in coords:
        distance, indices = tree.query([j for j in i], k=4)

        # if midway between multiple different points (max taken into account = 4)
        if any([j == distance[0] for j in distance[1:]]):
            all_idx = [i[1] for i in zip(distance, indices) if i[0] == distance[0]]
            all_idxs = [[(i - (i % geo_shape[1])) / geo_shape[1], i % geo_shape[1]] for i in all_idx]
            idxs = [np.mean(i) for i in list(zip(*all_idxs))][::-1]
        else:
            idx = indices[0]
            idxs = [(idx - (idx % geo_shape[1])) / geo_shape[1], idx % geo_shape[1]][::-1]

        new_coords.append(idxs)

    return new_coords


def convert_xy(
    im_input: Union[list, tuple, np.ndarray],
    input_format: str,
    output_format: str = "EPSG:4326",
):
    """
    Convert input coordinates from their coordinate reference system to the World Geodetic System 1984 (WGS 84) or
    other specified coordinate reference system.

    If im_input is a list/1D- array of x and y coordinates instead of a two dimensional coordinate grid, list/array
    first used to create a two dimensional coordinate grid

    :param im_input: list/1D-array of coordinate values, or 2D-coordinate grid
    :param input_format: coordinate reference system of input coordinates
    :param output_format: desired coordinate reference system (defaults to 'EPSG:4326')
    :returns: Tuple of converted x and y coordinates.
    """
    if isinstance(im_input, tuple):
        im_input = list(im_input)
    try:
        im_input[0].shape
    except AttributeError:
        im_input[0] = np.array(im_input[0])
        im_input[1] = np.array(im_input[1])

    if im_input[0].shape == im_input[1].shape and im_input[0].ndim == 2:
        x_mesh = im_input[0]
        y_mesh = im_input[1]
    else:
        x_mesh, y_mesh = np.meshgrid(im_input[0], im_input[1])
    transformer = Transformer.from_crs(input_format, output_format, always_xy=True)
    out_x, out_y = transformer.transform(x_mesh, y_mesh)
    return out_x, out_y


def coord_bounds(
    x_step: int,
    y_step: int,
    x_origin: int,
    y_origin: int,
    x_len: int,
    y_len: int,
    crs: dict,
) -> dict:
    """
    Return coordinate bounds of the data

    :param x_step:
    :param y_step:
    :param x_origin:
    :param y_origin:
    :param x_len:
    :param y_len:
    :param crs: dictionary of crs info of form {"epsg": str, "crs": pyproj.CRS()}
    :returns: Dictionary of bounding coordinates in both the image default CRS and EPSG:4326.
    """
    x_0, x_1, y_0, y_1 = (
        x_origin,
        x_origin + (x_step * (x_len - 1)),
        y_origin,
        y_origin + (y_step * (y_len - 1)),
    )

    lat_lon_0 = [float(i[0]) for i in convert_xy([x_0, y_0], crs["crs"])]
    lat_lon_1 = [float(i[0]) for i in convert_xy([x_1, y_0], crs["crs"])]
    lat_lon_2 = [float(i[0]) for i in convert_xy([x_1, y_1], crs["crs"])]
    lat_lon_3 = [float(i[0]) for i in convert_xy([x_0, y_1], crs["crs"])]

    bounds = {
        "EPSG:4326": Polygon([lat_lon_0, lat_lon_1, lat_lon_2, lat_lon_3]),
        crs["epsg"]: Polygon([[x_0, y_0], [x_1, y_0], [x_1, y_1], [x_0, y_1]]),
    }

    return bounds


def get_xy_arrays(x_step, y_step, x_origin, y_origin, x_len, y_len):
    """
    Return x and y arrays given step size, length of array, and point of origin

    :param x_step: x step size
    :param y_step: y step size
    :param x_origin: position of x origin
    :param y_origin: position of y origin
    :param x_len: number of x points
    :param y_len:  number of y points
    :returns: Tuple of x and y arrays.
    """
    x_array = np.arange(0, x_len) * x_step + x_origin
    y_array = np.arange(0, y_len) * y_step + y_origin

    return x_array, y_array


def y_on_line(
    p1: Tuple[Union[float, int], Union[float, int]],
    p2: Tuple[Union[float, int], Union[float, int]],
    x: Union[float, int],
) -> Union[float, int]:
    """
    Return y on line between two points for a given x

    :param p1: first point on line (x, y)
    :param p2: second point on line (x, y)
    :param x: x value of desired output point
    :returns: Y value on the line defined by ``p1`` and ``p2``.
    """
    x_0, y_0 = p1
    x_1, y_1 = p2
    return ((y_1 - y_0) / (x_1 - x_0)) * (x - x_1) + y_1
