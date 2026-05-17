import unittest

from eoio.readers.subset.roi_subset import (
    ROISubsetResolver,
    ResolvedROISubset,
    RasterSubsetError,
)


# Patch lazy_shapely for tests
def _fake_lazy_shapely_real():
    from shapely.geometry import Polygon, box, mapping, shape
    from shapely.ops import transform as shp_transform

    return Polygon, box, mapping, shape, shp_transform


class TestROISubsetResolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        import eoio.deps as mod

        mod.lazy_shapely = _fake_lazy_shapely_real

    def test_bbox_normalisation_and_run_success(self):
        roi = (0.0, 0.0, 10.0, 10.0)
        image_bounds = (-5.0, -5.0, 20.0, 20.0)
        resolver = ROISubsetResolver(roi, "EPSG:4326", "EPSG:4326", image_bounds)
        result = resolver.run()
        self.assertIsInstance(result, ResolvedROISubset)
        self.assertEqual(result.clip_box, roi)
        self.assertEqual(result.geometries[0]["type"], "Polygon")

    def test_point_box_in_geographic_crs_produces_polygon(self):
        resolver = ROISubsetResolver(((0.0, 0.0), 1000.0), "EPSG:4326", "EPSG:4326", (-1.0, -1.0, 1.0, 1.0))
        geom = resolver._normalise_roi()
        self.assertEqual(geom.geom_type, "Polygon")
        result = resolver.run()
        self.assertIsNotNone(result.clip_box)

    def test_run_raises_when_no_intersection(self):
        roi = (1000.0, 1000.0, 2000.0, 2000.0)
        image_bounds = (0.0, 0.0, 10.0, 10.0)
        resolver = ROISubsetResolver(roi, "EPSG:3857", "EPSG:3857", image_bounds)
        with self.assertRaises(RasterSubsetError):
            resolver.run()

    def test_bbox_with_invalid_order_raises(self):
        # xmax <= xmin and ymax <= ymin should be rejected
        with self.assertRaises(RasterSubsetError):
            ROISubsetResolver((10, 10, 5, 5), "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))

    def test_pointbox_negative_halfwidth_raises(self):
        with self.assertRaises(RasterSubsetError):
            ROISubsetResolver(((0, 0), -100), "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))

    def test_invalid_crs_formats_raise(self):
        # roi_crs_epsg missing EPSG:
        with self.assertRaises(RasterSubsetError):
            ROISubsetResolver((0, 0, 1, 1), "4326", "EPSG:3857", (0, 0, 10, 10))
        # roi_crs_epsg integer with invalid digit count
        with self.assertRaises(RasterSubsetError):
            ROISubsetResolver((0, 0, 1, 1), 99, "EPSG:3857", (0, 0, 10, 10))
        # image_crs_epsg missing EPSG:
        with self.assertRaises(RasterSubsetError):
            ROISubsetResolver((0, 0, 1, 1), "EPSG:4326", "3857", (0, 0, 10, 10))

    def test_transform_between_crs_changes_bounds_scale(self):
        roi = (-0.01, -0.01, 0.01, 0.01)
        resolver = ROISubsetResolver(
            roi,
            "EPSG:4326",
            "EPSG:3857",
            (-20037508.34, -20037508.34, 20037508.34, 20037508.34),
        )
        res = resolver.run()
        xmin, ymin, xmax, ymax = res.clip_box
        self.assertTrue(max(abs(x) for x in (xmin, ymin, xmax, ymax)) > 1000.0)

    def test_list_of_coords_polygon_is_accepted(self):
        roi = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        resolver = ROISubsetResolver(roi, "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))
        result = resolver.run()
        self.assertEqual(result.geometries[0]["type"], "Polygon")

    def test_raw_point_is_rejected(self):
        with self.assertRaises(RasterSubsetError):
            ROISubsetResolver((1.0, 2.0), "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))

    def test_image_bounds_tuple_is_converted_to_shapely_in_run(self):
        resolver = ROISubsetResolver((0, 0, 1, 1), "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))
        res = resolver.run()
        self.assertEqual(res.clip_box, (0.0, 0.0, 1.0, 1.0))

    def test_geom_to_geometries_returns_geojson_mapping(self):
        resolver = ROISubsetResolver((0, 0, 1, 1), "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))
        geom = resolver._normalise_roi()
        gs = resolver._geom_to_geometries(geom)
        self.assertIsInstance(gs, list)
        self.assertEqual(gs[0]["type"], "Polygon")

    def test_empty_list_of_coords_raises(self):
        with self.assertRaises(RasterSubsetError):
            ROISubsetResolver([], "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))

    # Intersection logic
    def test_roi_touching_image_bounds_is_accepted(self):
        # ROI equals image bounds -> intersects (boundary counts)
        roi = (0, 0, 10, 10)
        resolver = ROISubsetResolver(roi, "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))
        result = resolver.run()
        self.assertEqual(result.clip_box, (0.0, 0.0, 10.0, 10.0))

    # CRS transformation correctness
    def test_crs_transform_values_are_reasonable(self):
        roi = (-1, -1, 1, 1)  # degrees
        resolver = ROISubsetResolver(
            roi,
            "EPSG:4326",
            "EPSG:3857",
            (-20037508.34, -20037508.34, 20037508.34, 20037508.34),
        )
        result = resolver.run()
        xmin, ymin, xmax, ymax = result.clip_box
        # Should be in meters and span both sides of origin
        self.assertTrue(xmin < 0 < xmax)
        self.assertTrue(abs(xmax) > 100000)

    # GeoJSON and list-of-coords acceptance
    def test_geojson_dict_is_accepted_and_normalized(self):
        roi = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
        resolver = ROISubsetResolver(roi, "EPSG:4326", "EPSG:4326", (0, 0, 10, 10))
        result = resolver.run()
        self.assertEqual(result.geometries[0]["type"], "Polygon")


if __name__ == "__main__":
    unittest.main()
