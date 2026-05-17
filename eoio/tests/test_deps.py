import unittest
from unittest.mock import patch
import builtins

from eoio import deps


class TestDeps(unittest.TestCase):
    def test_require_extra_raises_clear_message(self):
        with self.assertRaises(ModuleNotFoundError) as ctx:
            deps._require_extra("raster", "rioxarray")
        msg = str(ctx.exception)
        self.assertIn("rioxarray", msg)
        self.assertIn("eoio[raster]", msg)

    def test_lazy_rasterio_missing_raises(self):
        with patch.object(builtins, "__import__", side_effect=ModuleNotFoundError("rasterio")):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                deps.lazy_rasterio()
        self.assertIn("rasterio", str(ctx.exception))

    def test_lazy_rioxarray_missing_raises(self):
        with patch.object(builtins, "__import__", side_effect=ModuleNotFoundError("rioxarray")):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                deps.lazy_rioxarray()
        self.assertIn("rioxarray", str(ctx.exception))

    def test_lazy_pyproj_missing_raises(self):
        with patch.object(builtins, "__import__", side_effect=ModuleNotFoundError("pyproj")):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                deps.lazy_pyproj()
        self.assertIn("pyproj", str(ctx.exception))

    def test_lazy_shapely_missing_raises(self):
        with patch.object(builtins, "__import__", side_effect=ModuleNotFoundError("shapely")):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                deps.lazy_shapely()
        self.assertIn("shapely", str(ctx.exception))

    def test_lazy_cfgrib_missing_raises(self):
        with patch.object(builtins, "__import__", side_effect=ModuleNotFoundError("cfgrib")):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                deps.lazy_cfgrib()
        self.assertIn("cfgrib", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
