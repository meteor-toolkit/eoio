"""eoio.readers.modis.tests.test_layout - unit tests for eoio.readers.modis.layout"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from eoio.readers.modis.layout import MODISLayout, MODISLayoutError


class TestMODISLayout(unittest.TestCase):
    def _make_modis_product(
        self,
        root: Path,
        *,
        product_name: str = "MOD02HDF.A2020001.0000.061.2020002121530.hdf",
        with_geolocation: bool = False,
        geolocation_name: str = "MOD03.A2020001.0000.061.2020002121530.hdf",
    ) -> Path:
        """
        Create a minimal MODIS product file structure for testing.

        :param root: Root directory for the product
        :param product_name: Name of the MODIS product file
        :param with_geolocation: Whether to create a geolocation file
        :param geolocation_name: Name of the geolocation product
        :return: Path to the product file
        """
        product_file = root / product_name
        product_file.write_text("hdf", encoding="utf-8")

        if with_geolocation:
            geolocation_file = root / geolocation_name
            geolocation_file.write_text("hdf", encoding="utf-8")

        return product_file

    def test_init_raises_if_path_missing(self):
        """Test that MODISLayout raises an error if the product path doesn't exist."""
        with TemporaryDirectory() as td:
            missing = Path(td) / "missing.hdf"
            with self.assertRaises(MODISLayoutError):
                MODISLayout(str(missing))

    def test_modis_dir_returns_parent_directory(self):
        """Test that modis_dir returns the parent directory of the product."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td))
            layout = MODISLayout(str(p))
            self.assertEqual(layout.modis_dir, Path(td))

    def test_modis_platform_returns_platform_code(self):
        """Test that modis_platform extracts the correct platform code."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.modis_platform, "MOD")

    def test_modis_platform_returns_myd_code(self):
        """Test that modis_platform extracts MYD platform code."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MYD02HDF.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.modis_platform, "MYD")

    def test_processing_level_returns_L1B_for_MOD02(self):
        """Test that processing_level correctly identifies L1B MOD02 products."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.processing_level, "L1B")

    def test_processing_level_returns_L1B_for_MYD02(self):
        """Test that processing_level correctly identifies L1B MYD02 products."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MYD02HDF.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.processing_level, "L1B")

    def test_processing_level_returns_L2_for_MOD09(self):
        """Test that processing_level correctly identifies L2 MOD09 products."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD09GA.A2020001.h00v00.061.2020003203106.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.processing_level, "L2")

    def test_processing_level_returns_L2_for_MYD09(self):
        """Test that processing_level correctly identifies L2 MYD09 products."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MYD09GA.A2020001.h00v00.061.2020003203106.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.processing_level, "L2")

    def test_processing_level_raises_for_unsupported_product(self):
        """Test that processing_level raises an error for unsupported products."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="INVALID.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            with self.assertRaises(MODISLayoutError):
                _ = layout.processing_level

    def test_file_res_key_returns_1_for_1km(self):
        """Test that file_res_key returns '1' for 1km resolution L1B product."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD021KM.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.file_res_key, "1")

    def test_file_res_key_returns_H_for_500m(self):
        """Test that file_res_key returns 'H' for 500m resolution L1B product."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.file_res_key, "H")

    def test_file_res_key_returns_Q_for_250m(self):
        """Test that file_res_key returns 'Q' for 250m resolution L1B product."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD02QKM.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.file_res_key, "Q")

    def test_file_res_key_L2_with_preferred_resolution(self):
        """Test that file_res_key handles L2 products with preferred_resolution."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD09GA.A2020001.h00v00.061.2020003203106.hdf")
            layout = MODISLayout(str(p), preferred_resolution=500)
            self.assertEqual(layout.file_res_key, "H")

    def test_proc_version_extracts_version(self):
        """Test that proc_version correctly extracts the version number."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            self.assertEqual(layout.proc_version, "061")

    def test_default_meas_vars_returns_36_bands_for_1km(self):
        """Test that default_meas_vars returns 36 bands for 1km resolution."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD021KM.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            vars_list = layout.default_meas_vars()
            self.assertEqual(len(vars_list), 36)
            self.assertEqual(vars_list[0], "Band 1")
            self.assertEqual(vars_list[-1], "Band 36")

    def test_default_meas_vars_returns_7_bands_for_500m(self):
        """Test that default_meas_vars returns 7 bands for 500m resolution."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            vars_list = layout.default_meas_vars()
            self.assertEqual(len(vars_list), 7)
            self.assertEqual(vars_list[0], "Band 1")
            self.assertEqual(vars_list[-1], "Band 7")

    def test_default_meas_vars_returns_2_bands_for_250m(self):
        """Test that default_meas_vars returns 2 bands for 250m resolution."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(Path(td), product_name="MOD02QKM.A2020001.0000.061.2020002121530.hdf")
            layout = MODISLayout(str(p))
            vars_list = layout.default_meas_vars()
            self.assertEqual(len(vars_list), 2)
            self.assertEqual(vars_list[0], "Band 1")
            self.assertEqual(vars_list[-1], "Band 2")

    def test_geolocation_path_finds_geolocation_in_same_dir(self):
        """Test that geolocation_path finds MOD03/MYD03 in the same directory."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(
                Path(td),
                product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf",
                with_geolocation=True,
                geolocation_name="MOD03.A2020001.0000.061.2020002121530.hdf",
            )
            layout = MODISLayout(str(p))
            geo_path = layout.geolocation_path()
            self.assertTrue(geo_path.exists())
            self.assertIn("MOD03", geo_path.name)

    def test_geolocation_path_raises_if_missing(self):
        """Test that geolocation_path raises an error if MOD03/MYD03 is not found."""
        with TemporaryDirectory() as td:
            p = self._make_modis_product(
                Path(td),
                product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf",
                with_geolocation=False,
            )
            layout = MODISLayout(str(p))
            with self.assertRaises(MODISLayoutError):
                layout.geolocation_path()

    def test_geolocation_path_uses_custom_geolocation_dir(self):
        """Test that geolocation_path searches in custom directory when provided."""
        with TemporaryDirectory() as td:
            td_path = Path(td)

            # Create product in one directory
            prod_dir = td_path / "products"
            prod_dir.mkdir()
            p = self._make_modis_product(
                prod_dir,
                product_name="MOD02HDF.A2020001.0000.061.2020002121530.hdf",
                with_geolocation=False,
            )

            # Create geolocation in another directory
            geo_dir = td_path / "geolocation"
            geo_dir.mkdir()
            self._make_modis_product(
                geo_dir,
                product_name="MOD03.A2020001.0000.061.2020002121530.hdf",
                with_geolocation=False,
            )

            layout = MODISLayout(str(p), geolocation_dir=str(geo_dir))
            geo_path = layout.geolocation_path()
            self.assertTrue(geo_path.exists())
            self.assertIn("MOD03", geo_path.name)


if __name__ == "__main__":
    unittest.main()
