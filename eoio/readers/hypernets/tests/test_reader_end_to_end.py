"""eoio.readers.tests.test_hypernets - test for eoio.readers.hypernets"""

import os.path
import unittest

import numpy.testing as npt

from eoio import read, product_options
import configparser
import os

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "etc",
    "test_file_paths.config",
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# Assume config has a section [hypernets] and key 'l1b_file'
file_path_l1b = config.get("HYPERNETS", "l1b_file", fallback=None)
file_path_l2a = config.get("HYPERNETS", "l2a_file", fallback=None)


__author__ = "Pieter De Vis <pieter.de.vis@npl.co.uk>"
__all__ = []


eoio_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
test_data_path = os.path.join(eoio_folder, "test_data")
read_params = None


class TestHypernetsReader(unittest.TestCase):
    def test_product_options(self):
        if not os.path.exists(file_path_l2a):
            self.skipTest(f"Test .input data not found at {file_path_l2a}")
        assert len(product_options(file_path_l2a, read_params).keys()) > 0

    def test_read_L2A_subset_masks(self):
        if not os.path.exists(file_path_l2a):
            self.skipTest(f"Test .input data not found at {file_path_l2a}")
        example_data_path = file_path_l2a

        subset_dict = {"mask": True}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 43)

        subset_dict = {"mask": False}

        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 44)

        subset_dict = {"mask": "no_clear_sky"}

        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 44)

        subset_dict = {"mask": ["saturation"]}

        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 43)

        subset_dict = {"mask": ["saturation", "outliers"]}

        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 43)

    def test_read_L2A_subset_metadata(self):
        if not os.path.exists(file_path_l2a):
            self.skipTest(f"Test .input data not found at {file_path_l2a}")
        example_data_path = file_path_l2a

        read_params = {"metadata_level": "all"}
        data = read(example_data_path, read_params=read_params)
        self.assertGreater(len(data.attrs.keys()), 0)

        read_params["metadata_level"] = "basic"
        data = read(example_data_path, read_params=read_params)
        self.assertGreater(len(data.attrs.keys()), 0)

        read_params["metadata_level"] = None
        data = read(example_data_path, read_params=read_params)
        self.assertEqual(len(data.attrs.keys()), 0)

    def test_read_L2A_subset_angle(self):
        if not os.path.exists(file_path_l2a):
            self.skipTest(f"Test .input data not found at {file_path_l2a}")
        example_data_path = file_path_l2a

        subset_dict = {"angle": {"vza": {"min": 35}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 18)

        example_data_path = file_path_l2a

        subset_dict = {"angle": {"vza": {"min": 35, "max": 45}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 6)

        subset_dict = {"angle": {"vza": {"min": 35, "max": 45}, "vaa": {"min": 0, "max": 90}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 1)

        subset_dict = {"angle": {"vza": {"min": 35, "max": 45}, "vaa": {"min": -90, "max": 90}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 3)

        subset_dict = {"angle": {"vza": {"min": -35, "max": 45}}}
        npt.assert_raises(ValueError, read, example_data_path, subset_dict, read_params=read_params)

        subset_dict = {"angle": {"vza": {"nearest": 10}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 6)

        subset_dict = {"angle": {"vza": {"nearest": 10}, "vaa": {"nearest": 83}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 1)

        subset_dict = {
            "angle": {
                "vza": {"nearest": 10, "tolerance": 7},
                "vaa": {"nearest": 83, "tolerance": 15},
            }
        }
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 4)

        subset_dict = {
            "angle": {"vza": {"nearest": 10}, "vaa": {"nearest": 83}},
            "wavelength": {"min": 500, "max": 550},
        }
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(data["reflectance"].values.shape, (103, 1))
        npt.assert_equal(data["u_rel_random_reflectance"].values.shape, (103, 1))
        npt.assert_equal(data["err_corr_systematic_reflectance"].values.shape, (103, 103))

    def test_read_L1B(self):
        if not os.path.exists(file_path_l1b):
            self.skipTest(f"Test .input data not found at {file_path_l1b}")
        subset_dict = {"angle": {"vza": {"min": 35}}}
        example_data_path = file_path_l1b

        from importlib.metadata import version

        print("numpy version:", version("numpy"))
        print("xarray version:", version("xarray"))

        data = read(example_data_path, subset=subset_dict, read_params=read_params)
        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 18)

        subset_dict = {"angle": {"vza": {"min": 35, "max": 45}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 6)

        subset_dict = {"angle": {"vza": {"nearest": 10}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 6)

        subset_dict = {"angle": {"vza": {"nearest": 10}, "vaa": {"nearest": 83}}}
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 1)

        subset_dict = {
            "angle": {
                "vza": {"nearest": 10, "tolerance": 7},
                "vaa": {"nearest": 83, "tolerance": 15},
            }
        }
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(len(data["viewing_azimuth_angle"].values), 4)

        subset_dict = {
            "angle": {"vza": {"nearest": 10}, "vaa": {"nearest": 83}},
            "wavelength": {"min": 500, "max": 550},
        }
        data = read(example_data_path, subset=subset_dict, read_params=read_params)

        npt.assert_equal(data["radiance"].values.shape, (103, 1))
        npt.assert_equal(data["u_rel_random_radiance"].values.shape, (103, 1))
        npt.assert_equal(data["err_corr_systematic_indep_radiance"].values.shape, (103, 103))


if __name__ == "__main__":
    unittest.main()
