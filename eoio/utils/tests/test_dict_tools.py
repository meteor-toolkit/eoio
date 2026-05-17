"""eoio.utils.tests.test_dict_tools - test for eoio.utils.dict_tools"""

import datetime as dt
import unittest
import unittest.mock as mock

from eoio.utils.dict_tools import (
    change_type,
    clean_dict,
    create_parent_key,
    dict_merge,
    empty_dict,
    get_nested_value,
    get_value,
    get_value_gen,
    key_exists,
    key_in_dict,
    key_present,
    list_keys,
    make_value_key,
    multiple_keys_present,
    pop_vals,
    remove_parent_tag,
    remove_tag,
    remove_tag_in_key,
    replace_key_names,
    rmv_empty_dict,
)


class MyDictTools(unittest.TestCase):
    def test_key_exists_any(self):
        input_dict = {"name": "filename", "age": 9}
        with mock.patch("eoio.utils.dict_tools.key_present", return_value=True):
            self.assertEqual(key_exists(input_dict, "name"), True)

    def test_key_exists_all(self):
        input_dict = {"name": "filename", "age": 9}
        with mock.patch("eoio.utils.dict_tools.multiple_keys_present", return_value=True):
            self.assertEqual(key_exists(input_dict, ["name", "age"], "all"), True)

    def test_key_exists_invalid_key(self):
        input_dict = {"name": "filename", "age": 9}
        self.assertRaises(ValueError, key_exists, input_dict, "name", "EVERYTHING")

    def test_key_present(self):
        test_dict = {
            "B": {
                "B1": [
                    {
                        "C1": {"C10": ["c", "c", "c"]},
                        "C3": {"C30": [{"C300": [{"C3000": "ccc"}]}]},
                    }
                ],
                "B2": {
                    "E": {"E1": {"E10": {"E100": [1, 2, 3]}}},
                },
            }
        }
        keys = ["B", "B1", "C1", "E100", "C30", "C300", "C3000"]
        for k in keys:
            self.assertEqual(key_present(test_dict, k), True, k)

        self.assertEqual(key_present(test_dict, keys), True)

        self.assertEqual(
            key_present(test_dict, "A"),
            False,
            "'A' determined to be present when not in dictionary",
        )

    def test_empty_dict(self):
        test_dict = {}
        self.assertEqual(empty_dict(test_dict), True)

        test_dict_2 = {
            "Staff": [
                {"Title": "Mr", "First_Name": None},
                {"Title": "Mx", "First_Name": []},
                {"Title": "Ms", "First_Name": "none"},
            ],
            "Students": {"Class": {}},
            "Rooms": {"Ground_Floor": ["A1", "A2", "B1", "B2"]},
            "Labs": {"Ground_Floor": [{"Science": []}]},
        }
        self.assertEqual(empty_dict(test_dict_2), True)

        for i in range(3):
            self.assertEqual(empty_dict(test_dict_2["Staff"][i]), True)

        self.assertEqual(empty_dict(test_dict_2["Students"]), True)
        self.assertEqual(empty_dict(test_dict_2["Rooms"]), False)
        self.assertEqual(empty_dict(test_dict_2["Labs"]), True)

    def test_rmv_empty_dict(self):
        input_dict = {
            "Staff": [
                {"Title": "Mr", "First_Name": None},
                {"Title": "Mx", "First_Name": []},
                {"Title": "Ms", "First_Name": "none"},
                {"Title": []},
            ],
            "Students": {"Class": {}},
            "Rooms": {"Ground_Floor": ["A1", "A2", "B1", "B2"]},
        }

        output_dict_1 = {
            "Staff": [{"Title": "Mr"}, {"Title": "Mx"}, {"Title": "Ms"}, {}],
            "Students": {},
            "Rooms": {"Ground_Floor": ["A1", "A2", "B1", "B2"]},
        }

        rmv_empty_dict(input_dict)

        self.assertDictEqual(input_dict, output_dict_1)

        output_dict_2 = {
            "Staff": [
                {"Title": "Mr"},
                {"Title": "Mx"},
                {"Title": "Ms"},
            ],
            "Rooms": {"Ground_Floor": ["A1", "A2", "B1", "B2"]},
        }

        rmv_empty_dict(input_dict)

        self.assertDictEqual(input_dict, output_dict_2)

    def test_key_in_dict(self):
        dict_1 = {"Name": "Table", "Location": "Kitchen", "Other": {"Age": 13}}
        dict_2 = {"Name": "Office", "Info": {"Location": "Outside"}}
        self.assertEqual(key_in_dict(dict_1, dict_2), True)
        self.assertEqual(key_in_dict(dict_1, dict_2["Info"]), True)
        self.assertEqual(key_in_dict(dict_1["Other"], dict_2), False)

    def test_dict_merge_no_common_keys(self):
        input_1 = {"A": "a"}
        input_2 = {"B": "b"}

        output_dict = {"A": "a", "B": "b"}

        with mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False):
            self.assertEqual(dict_merge([input_1, input_2]), output_dict)

    def test_dict_merge_common_keys_level_1(self):
        input_1 = {"Ground_Floor": "A1"}
        input_2 = {"Ground_Floor": "C1"}

        output_dict = {"Ground_Floor": ["A1", "C1"]}

        input_3 = {"Ground_Floor": ["A1", "A2", "B1", "B2"]}
        input_4 = {"Ground_Floor": ["C1", "C2", "D1", "D2"]}

        output_dict_1 = {"Ground_Floor": [["A1", "A2", "B1", "B2"], ["C1", "C2", "D1", "D2"]]}

        input_5 = {"Ground_Floor": ["E1", "E2", "F1", "F2"]}

        output_dict_2 = {
            "Ground_Floor": [
                ["A1", "A2", "B1", "B2"],
                ["C1", "C2", "D1", "D2"],
                ["E1", "E2", "F1", "F2"],
            ]
        }

        input_6 = {"Ground_Floor": "G1"}

        output_dict_3 = {"Ground_Floor": ["A1", "A2", "B1", "B2", "G1"]}

        with mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=True):
            self.assertEqual(dict_merge([input_1, input_2]), output_dict)
            self.assertEqual(dict_merge([input_3, input_4]), output_dict_1)
            self.assertEqual(dict_merge([output_dict_1, input_5]), output_dict_2)
            self.assertEqual(dict_merge([input_3, input_6]), output_dict_3)

    def test_dict_merge_multiple_common_keys_level_1(self):
        input_3 = {"Ground_Floor": ["A1", "A2", "B1", "B2"]}
        input_4 = {"Ground_Floor": ["C1", "C2", "D1", "D2"]}
        input_5 = {"Ground_Floor": ["E1", "E2", "F1", "F2"]}

        output_dict_2 = {
            "Ground_Floor": [
                ["A1", "A2", "B1", "B2"],
                ["C1", "C2", "D1", "D2"],
                ["E1", "E2", "F1", "F2"],
            ]
        }

        with mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=True):
            self.assertEqual(dict_merge([input_3, input_4, input_5]), output_dict_2)

    def test_dict_merge_common_keys_level_2(self):
        input_1 = {"Rooms": {"Ground_Floor": "A1"}}
        input_2 = {"Rooms": {"Ground_Floor": "A2"}}

        output_dict = {"Rooms": {"Ground_Floor": ["A1", "A2"]}}

        input_3 = {"Rooms": {"Ground_Floor": ["A1", "A2", "B1", "B2"]}}
        input_4 = {"Rooms": {"Ground_Floor": ["C1", "C2", "D1", "D2"]}}

        output_dict_1 = {"Rooms": {"Ground_Floor": [["A1", "A2", "B1", "B2"], ["C1", "C2", "D1", "D2"]]}}

        input_5 = {"Rooms": {"Ground_Floor": "B1"}}

        output_dict_2 = {"Rooms": {"Ground_Floor": ["A1", "A2", "B1"]}}

        with mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=True):
            self.assertEqual(dict_merge([input_1, input_2]), output_dict)
            self.assertEqual(dict_merge([input_3, input_4]), output_dict_1)
            self.assertEqual(dict_merge([output_dict, input_5]), output_dict_2)

    def test_dict_merge_common_values(self):
        input_1 = {"Rooms": {"Ground_Floor": "A1"}}

        output_dict = {"Rooms": {"Ground_Floor": ["A1", "A1"]}}

        with mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=True):
            self.assertEqual(dict_merge([input_1, input_1]), input_1)
            self.assertEqual(dict_merge([input_1, input_1], True), output_dict)

    # def test_dict_merge_multiple_keys(self):
    #     input_1 = {"Rooms": {"Ground_Floor": ["A1", "A2"], "First_Floor": ["B1", "B2"]}}
    #     input_2 = {"Rooms": {"Ground_Floor": "A3", "First_Floor": "B3"}}
    #
    #     output_1 = {"Rooms": {"Ground_Floor": ["A1", "A2", "A3"], "First_FLoor": ["B1", "B2", "B3"]}}
    #
    #     self.assertEqual(dict_merge(input_1, input_2))

    def test_clean_dict(self):
        input_1 = {"Rooms": {"Ground_Floor": "A1", "First_Floor": "B1"}}
        clean_dict(input_1, "Ground_Floor")
        output_dict = {"Rooms": {"First_Floor": "B1"}}

        self.assertEqual(input_1, output_dict)

        input_2 = {"Labs": {"Ground_Floor": [{"Science": None, "Art": None}]}}
        clean_dict(input_2, "Science")
        output_dict_2 = {"Labs": {"Ground_Floor": {"Art": None}}}

        self.assertEqual(input_2, output_dict_2)

        clean_dict(input_2, "Art")
        output_dict_3 = {}

        self.assertEqual(input_2, output_dict_3)

        input_3 = {"Labs": {"Ground_Floor": [{"Science": [], "Art": []}]}}
        clean_dict(input_3, ["Science", "Art"])

        self.assertEqual(input_3, output_dict_3)

        input_4 = {
            "Labs": {
                "Ground_Floor": [{"Science": [], "Art": []}],
                "First_Floor": [{"Science": [], "Art": []}],
            }
        }
        clean_dict(input_4, ["First_Floor"])

        self.assertEqual(input_4, output_dict_3)

    def test_get_value_gen(self):
        input_1 = {"Rooms": {"Ground_Floor": "A1"}}
        output_1 = [("Ground_Floor", "A1")]

        self.assertEqual(list(get_value_gen(input_1, "Ground_Floor")), output_1)

        input_2 = {"Rooms": {"Ground_Floor": "A1"}, "Labs": {"Ground_Floor": "L1"}}
        output_2 = [("Ground_Floor", "A1"), ("Ground_Floor", "L1")]

        self.assertEqual(list(get_value_gen(input_2, "Ground_Floor")), output_2)

        input_3 = {
            "Building": {
                "Rooms": {"Ground_Floor": "A1"},
                "Labs": {"Ground_Floor": "L1"},
            }
        }
        output_3 = [("Ground_Floor", "A1"), ("Ground_Floor", "L1")]

        self.assertEqual(list(get_value_gen(input_3, "Ground_Floor")), output_3)

        input_4 = {
            "Main_Building": {
                "Rooms": {"Ground_Floor": "A1"},
                "Labs": {"Ground_Floor": "L1"},
            },
            "Sports_Hall": {"Ground_Floor": "Changing_Rooms"},
        }
        output_4 = [
            ("Ground_Floor", "A1"),
            ("Ground_Floor", "L1"),
            ("Ground_Floor", "Changing_Rooms"),
        ]

        self.assertEqual(list(get_value_gen(input_4, "Ground_Floor")), output_4)

        input_5 = {
            "Labs": {
                "Ground_Floor": [{"Science": "G1", "Art": []}],
                "First_Floor": [{"Science": "F1", "Art": []}],
            }
        }
        output_5 = [("Science", "G1"), ("Science", "F1")]

        self.assertEqual(list(get_value_gen(input_5, "Science")), output_5)

        input_6 = {
            "Labs": {
                "Ground_Floor": [{"Science": "G1", "Art": []}],
                "First_Floor": [{"Science": "F1", "Art": []}],
            },
            "Subjects": {"Science": "Triple"},
        }
        output_6 = [("Science", "G1"), ("Science", "F1"), ("Science", "Triple")]

        self.assertEqual(list(get_value_gen(input_6, "Science")), output_6)

    @mock.patch("eoio.utils.dict_tools.get_value_gen", autospec=False)
    def test_get_value_multiple_values(self, mm):
        input_1 = {
            "Labs": {
                "Ground_Floor": [{"Science": "G1", "Art": []}],
                "First_Floor": [{"Science": "F1", "Art": []}],
            }
        }
        output_1 = [("Science", "G1"), ("Science", "F1")]

        def fake(input, key):
            yield from output_1

        mm.side_effect = fake
        self.assertEqual(get_value(input_1, "Science"), output_1)

    @mock.patch("eoio.utils.dict_tools.get_value_gen", autospec=True)
    def test_get_value_multiple_levels(self, mm):
        input_1 = {
            "Main_Building": {
                "Rooms": {"Ground_Floor": "A1"},
                "Labs": {"Ground_Floor": "L1"},
            },
            "Sports_Hall": {"Ground_Floor": "Changing_Rooms"},
        }
        output_1 = [
            ("Ground_Floor", "A1"),
            ("Ground_Floor", "L1"),
            ("Ground_Floor", "Changing_Rooms"),
        ]

        def fake(input, key):
            yield from output_1

        mm.side_effect = fake
        self.assertEqual(get_value(input_1, "Ground_Floor"), output_1)

    @mock.patch("eoio.utils.dict_tools.get_value_gen", autospec=True)
    def test_get_value_single_value(self, mm):
        input_1 = {"Main_Building": {"Labs": {"Ground_Floor": "L1"}}}
        output_1 = {"Ground_Floor": "L1"}
        iter_obj = [("Labs", output_1)]

        def fake(input, key):
            yield from iter_obj

        mm.side_effect = fake
        self.assertEqual(get_value(input_1, "Labs"), output_1)

    @mock.patch("eoio.utils.dict_tools.get_value_gen", autospec=True)
    def test_get_value_duplicate_value(self, mm):
        input_1 = {"Main_Building": {"Height": "5 m"}, "Sports_Hall": {"Height": "5 m"}}
        output_1 = "5 m"
        iter_obj = [("Height", "5 m"), ("Height", "5 m")]

        def fake(input, key):
            yield from iter_obj

        mm.side_effect = fake
        self.assertEqual(get_value(input_1, "Height"), output_1)

    @mock.patch("eoio.utils.dict_tools.get_value_gen", autospec=True)
    def test_get_value_nonexistent_key(self, mm):
        input_1 = {"Main_Building": {"Labs": {"Ground_Floor": "L1"}}}

        def fake(input, key):
            yield from []

        mm.side_effect = fake
        self.assertEqual(get_value(input_1, "Storage"), None)

    def test_change_type_str(self):
        input_1 = {"Numbers": "1 2"}
        output_1 = {"Numbers": [1, 2]}

        with mock.patch("eoio.utils.dict_tools.val_format", return_value=[1, 2]):
            change_type(input_1)
            self.assertEqual(input_1, output_1)

        input_2 = {"GPS_TIME": "2022-09-10T20:23:47 2022-09-10T20:23:57 2022-09-10T20:24:07 2022-09-10T20:24:17"}
        output_2 = {
            "GPS_TIME": [
                dt.datetime(2022, 9, 10, 20, 23, 47),
                dt.datetime(2022, 9, 10, 20, 23, 57),
                dt.datetime(2022, 9, 10, 20, 24, 7),
                dt.datetime(2022, 9, 10, 20, 24, 17),
            ]
        }

        with mock.patch(
            "eoio.utils.dict_tools.val_format",
            return_value=[
                dt.datetime(2022, 9, 10, 20, 23, 47),
                dt.datetime(2022, 9, 10, 20, 23, 57),
                dt.datetime(2022, 9, 10, 20, 24, 7),
                dt.datetime(2022, 9, 10, 20, 24, 17),
            ],
        ):
            change_type(input_2)
            self.assertEqual(input_2, output_2)

    def test_change_type_list(self):
        input_1 = {"Names": ["Adam Howes", "Mattea Goalen", "Sam Hunt", "Nicole Reynolds"]}
        output_1 = {"Names": ["Adam Howes", "Mattea Goalen", "Sam Hunt", "Nicole Reynolds"]}

        with mock.patch(
            "eoio.utils.dict_tools.val_format",
            side_effect=["Adam Howes", "Mattea Goalen", "Sam Hunt", "Nicole Reynolds"],
        ):
            change_type(input_1)
            self.assertEqual(input_1, output_1)

        input_2 = {
            "Labs": {
                "Ground_Floor": [{"Science": "G1"}],
                "First_Floor": [{"Science": "F1"}],
            }
        }
        output_2 = {
            "Labs": {
                "Ground_Floor": [{"Science": "G1"}],
                "First_Floor": [{"Science": "F1"}],
            }
        }

        with mock.patch("eoio.utils.dict_tools.val_format", side_effect=["G1", "F1"]):
            change_type(input_2)
            self.assertEqual(input_2, output_2)

        input_3 = {
            "Ground_Floor": [
                {"Room_Numbers": "1 2 3 4 5"},
                {"Room_Capacity": "30 30 30 30 30"},
                {"Ceiling_Height": "2.5 2.5 2.4 2.5 2.45"},
            ]
        }
        output_3 = {
            "Ground_Floor": [
                {"Room_Numbers": [1, 2, 3, 4, 5]},
                {"Room_Capacity": [30, 30, 30, 30, 30]},
                {"Ceiling_Height": [2.5, 2.5, 2.4, 2.5, 2.45]},
            ]
        }

        with mock.patch(
            "eoio.utils.dict_tools.val_format",
            side_effect=[
                [1, 2, 3, 4, 5],
                [30, 30, 30, 30, 30],
                [2.5, 2.5, 2.4, 2.5, 2.45],
            ],
        ):
            change_type(input_3)
            self.assertEqual(input_3, output_3)

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_basic(self, key_in_dict_mock):
        input_1 = {"Main_Building": {"Height": "5 m"}, "Sports_Hall": {"Height": "5 m"}}
        remove_tag(input_1, "Height")
        self.assertEqual(input_1, {"Main_Building": "5 m", "Sports_Hall": "5 m"})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_different_levels(self, key_in_dict_mock):
        input_1 = {"Main_Building": {"Height": "5 m"}, "Height": {"Indoor": "5 m"}}
        remove_tag(input_1, "Height")
        self.assertEqual(input_1, {"Main_Building": "5 m", "Indoor": "5 m"})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", side_effect=[True, False])
    def test_remove_tag_key_in_dict(self, key_in_dict_mock):
        input_1 = {"Main_Building": {"Height": "5 m"}, "Height": {"Indoor": "5 m"}}
        remove_tag(input_1, "Main_Building")
        self.assertEqual(input_1, {"Main_Building_Height": "5 m", "Height": {"Indoor": "5 m"}})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_list_in_dict(self, key_in_dict_mock):
        input_1 = {"Rooms": {"Ground_Floor": ["A1", "A2", "B1", "B2"]}}
        remove_tag(input_1, "Ground_Floor")
        self.assertEqual(input_1, {"Rooms": ["A1", "A2", "B1", "B2"]})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_single_list_in_dict(self, key_in_dict_mock):
        input_1 = {
            "Ground_Floor": [
                {
                    "Science": {"Room_Numbers": [1, 2, 3, 4]},
                    "Art": {"Room_Numbers": [5, 6, 7, 8]},
                }
            ]
        }
        remove_tag(input_1, "Ground_Floor")
        self.assertEqual(
            input_1,
            {
                "Ground_Floor": [
                    {
                        "Science": {"Room_Numbers": [1, 2, 3, 4]},
                        "Art": {"Room_Numbers": [5, 6, 7, 8]},
                    }
                ]
            },
        )

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=True)
    def test_remove_tag_nested_dict_in_list(self, key_in_dict_mock):
        input_1 = {
            "Ground_Floor": [
                {
                    "Science": {"Room_Numbers": [1, 2, 3, 4]},
                    "Art": {"Room_Numbers": [5, 6, 7, 8]},
                }
            ]
        }
        remove_tag(input_1, "Science")
        self.assertEqual(
            input_1,
            {
                "Ground_Floor": [
                    {
                        "Science_Room_Numbers": [1, 2, 3, 4],
                        "Art": {"Room_Numbers": [5, 6, 7, 8]},
                    }
                ]
            },
        )

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_multiple_nested_dict_in_list(self, key_in_dict_mock):
        input_1 = {
            "Ground_Floor": [
                {
                    "Science": {"Room_Numbers": [1, 2, 3, 4]},
                    "Art": {"Room_Numbers": [5, 6, 7, 8]},
                }
            ]
        }
        remove_tag(input_1, "Room_Numbers")
        self.assertEqual(input_1, {"Ground_Floor": [{"Science": [1, 2, 3, 4], "Art": [5, 6, 7, 8]}]})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_with_ignore(self, key_in_dict_mock):
        input_1 = {
            "Ground_Floor": {
                "Science": {"Room_Numbers": [1, 2, 3, 4]},
                "Art": {"Room_Numbers": [5, 6, 7, 8]},
            }
        }
        remove_tag(input_1, "Science", "Art")
        self.assertEqual(input_1, {"Ground_Floor": {"Room_Numbers": [1, 2, 3, 4]}})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_with_multiple_ignore(self, key_in_dict_mock):
        input_1 = {
            "Ground_Floor": {
                "Science": {"Room_Numbers": [1, 2, 3, 4]},
                "Art": {"Room_Numbers": [5, 6, 7, 8]},
                "Geography": {"Room_Numbers": [9, 10, 11, 12]},
            }
        }
        remove_tag(input_1, "Science", ["Art", "Geography"])
        self.assertEqual(input_1, {"Ground_Floor": {"Room_Numbers": [1, 2, 3, 4]}})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_with_none(
        self, key_in_dict_mock
    ):  # this scenario is the one desired for the removal of #text tag -> remove_tag(imput_dict, "#text")
        input_1 = {
            "Ground_Floor": {
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": None},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": None},
            }
        }
        remove_tag(input_1, "Room_Numbers")
        self.assertEqual(input_1, {"Ground_Floor": {"Science": [1, 2, 3, 4], "Art": [5, 6, 7, 8]}})

    @mock.patch("eoio.utils.dict_tools.key_in_dict", return_value=False)
    def test_remove_tag_with_multiple_therefore_unchanged(
        self, key_in_dict_mock
    ):  # if other keys present tag not removed
        input_1 = {
            "Ground_Floor": {
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [30, 30, 30, 30]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [20, 20, 20, 20]},
            }
        }
        remove_tag(input_1, "Room_Numbers")
        self.assertEqual(
            input_1,
            {
                "Ground_Floor": {
                    "Science": {
                        "Room_Numbers": [1, 2, 3, 4],
                        "Capacity": [30, 30, 30, 30],
                    },
                    "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [20, 20, 20, 20]},
                }
            },
        )

    def test_pop_vals(self):
        main_dict = {
            "Address": {
                "House": {"Floors": 2, "Rooms": 6},
                "People": {
                    "Family": ["a", "b", "c"],
                    "Friends": ["d", "e", "f"],
                },
            },
        }
        other_dict = {"House": None, "Family": None}

        pop_vals(main_dict, ["House", "Family"], other_dict)

        output_main_dict = {
            "Address": {
                "People": {"Friends": ["d", "e", "f"]},
            },
        }

        output_other_dict = {
            "House": {"Floors": 2, "Rooms": 6},
            "Family": ["a", "b", "c"],
        }

        self.assertEqual(output_main_dict, main_dict)
        self.assertEqual(output_other_dict, other_dict)

    def test_remove_tag_in_key(self):
        input_dict = {
            "Ground_Floor": {
                "Class_List": {
                    "Science": {
                        "Room_Numbers": [1, 2, 3, 4],
                        "Capacity": [20, 20, 20, 20],
                    },
                    "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                    "Geography": {
                        "Room_Numbers": [9, 10, 11, 12],
                        "Capacity": [20, 20, 30, 30],
                    },
                },
            },
        }
        output_dict = {
            "Ground_Floor": {
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [20, 20, 20, 20]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                "Geography": {
                    "Room_Numbers": [9, 10, 11, 12],
                    "Capacity": [20, 20, 30, 30],
                },
            }
        }
        remove_tag_in_key(input_dict, "List")
        self.assertEqual(input_dict, output_dict)

    def test_list_keys(self):
        input_dict = {
            "Ground_Floor": {
                "@floor_id": 0,
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [20, 20, 20, 20]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                "Geography": {
                    "Room_Numbers": [9, 10, 11, 12],
                    "Capacity": [20, 20, 30, 30],
                },
            }
        }
        output_list = [
            "Art",
            "Capacity",
            "Geography",
            "Ground_Floor",
            "Room_Numbers",
            "Science",
        ]
        self.assertEqual(list_keys(input_dict), output_list)

    def test_list_keys_tag(self):
        input_dict = {
            "Ground_Floor": {
                "@floor_id": 0,
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [20, 20, 20, 20]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                "Geography": {
                    "Room_Numbers": [9, 10, 11, 12],
                    "Capacity": [20, 20, 30, 30],
                },
            }
        }
        output_list = [
            "@floor_id",
            "Art",
            "Capacity",
            "Geography",
            "Ground_Floor",
            "Room_Numbers",
            "Science",
        ]
        self.assertEqual(list_keys(input_dict, True), output_list)

    def test_list_keys_list(self):
        input_dict = {
            "Ground_Floor": {
                "@floor_id": 0,
                "Science": [
                    {"Room_Numbers": [1, 2, 3, 4]},
                    [{"Capacity": [20, 20, 20, 20]}],
                ],
                "Geography": [
                    {"Room_Numbers": [9, 10, 11, 12]},
                    [
                        {
                            "Capacity": [20, 20, 30, 30],
                        }
                    ],
                ],
            }
        }
        output_list = [
            "Capacity",
            "Geography",
            "Ground_Floor",
            "Room_Numbers",
            "Science",
        ]
        self.assertEqual(list_keys(input_dict), output_list)

    def test_remove_parent_tag(self):
        input_dict = {"Satellite": {"S2": {"S2_RESOLUTION": 60, "S2_HEIGHT": 70}}}
        output_dict = {"Satellite": {"S2": {"RESOLUTION": 60, "HEIGHT": 70}}}
        remove_parent_tag(input_dict)

        self.assertDictEqual(input_dict, output_dict)

    def test_create_parent_key(self):
        input_dict = {"Satellite": {"S2_RESOLUTION": 60, "S2_HEIGHT": 70}}
        output_dict = {"Satellite": {"S2": {"RESOLUTION": 60, "HEIGHT": 70}}}
        create_parent_key(input_dict)

        self.assertDictEqual(input_dict, output_dict)

    def test_create_parent_key_ignore(self):
        input_dict = {"Satellite": {"S2_RESOLUTION": 60, "S2_HEIGHT": 70}}
        output_dict = {"Satellite": {"S2_RESOLUTION": 60, "S2_HEIGHT": 70}}
        create_parent_key(input_dict, to_ignore="S2")

        self.assertDictEqual(input_dict, output_dict)

    def test_create_parent_key_add(self):
        input_dict = {"Satellite": {"S2_RESOLUTION_ID_0": 60, "S2_HEIGHT_ID_0": 70}}
        output_dict = {"Satellite": {"S2": {"RESOLUTION": {"ID": {"0": 60}}, "HEIGHT": {"ID": {"0": 70}}}}}
        create_parent_key(input_dict, to_add=["RESOLUTION", "HEIGHT", "ID"])

        self.assertDictEqual(input_dict, output_dict)

    def test_create_parent_key_add_ignore(self):
        input_dict = {"Satellite": {"S2_LAT_RESOLUTION_ID_0": 60, "S2_LAT_HEIGHT_ID_0": 70}}
        output_dict = {"Satellite": {"S2_LAT": {"RESOLUTION": {"ID": {"0": 60}}, "HEIGHT": {"ID": {"0": 70}}}}}
        create_parent_key(input_dict, to_ignore=["S2"], to_add=["RESOLUTION", "HEIGHT", "ID"])

        self.assertDictEqual(input_dict, output_dict)

    def test_multiple_keys_present_single_key_dict(self):
        input_dict = {
            "Ground_Floor": {
                "@floor_id": 0,
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [20, 20, 20, 20]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                "Geography": {
                    "Room_Numbers": [9, 10, 11, 12],
                    "Capacity": [20, 20, 30, 30],
                },
            }
        }
        input_keys = "Science"

        self.assertTrue(multiple_keys_present(input_dict, input_keys))

    def test_multiple_keys_present_single_key_list(self):
        input_dict = [
            {
                "Ground_Floor": {
                    "@floor_id": 0,
                    "Science": {
                        "Room_Numbers": [1, 2, 3, 4],
                        "Capacity": [20, 20, 20, 20],
                    },
                    "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                    "Geography": {
                        "Room_Numbers": [9, 10, 11, 12],
                        "Capacity": [20, 20, 30, 30],
                    },
                }
            }
        ]
        input_keys = "Geography"

        self.assertTrue(multiple_keys_present(input_dict, input_keys))

    def test_multiple_keys_present_nested_list(self):
        input_dict = [
            [
                {
                    "Ground_Floor": {
                        "@floor_id": 0,
                        "Science": {
                            "Room_Numbers": [1, 2, 3, 4],
                            "Capacity": [20, 20, 20, 20],
                        },
                        "Art": {
                            "Room_Numbers": [5, 6, 7, 8],
                            "Capacity": [25, 20, 24, 24],
                        },
                        "Geography": {
                            "Room_Numbers": [9, 10, 11, 12],
                            "Capacity": [20, 20, 30, 30],
                        },
                    }
                }
            ]
        ]
        input_keys = "Geography"

        self.assertTrue(multiple_keys_present(input_dict, input_keys))

    def test_multiple_keys_present_multiple_keys_dict(self):
        input_dict = {
            "Ground_Floor": {
                "@floor_id": 0,
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [20, 20, 20, 20]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                "Geography": {
                    "Room_Numbers": [9, 10, 11, 12],
                    "Capacity": [20, 20, 30, 30],
                },
            }
        }
        input_keys = ["Science", "Art"]

        self.assertTrue(multiple_keys_present(input_dict, input_keys))

    def test_multiple_keys_present_multiple_keys_diff_levels_dict(self):
        input_dict = {
            "Ground_Floor": {
                "@floor_id": 0,
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [20, 20, 20, 20]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                "Geography": {
                    "Room_Numbers": [9, 10, 11, 12],
                    "Capacity": [20, 20, 30, 30],
                },
            }
        }
        input_keys = ["Science", "Capacity"]

        self.assertTrue(multiple_keys_present(input_dict, input_keys))

    def test_get_nested_value(self):
        input_dict = {
            "Ground_Floor": {
                "@floor_id": 0,
                "Science": {"Room_Numbers": [1, 2, 3, 4], "Capacity": [20, 20, 20, 20]},
                "Art": {"Room_Numbers": [5, 6, 7, 8], "Capacity": [25, 20, 24, 24]},
                "Geography": {
                    "Room_Numbers": [9, 10, 11, 12],
                    "Capacity": [20, 20, 30, 30],
                },
            }
        }
        input_keys = ["Science", "Capacity"]

        self.assertTrue(get_nested_value(input_dict, input_keys), [20, 20, 20, 20])

    def test_get_dict_path(self):
        pass

    def test_make_value_key(self):
        input_iterable = [
            {"ID": "name", "value": "John", "units": None},
            {"ID": "age", "value": "27", "units": "year"},
        ]
        key = "ID"
        output = {
            "name": {"value": "John", "units": None},
            "age": {"value": "27", "units": "year"},
        }

        self.assertEqual(make_value_key(input_iterable, key), output)

    def test_replace_key_names(self):
        input_dict = {"A": 1, "AA": {"B": 2, "C": 3, "CC": {"D": 4}}}
        key_mapping = {"CC": "cc", "A": "a"}

        replace_key_names(input_dict, key_mapping)

        output_dict = {"a": 1, "AA": {"B": 2, "C": 3, "cc": {"D": 4}}}

        self.assertDictEqual(input_dict, output_dict)


if __name__ == "__main__":
    unittest.main()
