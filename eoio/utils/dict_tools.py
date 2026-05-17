"""eoio.utils.dict_tools - dictionary utility functions"""

from copy import copy, deepcopy
from typing import Any, Generator, List, Optional, Union

import numpy as np

from eoio.utils.formatters import val_format

__author__ = "Mattea Goalen <mattea.goalen@npl.co.uk>"

__all__ = [
    "key_exists",
    "key_present",
    "multiple_keys_present",
    "key_in_dict",
    "get_value_gen",
    "get_value",
    "dict_merge",
    "clean_dict",
    "empty_dict",
    "rmv_empty_dict",
    "change_type",
    "remove_tag",
    "pop_vals",
    "remove_tag_in_key",
    "list_keys",
    "remove_parent_tag",
    "create_parent_key",
    "get_nested_value",
    "get_dict_path",
    "make_value_key",
    "replace_key_names",
]


def key_exists(test_dict: dict, keys: Union[str, list], include: str = "any"):
    """
    Determine whether a key (or multiple keys) are present in a nested iterable.

    Default ('any' input) return True if any of the listed keys are in the iterable, else False.
    If 'all' input selected for 'include', return True if all listed keys are in iterable, else False.

    :param test_dict: iterable through which to search
    :param keys: keys to search for in iterable
    :param include: whether to search for 'all' keys or 'any' keys
    :returns: ``True`` when the requested key or keys are present, otherwise ``False``.
    """
    if include == "any":
        value = key_present(test_dict, keys)
    elif include == "all":
        value = multiple_keys_present(test_dict, keys)
    else:
        raise ValueError("'include' value must be either 'any' or 'all'")
    return value


def key_present(test: Union[dict, list], keys: Union[str, list]) -> bool:
    """
    Determine whether a key is present in a nested iterable

    :param test: iterable through which to search
    :param keys: keys to search for in iterable
    :returns: ``True`` when any requested key is present, otherwise ``False``.
    """
    value = False
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(test, dict):
        for k, v in zip(list(test.keys()), list(test.values())):
            if value is True:
                return value
            elif k in keys:
                value = True
            elif isinstance(v, dict) and v != {}:
                value = key_present(test[k], keys)
            elif isinstance(v, list):
                value = key_present(test[k], keys)
            else:
                pass
    elif isinstance(test, list) and all([isinstance(i, dict) for i in test]):
        for i in range(len(test)):
            value = key_present(test[i], keys)
            if value is True:
                return value
    return value


def multiple_keys_present(test: Union[dict, list], keys: Union[str, list]) -> bool:
    """
    Determine whether all keys are present in a nested iterable

    :param test: iterable through which to search
    :param keys: keys to search for in iterable
    :returns: ``True`` when all requested keys are present, otherwise ``False``.
    """
    value = False
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(test, dict):
        for k, v in zip(list(test.keys()), list(test.values())):
            if value is True:
                return value
            elif k in keys:
                keys.remove(k)
                if len(keys) == 0:
                    value = True
                    return value
            if isinstance(v, dict) and v != {}:
                value = multiple_keys_present(test[k], keys)
            elif isinstance(v, list):
                value = multiple_keys_present(test[k], keys)
            else:
                pass
    elif isinstance(test, list) and all([isinstance(i, (dict, list)) for i in test]):
        for i in range(len(test)):
            value = multiple_keys_present(test[i], keys)
            if value is True:
                return value
    return value


def empty_dict(test_dict) -> bool:
    """
    Return whether there are empty values within a dictionary

    :param test_dict: dictionary through which to iterate
    :returns: ``True`` when empty values are present, otherwise ``False``.
    """
    value = False
    if len(test_dict) == 0:
        value = True
    elif isinstance(test_dict, dict):
        for k, v in test_dict.items():
            if value is True:
                return value
            elif v is None:
                value = True
            elif isinstance(v, str) and v == "none":
                value = True
            elif isinstance(v, dict):
                if len(v.values()) == 0:
                    value = True
                else:
                    value = empty_dict(v)
            elif isinstance(v, list):
                if len(v) == 0:
                    value = True
                for i in v:
                    if isinstance(i, dict):
                        value = empty_dict(i)
                    elif i is None:
                        value = True
            else:
                value = False
    return value


def rmv_empty_dict(test_dict) -> None:
    """
    Remove empty dictionary values

    :param test_dict: dictionary with empty values in
    """
    for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
        if isinstance(v, dict):
            if len(v) == 0:
                test_dict.pop(k)
            else:
                rmv_empty_dict(v)
        elif v is None:
            test_dict.pop(k)
        elif isinstance(v, str) and v == "none":
            test_dict.pop(k)
        elif isinstance(v, list):
            if len(v) == 0:
                del test_dict[k]
            elif any([isinstance(i, dict) for i in v]):
                for i in list(v):
                    if isinstance(i, dict):
                        if len(i) == 0:
                            del v[v.index(i)]
                        else:
                            rmv_empty_dict(i)
                    elif i is None:
                        del v[v.index(i)]
            elif any([i is None for i in v]):
                for i in list(v):
                    if i is None:
                        del v[v.index(i)]
            else:
                pass


def key_in_dict(test_dict, other_dict) -> bool:
    """
    Return whether a key is jointly present in two input dictionaries.

    Note: does not search through nested dictionary keys

    :param test_dict: first dictionary
    :param other_dict: second dictionary
    :returns: ``True`` when any key is shared, otherwise ``False``.
    """
    if isinstance(test_dict, dict) and isinstance(other_dict, dict):
        for k in list(test_dict.keys()):
            if k in list(other_dict.keys()):
                return True
            else:
                pass
    return False


def dict_merge(input_dicts: List[dict], include_duplicates=False) -> dict:
    """
    Merge two (or more) dictionaries together, extending common key values if dictionary values differ

    :param input_dicts: dictionaries to be merged
    :param include_duplicates: whether or not to include duplicate key values
    :returns: Merged dictionary.
    """
    if len(input_dicts) > 2:
        main_dict = input_dicts[0]
        for d in input_dicts[1:]:
            main_dict = dict_merge([main_dict, d])
        return main_dict

    main_dict = deepcopy(input_dicts[0])
    add_dict = input_dicts[1]
    if key_in_dict(main_dict, add_dict):
        for k, v in add_dict.items():
            if k in main_dict.keys():
                if not isinstance(main_dict[k], dict):
                    if v == main_dict[k] and include_duplicates is False:
                        pass
                    else:
                        if isinstance(main_dict[k], list) and isinstance(v, list):
                            if np.array(main_dict[k]).shape == np.array(v).shape:
                                # if all(isinstance(i, type(v[0])) for i in main_dict[k]):
                                main_dict[k] = [main_dict[k], v]
                            else:
                                main_dict[k].append(v)
                        else:
                            try:
                                main_dict[k].append(v)
                            except AttributeError:
                                main_dict[k] = [main_dict[k], v]
                elif main_dict[k] == {}:
                    main_dict[k] = v
                elif isinstance(main_dict[k], dict) and isinstance(add_dict[k], dict):
                    main_dict[k] = dict_merge([main_dict[k], add_dict[k]], include_duplicates)
                else:
                    print(add_dict)
            else:
                main_dict.update({k: v})
    else:
        main_dict = {**input_dicts[0], **add_dict}

    return main_dict


def clean_dict(test_dict: dict, keys: Union[list, str]) -> None:
    """
    Remove specified key/s from a dictionary

    :param test_dict: input dictionary from which to remove keys
    :param keys: key/s to remove from the dictionary
    """
    if isinstance(keys, list) is False:
        keys = [keys]
    for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
        if k in keys:
            test_dict.pop(k)
        elif isinstance(v, dict):
            clean_dict(test_dict[k], keys)
        elif isinstance(v, list) and all([isinstance(i, dict) for i in v]):
            for i in v:
                clean_dict(test_dict[k][v.index(i)], keys)
            if test_dict[k] == [{}]:
                test_dict.pop(k)
    for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
        if isinstance(v, (list, dict)) and len(v) == 0:
            test_dict.pop(k)
        elif isinstance(v, list) and len(v) == 1:
            test_dict[k] = v[0]


def get_value_gen(test_dict: dict, key: str) -> Generator:
    """
    Get generator function of dictionary values associated with the specified key

    :param test_dict: input iterator in which to search for the key-values pair/s
    :param key: key to use to search through dictionary
    :returns: Generator yielding key-value pairs matching ``key``.
    """
    if isinstance(test_dict, dict):
        for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
            if k == key:
                t = deepcopy(test_dict.get(k))
                yield k, t
            elif isinstance(v, list) and all([isinstance(i, dict) for i in v]):
                for i, vel in enumerate(v):
                    yield from get_value_gen(test_dict[k][i], key)
            else:
                yield from ([] if not isinstance(v, dict) else get_value_gen(test_dict[k], key))
    elif isinstance(test_dict, list) and all([isinstance(i, dict) for i in test_dict]):
        for i, vel in enumerate(test_dict):
            yield from get_value_gen(test_dict[i], key)


def get_value(test_dict, key, multiple=False, default: Optional[Any] = None):
    """
    Return dictionary values associated with the specified key

    :param multiple: if multiple keys have same name within the dictionaries, specifying True will return all as a list
    :param test_dict: input dictionary in which to search for the key-value pair/s
    :param key: key to use to search through dictionary
    :returns: Matching value or values associated with ``key``.
    """
    value_list = list(get_value_gen(test_dict, key))
    try:
        if len(value_list) == 1 or all([True if i[1] == value_list[0][1] else False for i in value_list]):
            if multiple:
                return value_list
            else:
                return dict(value_list)[key]
    except KeyError:
        return default
    except ValueError:
        pass
    if multiple is True and value_list:
        return value_list
    elif value_list:
        print(
            "Multiple different values found to be associated with '{}'. Consider filtering dictionary further, or include multiple=True to return all values.".format(
                key
            )
        )
        return value_list
    print("No value found associated with '{}'. Check spelling and letter case.".format(key))
    return default


def change_type(test_dict) -> None:
    """
    Convert value types from str to other recognised types in dictionary.

    Recognised types include:

    - int
    - float
    - list
    - datetime objects (valid string formats specified at links below)
        https://docs.python.org/3/library/datetime.html#datetime.datetime.fromisoformat
        https://docs.python.org/3/library/datetime.html#datetime.time.fromisoformat

    Uses eoio.utils.formatters.val_format to convert values

    :param test_dict: dictionary with values to changes
    """
    if isinstance(test_dict, dict):
        for k, v in test_dict.items():
            if isinstance(v, str):
                test_dict[k] = val_format(v)
            elif isinstance(v, list):
                for i, j in enumerate(v):
                    if isinstance(j, str):
                        test_dict[k][i] = val_format(j)
                    else:
                        change_type(j)
            elif isinstance(v, dict):
                change_type(v)
    elif isinstance(test_dict, list):
        for i in test_dict:
            change_type(i)


def remove_tag(
    test_dict: dict,
    tag: str,
    ignore: Optional[Union[list, str]] = None,
    in_list: Optional[dict] = None,
) -> None:
    """
    Remove specified tag/key from dictionary and move the values up a level in the dictionary.

    Note: if other keys are also present at the same level in the dictionary, tag isn't removed
    unless other keys are specified as ignore

    :param test_dict: dictionary in which tags are to be removed
    :param tag: key to remove from dictionary
    :param ignore: list or str of keys to ignore
    :param in_list: dictionary used within function to determine whether the tag is in the nested iterable or not
    """
    if in_list is None:
        in_list = {"value": False}

    if isinstance(ignore, str) or ignore is None:
        ignore = [ignore]
    if isinstance(test_dict, dict):
        for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
            if any([v is None, v == "none"]):
                test_dict.pop(k)
            if k in ignore:
                test_dict.pop(k)
        if tag in test_dict.keys():
            for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
                if k == tag:
                    vals = test_dict[tag]
                    if key_in_dict(vals, test_dict) or any(
                        [True if all([key_in_dict(vals, v), tag != k]) else False for k, v in test_dict.items()]
                    ):
                        vals_2 = {}
                        for k2, v2 in vals.items():
                            vals_2[tag + "_" + k2] = v2
                        vals = vals_2
                    if isinstance(vals, dict):
                        test_dict.pop(tag)
                        test_dict.update(vals)
                    elif all([True if any([tag == k, ignore == k]) else False for k in test_dict.keys()]):
                        test_dict = vals
                        in_list["value"] = True
                elif isinstance(v, dict):
                    remove_tag(v, tag, ignore, in_list)
                    if in_list["value"] is True:
                        test_dict[k] = test_dict[k].get(tag)
                        in_list["value"] = False
        else:
            for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
                if isinstance(v, dict):
                    remove_tag(v, tag, ignore, in_list)
                    if in_list["value"] is True:
                        test_dict[k] = test_dict[k].get(tag)
                        in_list["value"] = False
                elif isinstance(v, list):
                    if all([isinstance(i, dict) for i in v]):
                        for i, vel in enumerate(v):
                            remove_tag(vel, tag, ignore, in_list)
                            if in_list["value"] is True:
                                if all(
                                    [
                                        (True if any([tag == k1, ignore == k1]) else False)
                                        for k1 in test_dict[k][i].keys()
                                    ]
                                ):
                                    test_dict[k][i] = test_dict[k][i].get(tag)
                                in_list["value"] = False
    elif isinstance(test_dict, list):
        if all([isinstance(i, dict) for i in test_dict]):
            for i, vel in enumerate(test_dict):
                remove_tag(vel, tag, ignore, in_list)
                if in_list["value"] is True:
                    if all([True if any([tag == k1, ignore == k1]) else False for k1 in test_dict[i].keys()]):
                        test_dict[i] = test_dict[i].get(tag)
                    in_list["value"] = False


def pop_vals(test_dict, key_list, d):
    """
    Remove specified key-value pairs from dictionary and assign to a second dictionary

    :param test_dict: dictionary from which to remove the values from
    :param key_list: keys of the values you wish to transfer to the second dictionary
    :param d: second dictionary in which key-value pairs will be transferred
    """
    test_dict2 = copy(test_dict)
    for k, v in test_dict2.items():
        if k in key_list:
            d[k] = test_dict.pop(k)
        elif isinstance(v, dict):
            pop_vals(v, key_list, d)
        elif isinstance(v, list) and all([isinstance(i, dict) for i in v]):
            for i in v:
                pop_vals(i, key_list, d)


def remove_tag_in_key(test_dict: dict, tag: str) -> None:
    """
    Remove keys in nested dict containing the string tag

    :param test_dict: nested dictionary
    :param tag: string tag to search for in keys
    """
    for k, v in zip(list(test_dict.keys()), list(test_dict.values())):
        if tag in k:
            data = test_dict.pop(k)
            if isinstance(data, dict):
                if all([isinstance(v2, dict) for v2 in data.values()]):
                    remove_tag_in_key(data, tag)
                test_dict.update(data)
        elif isinstance(v, dict):
            remove_tag_in_key(v, tag)


def list_keys(test_dict, tags=False, new_list=None):
    """
    List all keys in nested dictionary, useful when searching for keys in a Dataset dictionary

    :param test_dict: dictionary to search
    :param tags: whether or not to include keys with '@' or '#' in the list
    :param new_list: list of keys to append to, default None creates a new empty list to return
    :returns: Sorted list of unique keys found in the nested dictionary.
    """
    if new_list is None:
        new_list = []
    if isinstance(test_dict, dict):
        for k, v in test_dict.items():
            if all(["@" not in k, "#" not in k]):
                new_list.append(k)
            elif tags is True:
                new_list.append(k)
            if isinstance(v, (dict, list)):
                new_list = list_keys(v, tags, new_list)
    elif isinstance(test_dict, list) and any([isinstance(i, dict) for i in test_dict]):
        for i in test_dict:
            new_list = list_keys(i, tags, new_list)
    new_list = [*set(new_list)]
    new_list.sort()
    return new_list


def remove_parent_tag(input_dict: dict) -> None:
    """
    Remove text common between a key and the key one level up from it in a nested dictionary

    :param input_dict: input dictionary to modify
    """
    for k, v in zip(list(input_dict.keys()), list(input_dict.values())):
        if isinstance(v, dict):
            k_split = k.split("_")  # split words up if multiple in key
            for k2, v2 in zip(list(v.keys()), list(v.values())):
                k2_split = k2.split("_")
                matches = k2_split[0] in k_split
                if matches:
                    new_k = k2.replace(k2_split[0] + "_", "")
                    input_dict[k].update({new_k: input_dict[k][k2]})
                    input_dict[k].pop(k2)
            remove_parent_tag(input_dict[k])


def create_parent_key(
    input_dict: dict,
    delim="_",
    to_ignore: Optional[Union[list, str]] = None,
    to_add: Optional[Union[list, str]] = None,
) -> None:
    """
    Combine keys with common text under a new parent key in dictionary

    :param input_dict: dictionary to modify
    :param delim: delimiter to use when splitting the keys, defaults to _
    :param to_ignore: list or str of any text values to ignore as first values
    :param to_add: list or str of any text to add to first values (things you want as a key, starting from first level
                   with no shared keys)
    """
    ignored = False
    # set to_ignore and to_add values to list
    if isinstance(to_ignore, str):
        to_ignore = [to_ignore]
    if not to_ignore:
        to_ignore = []
    if not isinstance(to_ignore, list):
        raise ValueError(
            """
        '{}' is an invalid input for to_ignore parameter. 
        Please input a list or string, or leave empty if not 
        required.""".format(
                to_ignore,
            )
        )
    if isinstance(to_add, str):
        to_add = [to_add]
    if not to_add:
        to_add = []
    if not isinstance(to_add, list):
        raise ValueError(
            """
        '{}' is an invalid input for to_add parameter. 
        Please input a list or string, or leave empty if not 
        required.""".format(
                to_add,
            )
        )

    # check if input dictionary
    if isinstance(input_dict, dict):
        # calculate first text strings in keys
        first_key_vals = [i.split(delim)[0] for i in list(input_dict.keys())]
        common_text = [i for i in sorted([*set(first_key_vals)]) if first_key_vals.count(i) > 1 and i not in to_ignore]
        # set new keys if using to_ignore
        if len(common_text) == 0 and len([item for item in to_ignore if item in first_key_vals]) != 0:
            common_text = list(
                set(
                    [
                        i.split(delim)[i.index(first_key_vals[j])]
                        + "_"
                        + i.split(delim)[i.index(first_key_vals[j]) + 1]
                        for j, i in enumerate(list(input_dict.keys()))
                    ]
                )
            )
            ignored = True
        # set new keys if using to_add
        if len(common_text) == 0 and len([item for item in to_add if item in first_key_vals]) != 0:
            common_text = first_key_vals

        # move matching dictionary key, value pairs into new parent dictionaries
        for new_k in common_text:
            # make new key if full key does not exist already
            if new_k not in input_dict:
                # create new key
                input_dict.update({new_k: {}})
                # update new key and delete old keys
                for k, v in zip(list(input_dict.keys()), list(input_dict.values())):
                    if k != new_k and (ignored or k.split(delim)[0] == new_k):
                        input_dict[new_k].update({k.replace(new_k + delim, ""): v})
                        input_dict.pop(k)
        # iter through value dictionaries
        for k, v in input_dict.items():
            if isinstance(v, dict):
                create_parent_key(input_dict[k], delim, to_ignore, to_add)


def get_nested_value(input_dict: dict, keys: list):
    """
    Return a single value in a nested dictionary at the path loosely defined by the keys.

    Note
    Useful if there are multiple values return by 'get_value' as you can filter the nested
    dictionaries that being searched

    Example dictionary to search through and appropriate keys
    example_dict =
        {"Satellite_1": {"Bands": {"B01": {"Wavelength": 10}, "B02": { "Wavelength": 20}}},
        "Satellite_2": {"Bands": {"B01": {"Wavelength": 30}, "B02": { "Wavelength": 40}}}}
    example_keys =
        ["Satellite_2", "B01", "Wavelength"]

    get_nested_value(example_dict, example_keys) == 30

    :param input_dict: input dictionary through which to search
    :param keys: list of keys ordered from out to in
    """
    value = input_dict
    for k in keys:
        value = get_value(value, k)
    return value


def get_dict_path(input_dict: dict, value: str, new_list: Optional[list] = None) -> list:
    """
    Return list of keys to get to value in input dictionary. Empty list returned if value isn't present in dictionary

    :param input_dict: input dictionary through which to search for the value given
    :param value: key to look for in the input dictionary
    :param new_list: optional list to append key path to
    :returns: List containing the key path to ``value``.
    """
    if new_list is None:
        new_list = []
    for k, v in input_dict.items():
        if key_exists(v, value):
            new_list.append(k)
            new_list = get_dict_path(v, value, new_list)
    return new_list


def make_value_key(input_iterable: Union[list, dict], key: str) -> Union[dict, list]:
    """
    Convert a list of dictionaries into a dictionary using value denoted by the key as a key name.

    Example:
        input_iterable = [{"ID": "name", "value": "John", "units": None}, {"ID": "age", "value": "27", "units": "year"}]
        key = "ID"
        output = {"name": {"value": "John", "units": None}, "age": {"value": "27", "units": "year"}}

    :param input_iterable: dictionary or list to reformat
    :param key: key of the value to use as a key for the dictionaries in a list
    """
    if isinstance(input_iterable, dict):
        for k, v in input_iterable.items():
            input_iterable[k] = make_value_key(v, key)
        return input_iterable
    elif isinstance(input_iterable, list) and all([isinstance(i, dict) for i in input_iterable]):
        if all([key in i.keys() for i in input_iterable]):
            return dict([(i.pop(key), i) for i in input_iterable])
        else:
            return input_iterable
    else:
        return input_iterable


def replace_key_names(input_dict: dict, key_mapping: dict):
    """
    Map old key names to new ones provided in `key_mapping` in dictionary

    :param input_dict: input dictionary in which to apply the key name changes
    :param key_mapping: dictionary mapping old key names to new ones
    """
    for k, v in zip(list(input_dict.keys()), list(input_dict.values())):
        if k in key_mapping.keys():
            input_dict[key_mapping[k]] = input_dict.pop(k)
            if isinstance(v, dict):
                replace_key_names(input_dict[key_mapping[k]], key_mapping)
        else:
            if isinstance(v, dict):
                replace_key_names(input_dict[k], key_mapping)
