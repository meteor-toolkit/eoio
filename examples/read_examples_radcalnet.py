if __name__ == "__main__":
    from eoio.interface import (
        read,
    )
    import os.path
    import configparser

    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "eoio",
        "etc",
        "test_file_paths.config",
    )
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    # Assume config has a section [RadCalNet] and key 'output_file'
    test_path = config.get("RadCalNet", "output_file", fallback=None)

    SUBSET = {
        "wavelength": {"min": 420, "max": 460},
        "time_of_day_utc": {"min": "9:00", "max": "11:00"},
        "time_of_day_local": {"min": "12:00", "max": "13:00"},
        "angle": {
            "sza": {"min": 40, "max": 80},
            "saa": {"min": 0, "max": 180},
        },
        "datetime": {"nearest": "20171231", "tolerance_days": 300},
    }

    ds_out = read(test_path, subset=SUBSET)
    print(ds_out)

    ds_in = read(
        test_path.replace("GONA01_2022_187_v04.09.output", "GONA01_2022_187_v00.09.input"),
        # subset=SUBSET
    )
    print(ds_in)
