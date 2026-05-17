import numpy as np
from dateutil import parser
from datetime import datetime, time


class utils:
    """
    Utility functions for reading radcalnet data

    """

    @staticmethod
    def subset_to_timestamp(date, stime) -> datetime:
        # parse the date to give datetime fmt="dd-mm-yyyy"
        if isinstance(date, np.datetime64) | isinstance(date, str):
            date = parser.parse(str(date)).date()

        # parse and return stime as full datetime.datetime
        if isinstance(stime, datetime):
            return stime
        elif isinstance(stime, time):
            return datetime.combine(date, stime)
        elif isinstance(stime, np.datetime64):
            return datetime.combine(date, parser.parse(str(stime)).time())
        elif isinstance(stime, str):
            if ":" in stime:
                return datetime.combine(date, parser.parse(stime).time())

        raise ValueError("incorrect format for time subset, must be datetime, time or string in format HH:MM")
