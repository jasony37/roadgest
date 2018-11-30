import pandas as pd
import pytz
import datetime

pems_time_format = "%m/%d/%Y %H:%M"
mph_to_mps = 0.44704


def time_pems_to_unix(time_str, timezone):
    naive = datetime.datetime.strptime(time_str, pems_time_format)
    localized = timezone.localize(naive)
    utc_tz = pytz.timezone("UTC")
    utc_time = localized.astimezone(utc_tz)
    return utc_time.timestamp()


class StatDetector(object):
    def __init__(self, csv_path):
        self._timezone = pytz.timezone('US/Pacific')
        self._csv_path = csv_path
        self.data = self.data_from_csv()

    def time_to_unix(self, time_str):
        return time_pems_to_unix(time_str, self._timezone)

    def data_from_csv(self):
        data = pd.read_csv(self._csv_path,
                           usecols=["5 Minutes", "Flow (Veh/5 Minutes)", "Speed (mph)"],
                           converters={"5 Minutes": self.time_to_unix})
        data["Flow (Veh/5 Minutes)"] /= 300.0
        data["Speed (mph)"] *= mph_to_mps
        data.columns = ["time", "flow", "speed"]
        return data