import os
import numpy as np
import pandas as pd
import pytz
import datetime
import warnings

pems_time_format = "%m/%d/%Y %H:%M"
mph_to_mps = 0.44704


def time_pems_to_unix(time_str, timezone):
    naive = datetime.datetime.strptime(time_str, pems_time_format)
    localized = timezone.localize(naive)
    utc_tz = pytz.timezone("UTC")
    utc_time = localized.astimezone(utc_tz)
    return utc_time.timestamp()


def lin_interp(val, inputs, outputs):
    denom = inputs[1] - inputs[0]
    assert(denom > 0.0)
    slope = (outputs[1] - outputs[0]) / (inputs[1] - inputs[0])
    return outputs[0] + slope * (val - inputs[0])


class StatDetector(object):
    def __init__(self, dirname, vds_id):
        self._timezone = pytz.timezone('US/Pacific')
        self._csv_path = os.path.join(dirname, "pems_vds_{}.csv".format(vds_id))
        self.vds_id = vds_id
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

    def calc_val_at_time(self, col, time):
        delta_times = time - self.data['time']
        filt = delta_times >= 0
        before, after = (delta_times[filt], delta_times[np.invert(filt)])
        idx1 = before.idxmin() if not before.empty else None
        idx2 = after.idxmax() if not after.empty else None
        if idx1 is None and idx2 is None:
            raise RuntimeError("VDS {}: times are empty!".format(self.vds_id))
        elif idx1 is None and idx2 is not None:
            warnings.warn("VDS {}: time outside data. Last time's {} taken".format(self.vds_id, col))
            return self.data[col][idx2]
        elif idx2 is None and idx1 is not None:
            warnings.warn(
                "VDS {}: time outside data. First time's {} taken".format(self.vds_id, col))
            return self.data[col][idx1]
        else:
            return lin_interp(time,
                              self.data['time'].values[[idx1, idx2]],
                              self.data[col].values[[idx1, idx2]])


def read_detector_if_exists(row_with_vds_id, dirname):
    vds_id = row_with_vds_id['vds_id']
    if vds_id is not None and not np.isnan(vds_id):
        return StatDetector(dirname, int(vds_id))
    return None