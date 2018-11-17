import os
import numpy as np
import pandas as pd
import warnings
from proc import gps


class CabData(object):
    def __init__(self, dirname):
        self._dirname = dirname
        self._fname = os.path.join(dirname, "_cabs.txt")
        self.cab_list = None
        self.cab_traces = None
        cab_traces_file = os.path.join(dirname, "cab_traces.pickle")
        self.read_cablist()
        if os.path.isfile(cab_traces_file):
            self.cabtraces_from_save(cab_traces_file)
        else:
            self.read_cabtraces()
            self.cab_traces.to_pickle(cab_traces_file)

    def _tag_value(self, line, tag_name):
        try:
            search_str = '{}="'.format(tag_name)
            start_idx = line.index(search_str) + len(search_str)
            end_idx = line.index('"', start_idx)
            return line[start_idx:end_idx]
        except ValueError as valerr:
            print(valerr)

    def _proc_line(self, line):
        line_s = line.rstrip()
        if line_s.startswith('<') and line_s.endswith('/>'):
            cab_id = self._tag_value(line_s, "cab id")
            updates = int(self._tag_value(line_s, "updates"))
            cab = {'id': cab_id, 'updates': updates}
            return cab
        else:
            warnings.warn("Line does not look like a tag. Ignored!")

    def read_cablist(self):
        cab_list = []
        with open(self._fname, 'r') as fin:
            for line in fin:
                cab_data = self._proc_line(line)
                if cab_data is not None:
                    cab_list.append(cab_data)
        self.cab_list = pd.DataFrame(cab_list)

    def cab_id_to_fname(self, cab_id):
        return os.path.join(self._dirname, "new_{}.txt".format(cab_id))

    def read_cabtraces(self):
        assert(self.cab_list is not None)
        id_col = self.cab_list['id'].apply(self.cab_id_to_fname)
        flist = self.cab_list.assign(fname=id_col)
        cab_data_list = []
        for cabf in flist.itertuples():
            cab_data = pd.read_csv(cabf.fname, delim_whitespace=True,
                                   names=['lat', 'long', 'occupancy', 'time'])
            cab_data_list.append(cab_data.assign(cab_id=cabf.id))
        self.cab_traces = pd.concat(cab_data_list, ignore_index=True)
        self.cab_traces.sort_values(by=['cab_id', 'time'], inplace=True)
        self.cab_traces.reset_index(drop=True, inplace=True)

    def cabtraces_from_save(self, fname):
        self.cab_traces = pd.read_pickle(fname)

    def calc_xy(self, lat_center, long_center):
        self.cab_traces['x'] = gps.long_to_x(self.cab_traces['long'],
                                             lat_center, long_center)
        self.cab_traces['y'] = gps.lat_to_y(self.cab_traces['lat'], lat_center)

    def check_on_section(self, road_section):
        """
        For each cab in self.cab_traces, data must previously be sorted by
        increasing time
        :param road_section:
        :return:
        """
        segments = zip(road_section.section['x'], road_section.section['y'])
        segments = pd.Series(list(segments))
        segments = pd.DataFrame({'start': segments, 'end': segments.shift(-1)})
        segments = segments[:-1]
        seg_assn = self.cab_traces.apply(gps.assign_segment,
                                         axis=1, args=[segments, 20.0])

        self.cab_traces['segment'] = seg_assn
        # self.cab_traces.groupby('cab_id').first()
        pass


class RoadSection(object):
    def __init__(self, fname):
        self.section = pd.read_csv(fname)
        self.center = np.mean(self.section[['lat', 'long']])
        self.calc_xy()
        self.approx_len = self.calc_approx_len()

    def calc_xy(self):
        self.section['x'] = gps.long_to_x(self.section['long'], *self.center)
        self.section['y'] = gps.lat_to_y(self.section['lat'],
                                         self.center['lat'])

    def calc_approx_len(self):
        xy = self.section[['x', 'y']]
        return np.linalg.norm(xy.iloc[0] - xy.iloc[-1])