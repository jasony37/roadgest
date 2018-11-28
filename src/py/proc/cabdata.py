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

    def calc_deltas(self, dtime_max=90):
        """
        :param dtime_max: Maximum delta time for valid velocity, in seconds
        :return:
        """
        cab_traces_grouped = self.cab_traces.groupby('cab_id')[['time', 'x', 'y']]
        delta = cab_traces_grouped.shift(0) - cab_traces_grouped.shift(1)
        delta.loc[delta['time'] > dtime_max, 'time'] = np.nan
        self.cab_traces['vx'] = delta['x'] / delta['time']
        self.cab_traces['vy'] = delta['y'] / delta['time']
        self.cab_traces['dir'] = np.arctan2(self.cab_traces['vy'], self.cab_traces['vx'])

    def assign_road_segments(self, road_section, dist_thresh, angle_thresh, time_lims=None):
        """
        For each cab in self.cab_traces, data must previously be sorted by
        increasing time
        :param road_section: RoadSection instance
        :param dist_thresh: threshold of distance to road segment, in meters
        :param angle_thresh: threshold of gps direction to road angle, in radians
        :param time_lims: tuple (start, end) between which to assign segments
        :return:
        """
        cab_traces = self.cab_traces
        if time_lims is not None:
            rel_rows = cab_traces['time'].between(time_lims[0], time_lims[1])
        else:
            rel_rows = cab_traces['time']
        cab_traces = cab_traces[rel_rows]
        seg_assn = gps.assign_segments(cab_traces, road_section, dist_thresh, angle_thresh)
        seg_assn = pd.DataFrame({'segment': seg_assn})
        seg_assn.set_index(self.cab_traces.index[rel_rows], inplace=True)
        self.cab_traces['segment'] = seg_assn
        # self.cab_traces.groupby('cab_id').first()
        pass
