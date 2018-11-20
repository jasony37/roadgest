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

    def check_on_section(self, road_section, dist_thresh, time_lims=None):
        """
        For each cab in self.cab_traces, data must previously be sorted by
        increasing time
        :param road_section:
        :param dist_thresh
        :return:
        """
        cab_traces = self.cab_traces
        if time_lims is not None:
            rel_rows = cab_traces['time'].between(time_lims[0], time_lims[1])
            cab_traces = cab_traces[rel_rows]
        seg_assn = gps.assign_segments(cab_traces, road_section, dist_thresh)
        if time_lims is not None:
            seg_assn = pd.DataFrame({'segment': seg_assn})
            seg_assn.set_index(self.cab_traces.index[rel_rows], inplace=True)
        self.cab_traces['segment'] = seg_assn
        # self.cab_traces.groupby('cab_id').first()
        pass


class RoadSection(object):
    def __init__(self, fname):
        self.section = pd.read_csv(fname)
        self.center = np.mean(self.section[['lat', 'long']])
        self.calc_xy()
        self.segments = self.calc_segments()
        self.approx_len = self.calc_approx_len()

    def calc_xy(self):
        self.section['x'] = gps.long_to_x(self.section['long'], *self.center)
        self.section['y'] = gps.lat_to_y(self.section['lat'], self.center['lat'])

    def calc_approx_len(self):
        xy = self.section[['x', 'y']]
        return np.linalg.norm(xy.iloc[0] - xy.iloc[-1])

    def calc_segments(self):
        xy_labels = ['x', 'y']
        segments = self.section[xy_labels]
        ends = segments.shift(-1)
        segments = segments.assign(x2=ends['x'], y2=ends['y'])
        segments.columns = [['start', 'start', 'end', 'end'], xy_labels + xy_labels]
        segments = segments[:-1]
        return segments

    def min_pt_dist_approx(self, points):
        sec_start = self.section.loc[0, ['x', 'y']]
        sec_start = sec_start.squeeze()  # turn into Series
        pt_to_start_dists = np.square(points - sec_start)
        pt_to_start_dists = np.sqrt(pt_to_start_dists.sum(axis=1))
        min_dists = pt_to_start_dists - self.approx_len
        return min_dists.clip(lower=0.0)

    def point_segment_dists(self, point, filt=None):
        """
        point_segment_dists
        Find point-to-line-segment distance to each segment in the road section
        :param point: Container that has x, y of point
        :param filt: Boolean array: only find distances of segments where
        its corresponding value in filt is True
        :return: Pandas Series of distances between the point and each segment
        """
        dists = pd.Series([0] * len(self.segments['start'].index))
        line_vec = self.segments['end'] - self.segments['start']
        len_sqr = np.sum(np.square(line_vec), 1)
        if filt is None:
            point = point.squeeze()
            filt = len_sqr == 0.0
            if np.any(filt):
                dists[filt] = gps.dist_sqr(point, self.segments['start'][filt])
            filt = np.invert(filt)
            if np.any(filt):
                dists[filt] = self.point_segment_dists(point, filt)
            return dists.squeeze()
        else:
            starts = self.segments['start'][filt]
            starts_to_point = point - starts
            pt_proj_dists = starts_to_point.multiply(line_vec.div(len_sqr, axis=0), axis=0)
            pt_proj_dists = pt_proj_dists.sum(1)
            pt_proj_dists.clip(0.0, 1.0, inplace=True)
            pt_projs = starts + line_vec.multiply(pt_proj_dists, axis=0)
            return np.sqrt(gps.dist_sqr(point, pt_projs))
