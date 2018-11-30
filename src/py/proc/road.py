import numpy as np
import pandas as pd

from proc import gps


class RoadSection(object):
    def __init__(self, fname):
        self.section = pd.read_csv(fname)
        self._process_data()

    def _calc_xy(self):
        self.section['x'] = gps.long_to_x(self.section['long'], *self.center)
        self.section['y'] = gps.lat_to_y(self.section['lat'], self.center['lat'])

    def _calc_approx_len(self):
        xy = self.section[['x', 'y']]
        return np.linalg.norm(xy.iloc[0] - xy.iloc[-1])

    def _calc_segments(self):
        segments = self.section[['x', 'y']]
        ends = segments.shift(-1)
        segments = segments.assign(x2=ends['x'], y2=ends['y'])
        segments.columns = [['start', 'start', 'end', 'end'], ['x', 'y', 'x', 'y']]
        segments = segments[:-1]
        self.segments = segments

    def _form_ramp_col(self):
        ramps = self.section[['on_ramp', 'off_ramp']].shift(-1)
        ramps = ramps[:-1]
        assert not np.any(ramps.sum(axis=1) > 1)
        col = pd.Series(['none'] * len(ramps.index))
        col[ramps['on_ramp'] == 1] = 'on'
        col[ramps['off_ramp'] == 1] = 'off'
        return col

    def _calc_segment_props(self):
        vecs = self.segments['end'] - self.segments['start']
        self.segments['angle'] = np.arctan2(vecs['y'], vecs['x'])
        self.segments['length'] = np.sqrt(np.sum(np.square(vecs), 1))
        self.segments['ramp'] = self._form_ramp_col()
        self.n_ramps = np.sum(self.segments['ramp'] != 'none')

    def _process_data(self):
        self.center = np.mean(self.section[['lat', 'long']])
        self.extents = gps.Extents(self.section[['lat', 'long']], 0.0005, 0.0005)
        self._calc_xy()
        self._calc_segments()
        self._calc_segment_props()
        self.approx_len = self._calc_approx_len()

    def get_ramp_indexes(self):
        return self.segments['ramp'].index[self.segments['ramp'] != 'none']

    def get_onramp_mask_of_nramps(self):
        return self.segments['ramp'][self.get_ramp_indexes()] == 'on'

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
