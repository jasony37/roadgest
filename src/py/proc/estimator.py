import numpy as np
import pandas as pd

max_age_use_prev_speed = 1800
velocity_default = 22
measured_flow_idxs = [1, 2]


class RoadStateEstimator(object):
    def __init__(self, road_section, timestep, start_time):
        self.time = start_time
        self.timestep = timestep
        self.road_section = road_section
        self.n_segments = len(self.road_section.segments.index)
        self.n_states = self.n_segments + self.road_section.n_ramps
        self.state = self._init_state()
        self.covar = self._init_covar()
        self.prev_speeds = pd.DataFrame({'speed': np.full(self.n_segments, np.nan),
                                         'age': [0] * self.n_segments})
        self._transition_mat_template = self._calc_transition_mat_template()
        self._input_transition_mat = self._calc_input_transition_mat()
        self._state_to_meas_mat = self._calc_state_to_meas_mat(measured_flow_idxs)
        print(self._state_to_meas_mat)

    def _init_state(self):
        # for first density we can use flow coming into section
        # for subsequent densities use previous density, or flow if part of measurement
        return np.array([0] * self.n_states)

    def _init_covar(self):
        return np.identity(self.n_states)

    def _valid_prev_speeds(self, max_age):
        valid_prev_speeds = np.invert(np.isnan(self.prev_speeds['speed']))
        return np.logical_and(valid_prev_speeds, self.prev_speeds['age'] <= max_age)

    def _calc_transition_mat_template(self):
        mat_ul = np.zeros([self.n_segments, self.n_segments])
        mat_ll = np.zeros([self.road_section.n_ramps, self.n_segments])
        mat_ur = np.transpose(mat_ll).copy()
        onramp_col_mask = self.road_section.get_onramp_mask_of_nramps().reset_index(drop=True)
        mat_ur[self.road_section.segments['ramp'] == 'on', onramp_col_mask] = 1
        mat_ur[self.road_section.segments['ramp'] == 'off', np.invert(onramp_col_mask)] = -1
        mat_lr = np.identity(self.road_section.n_ramps)
        return np.block([[mat_ul, mat_ur],
                         [mat_ll, mat_lr]])

    def _calc_transition_mat(self, segment_vels):
        vels = segment_vels.reindex(pd.Index(range(self.n_segments)))
        fill_with_prev = np.logical_and(self._valid_prev_speeds(max_age_use_prev_speed),
                                        np.isnan(vels))
        vels[fill_with_prev] = self.prev_speeds['speed'][fill_with_prev]
        vels[np.isnan(vels)] = velocity_default
        outlet_terms = 1.0 - (self.timestep / self.road_section.segments['length']) * vels
        inlet_terms = self.timestep / self.road_section.segments.loc[1:, 'length']
        inlet_terms = inlet_terms.reset_index(drop=True) * vels[:-1]
        mat_ul = np.diag(outlet_terms) + np.diag(inlet_terms, k=-1)
        mat_full = self._transition_mat_template.copy()
        mat_full[:self.n_segments, :self.n_segments] = mat_ul
        return mat_full

    def _calc_input_transition_mat(self):
        mat = np.zeros([self.n_states, self.n_segments - self.road_section.n_ramps + 1])
        mat[0, 0] = self.timestep / self.road_section.segments['length'][0]
        return mat

    def _calc_state_to_meas_mat(self, meas_flow_idxs):
        mat = np.zeros([self.road_section.n_ramps, self.n_states])
        mat[np.arange(self.road_section.n_ramps - 1), meas_flow_idxs] = 1
        mat[-1, self.n_segments - 1] = 1  # flow out of last segment
        return mat

    def predict(self):

        pass
