import numpy as np
import pandas as pd
import logging

max_age_use_prev_speed = 120
max_speed_plausible = 42.5
velocity_default = pd.Series([31.3, 29.2, 30.1, 30.1, 29.5, 27.4])
vel_var_default = pd.Series([2.18, 0.83, 1.63, 1.63, 2.84, 2.84])
measured_flow_idxs = [1, 2]
gps_pos_var = 3.0**2
gps_interval = 60  # time between measurements, in seconds
typ_segment_density = 0.04  # typical segment density, in vehicles/meter
density_process_noise = 2.5e-5  # expected noise in segment density in 1 sec
ramp_density_init_val = 0.005
ramp_density_process_noise = 1e-5


class RoadSpeedBuffer(object):
    def __init__(self, n_segments):
        self.data = pd.DataFrame({'speed': np.full(n_segments, np.nan),
                                  'age': [0] * n_segments})

    def get_valid_speeds_filt(self, max_age):
        valid_prev_speeds = np.invert(np.isnan(self.data['speed']))
        return np.logical_and(valid_prev_speeds, self.data['age'] <= max_age)

    def valid_speeds(self, max_age):
        tmp = self.data['speed'].copy()
        tmp[np.invert(self.get_valid_speeds_filt(max_age))]['speed'] = np.nan
        return tmp

    def update(self, cur_speeds):
        avail_idxs = list(cur_speeds.index)
        self.data['age'] += 1
        self.data.loc[avail_idxs, 'speed'] = cur_speeds
        self.data.loc[avail_idxs, 'age'] = 0


class RoadStateEstimator(object):
    def __init__(self, road_section, timestep, start_time):
        self.logger = logging.getLogger("RoadStateEstimator")
        self.time = start_time
        self.timestep = timestep
        self.road_section = road_section
        self.n_segments = len(self.road_section.segments.index)
        self.n_states = self.n_segments + self.road_section.n_ramps
        self.state = self._init_state()
        self.error_covar = self._init_error_covar()
        self.speed_buffer = RoadSpeedBuffer(self.n_segments)
        self._cur_speeds = None
        self._transition_mat_template = self._calc_transition_mat_template()
        self._input_transition_mat = self._calc_input_transition_mat()
        self._state_to_meas_mat = self._calc_state_to_meas_mat(measured_flow_idxs)
        self._default_process_covar = self._init_def_process_covar()
        self._measurement_covar = self._calc_measurement_covar()

    def _init_state(self):
        # for first density we can use flow coming into section
        # for subsequent densities use previous density, or flow if part of measurement
        state = np.full(self.n_states, np.nan)
        densities = self.road_section.calc_density_meas_at_time(self.time)
        if len(densities) > 0:
            idx1 = np.array(densities.index)
            idx1[1:] -= 1
            idx2 = idx1[1:]
            idx1 = idx1[:-1]
            densities = np.array(densities)
            for ii in range(len(idx1)):
                state[idx1[ii]:idx2[ii]] = densities[ii]
            state[self.n_segments - 1] = densities[-1]
        state[self.n_segments:] = ramp_density_init_val
        self._increment_time()
        return state

    def _init_error_covar(self):
        segment_terms = np.full(self.n_segments, (typ_segment_density / 5.0)**2)
        ramp_terms = np.full(self.road_section.n_ramps, (ramp_density_init_val / 2.0)**2)
        return np.diag(np.concatenate((segment_terms, ramp_terms)))

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

    def _calc_cur_speeds(self, segment_speeds):
        speeds = segment_speeds.reindex(pd.Index(range(self.n_segments)))
        fill_with_prev = np.logical_and(self.speed_buffer.get_valid_speeds_filt(max_age_use_prev_speed),
                                        np.isnan(speeds))
        speeds[fill_with_prev] = self.speed_buffer.data['speed'][fill_with_prev]
        self._cur_speeds = speeds

    def _calc_transition_mat(self):
        speeds = self._cur_speeds.copy()
        speeds[np.isnan(speeds)] = velocity_default
        outlet_terms = 1.0 - (self.timestep / self.road_section.segments['length']) * speeds
        inlet_terms = self.timestep / self.road_section.segments.loc[1:, 'length']
        inlet_terms = inlet_terms.reset_index(drop=True) * speeds[:-1]
        mat_ul = np.diag(outlet_terms) + np.diag(inlet_terms, k=-1)
        mat_full = self._transition_mat_template.copy()
        mat_full[:self.n_segments, :self.n_segments] = mat_ul
        return mat_full

    def _calc_input_mat(self):
        detector_first = self.road_section.section['vds_data'][0]
        flow_in = detector_first.calc_val_at_time('flow', self.time)
        u = np.zeros(self.n_segments - self.road_section.n_ramps + 1)
        u[0] = flow_in
        return u

    def _calc_input_transition_mat(self):
        mat = np.zeros([self.n_states, self.n_segments - self.road_section.n_ramps + 1])
        mat[0, 0] = self.timestep / self.road_section.segments['length'][0]
        return mat

    def _calc_state_to_meas_mat(self, meas_flow_idxs):
        mat = np.zeros([self.road_section.n_ramps, self.n_states])
        mat[np.arange(self.road_section.n_ramps - 1), meas_flow_idxs] = 1
        mat[-1, self.n_segments - 1] = 1  # flow out of last segment
        return mat

    def _init_def_process_covar(self):
        gps_vel_var = (2.0 / gps_interval**2) * gps_pos_var
        flow_terms = self.timestep / self.road_section.segments['length'] * typ_segment_density
        flow_terms = np.square(flow_terms) * gps_vel_var
        flow_terms[1:] *= 2.0
        density_noise_var = (self.timestep * density_process_noise)**2
        density_terms = np.full(self.n_segments, density_noise_var)
        ramp_noise_var = (self.timestep * ramp_density_process_noise)**2
        ramp_terms = np.zeros(self.n_segments)
        ramp_terms[self.road_section.segments['ramp'] != 'none'] = ramp_noise_var
        vars_total = flow_terms + density_terms + ramp_terms
        vars_total = np.append(vars_total, [ramp_noise_var] * self.road_section.n_ramps)
        return np.diag(vars_total)

    def _calc_process_covar(self):
        covar = self._default_process_covar
        flow_add = np.zeros(self.n_segments)
        filt = np.isnan(self._cur_speeds)
        flow_add[filt] = self.timestep / self.road_section.segments['length'][filt] * typ_segment_density
        flow_add[filt] = np.square(flow_add[filt]) * vel_var_default[filt]
        flow_add = np.append(flow_add, np.zeros(self.road_section.n_ramps))
        covar += np.diag(flow_add)
        return covar


    def _calc_measurement_covar(self):
        return np.diag([3.12e-4, 6.82e-4, 2.70e-4])

    def _calc_kalman_gain(self):
        H = self._state_to_meas_mat
        Ht = H.transpose()
        P = self.error_covar
        R = self._measurement_covar
        numerator = np.matmul(P, Ht)
        denominator = np.matmul(np.matmul(H, P), Ht) + R
        return np.matmul(numerator, np.linalg.inv(denominator))

    def _increment_time(self):
        self.time += self.timestep

    def _limit_state_to_positive(self):
        self.state = np.clip(self.state, 0.0, None)

    def predict(self, segment_speeds):
        self._calc_cur_speeds(segment_speeds)
        A = self._calc_transition_mat()
        B = self._input_transition_mat
        u = self._calc_input_mat()
        self.state = np.matmul(A, self.state) + np.matmul(B, u)
        self._limit_state_to_positive()
        self.error_covar = np.matmul(np.matmul(A, self.error_covar), A.transpose())
        self.error_covar += self._calc_process_covar()

    def update(self, meas):
        K = self._calc_kalman_gain()
        innov = meas - np.matmul(self._state_to_meas_mat, self.state)
        self.state += np.matmul(K, innov)
        self._limit_state_to_positive()
        I = np.identity(self.n_states)
        error_covar_factor = I - np.matmul(K, self._state_to_meas_mat)
        self.error_covar = np.matmul(error_covar_factor, self.error_covar)

    def run_iteration(self, cab_data):
        self.logger.info("========= ITERATION @ {} ==========".format(self.time + self.timestep))
        times = [self.time - self.timestep + 1, self.time]
        segment_speeds = cab_data.calc_avg_segment_speeds(self.road_section.segments, times)
        segment_speeds[segment_speeds > max_speed_plausible] = np.nan
        self.predict(segment_speeds)
        self.logger.info("Non-default speeds used in meas: {}".format(self._cur_speeds))
        self.logger.info("Predicted state: {}".format(self.state))
        self._increment_time()
        self.speed_buffer.update(segment_speeds)
        meas = self.road_section.calc_density_meas_at_time(self.time)
        self.update(np.array(meas)[1:])
        self.logger.info("Updated state: {}".format(self.state))