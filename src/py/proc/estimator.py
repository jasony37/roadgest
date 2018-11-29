import numpy as np
import pandas as pd


class RoadStateEstimator(object):
    def __init__(self, road_section, timestep, start_time):
        self.time = start_time
        self.timestep = timestep
        self.road_section = road_section
        self.n_segments = len(self.road_section.segments.index)
        self.n_states = self.n_segments + self.road_section.n_ramps
        self.state = self._init_state()
        self.covar = self._init_covar()
        pass

    def _init_state(self):
        # for first density we can use flow coming into section
        # for subsequent densities use previous density, or flow if part of measurement
        return np.array([0] * self.n_states)

    def _init_covar(self):
        return np.identity(self.n_states)

    def _calc_transition_mat(self, segment_vels):
        trans = np.identity(self.n_states)
        vels = pd.Index(range(self.n_segments))
        outlet_terms = 1.0 - self.timestep / self.road_section['lengths']