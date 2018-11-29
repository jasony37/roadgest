import numpy as np


class RoadStateEstimator(object):
    def __init__(self, road_section, timestep):
        self.timestep = timestep
        self.road_section = road_section
        self.n_segments = len(self.road_section.segments.index)
        self.n_states = self.n_segments + self.road_section.n_ramps
        self.state = self._init_state()
        pass

    def _init_state(self):
        return np.array([0] * self.n_states)

