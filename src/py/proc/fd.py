import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def calc_flows(data):
    densities = data.filter(regex="density_")
    speeds = data.filter(regex="speed_")
    n_segments = len(speeds.columns)
    rel_density_cols = ['density_{}'.format(i) for i in range(n_segments)]
    flows = np.multiply(densities[rel_density_cols], speeds)
    flows.columns = ['flow_{}'.format(i) for i in range(n_segments)]
    return flows


def fit_fd(fname):
    data = pd.read_csv(fname)
    flows = calc_flows(data)
    data = pd.concat([data, flows], axis=1)
    speed_thresh = 25
    for i in range(6):
        cur_valid = data[data['flow_{}'.format(i)] < 3.0]
        filt = cur_valid['speed_{}'.format(i)] >= speed_thresh
        data_free = cur_valid[filt]
        segment_densities = data_free['density_{}'.format(i)]
        segment_flows = data_free['flow_{}'.format(i)]
        A = np.vstack([segment_densities.values, np.ones(len(segment_densities))]).T
        fit = np.linalg.lstsq(A, segment_flows.values)
        data_remain = cur_valid[np.invert(filt)]
        plt.scatter(segment_densities, segment_flows, color='r', s=2)
        plt.plot(segment_densities, fit[0][0] * segment_densities + fit[0][1], color='r')
        plt.scatter(data_remain['density_{}'.format(i)], data_remain['flow_{}'.format(i)], color='g', s=2)
        plt.title(str(i))
        plt.show()