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


def fit_freeflow(densities, flows):
    A = np.vstack([densities.values, np.ones(len(densities))]).T
    fit = np.linalg.lstsq(A, flows.values)
    return fit[0]


def calc_capacity(flows):
    tolerance = 0.005
    min_pts_at_max = 3
    flows_remain = flows
    while len(flows_remain) >= 1:
        cur_max = flows_remain.max()
        at_max = np.logical_and(cur_max - tolerance <= flows, flows <= cur_max + tolerance)
        if np.sum(at_max) >= min_pts_at_max:
            return cur_max
        else:
            flows_remain = flows_remain[np.invert(at_max)]


def fit_fd(fname):
    data = pd.read_csv(fname)
    flows = calc_flows(data)
    data = pd.concat([data, flows], axis=1)
    speed_thresh = 25
    for i in range(6):
        cur_valid = data
        filt = cur_valid['speed_{}'.format(i)] >= speed_thresh
        data_free = cur_valid[filt]
        segment_densities = data_free['density_{}'.format(i)]
        segment_flows = data_free['flow_{}'.format(i)]
        freeflow_fit = fit_freeflow(segment_densities, segment_flows)
        flow_cap = calc_capacity(segment_flows)
        data_remain = cur_valid[np.invert(filt)]
        plt.scatter(segment_densities, segment_flows, color='r', s=2)
        plt.plot(segment_densities, freeflow_fit[0] * segment_densities + freeflow_fit[1], color='r')
        plt.plot(segment_densities, np.full(len(segment_densities), flow_cap))
        plt.scatter(data_remain['density_{}'.format(i)], data_remain['flow_{}'.format(i)], color='g', s=2)
        plt.title(str(i))
        plt.show()
