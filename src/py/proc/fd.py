import numpy as np
import pandas as pd

import vis.fd


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
    fit = np.linalg.lstsq(A, flows.values, rcond=None)
    return fit[0]


def calc_capacity(densities, flows):
    tolerance = 0.005
    min_pts_at_max = 3
    flows_remain = flows
    while len(flows_remain) >= 1:
        cur_max = flows_remain.max()
        at_max = np.logical_and(cur_max - tolerance <= flows, flows <= cur_max + tolerance)
        if np.sum(at_max) >= min_pts_at_max:
            if (densities[at_max].max() - densities[at_max].min()) >= 0.001:
                return cur_max
        flows_remain = flows_remain[np.invert(at_max)]


def density_at_flow_cap(freeflow_fit, flow_cap):
    if abs(freeflow_fit[0]) < 0.00001:
        raise ValueError("Free-flow fit has 0 slope!")
    return (flow_cap - freeflow_fit[1]) / freeflow_fit[0]


def calc_bin_flow(flows):
    q1 = flows.quantile(q=0.25)
    q3 = flows.quantile(q=0.75)
    min_outlier_flow = q3 + 1.5*(q3-q1)
    return flows[flows < min_outlier_flow].max()


def fit_cong(density, flow, crit_pt):
    density_shifted = density - crit_pt[0]
    flow_shifted = flow - crit_pt[1]
    fit = np.linalg.lstsq(density_shifted[:, np.newaxis], flow_shifted, rcond=None)
    fit = fit[0]
    fit = np.append(fit, -fit[0] * crit_pt[0] + crit_pt[1])
    vis.fd.plot_cong_bins(density, flow)
    return fit


def calc_cong_speed_param(data, rho_crit, q_crit):
    filt = np.logical_and(data['flow'] <= q_crit, data['density'] > rho_crit)
    data_cong = data[filt].sort_values('density')
    idxs = data_cong.index
    data_cong['bin'] =  pd.Series(np.arange(len(data_cong)) // 10 + 1, index=idxs)
    grouped_bins = data_cong.groupby('bin')
    bin_densities = grouped_bins['density'].mean()
    bin_flows = grouped_bins['flow'].agg(calc_bin_flow)
    return fit_cong(bin_densities, bin_flows, (rho_crit, q_crit))


def fit_fd(fname):
    data = pd.read_csv(fname)
    flows = calc_flows(data)
    data = pd.concat([data, flows], axis=1)
    speed_thresh = 25
    for i in range(6):
        cur_data = pd.DataFrame({'speed': data['speed_{}'.format(i)],
                                 'density': data['density_{}'.format(i)],
                                 'flow': data['flow_{}'.format(i)]})
        filt = cur_data['speed'] >= speed_thresh
        data_free = cur_data[filt]
        segment_free_rho = data_free['density']
        segment_free_q = data_free['flow']
        fit_data = {}
        fit_data['freeflow_filt'] = filt
        fit_data['freeflow_fit'] = fit_freeflow(segment_free_rho, segment_free_q)
        fit_data['flow_cap'] = calc_capacity(segment_free_rho, segment_free_q)
        rho_crit = density_at_flow_cap(fit_data['freeflow_fit'], fit_data['flow_cap'])
        fit_data['rho_crit'] = rho_crit
        try:
            fit_data['cong_fit'] = calc_cong_speed_param(cur_data, rho_crit, fit_data['flow_cap'])
        except np.linalg.LinAlgError:
            pass
        vis.fd.plot_fd(cur_data['density'], cur_data['flow'], fit_data)
