import numpy as np
import pandas as pd

import vis.fd


def calc_flows(data, n_segments):
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
    # vis.fd.plot_cong_bins(density, flow)
    return fit


def calc_cong_bins(data, rho_crit, q_crit):
    filt = np.logical_and(data['flow'] <= q_crit, data['density'] > rho_crit)
    data_cong = data[filt].sort_values('density')
    idxs = data_cong.index
    data_cong['bin'] = pd.Series(np.arange(len(data_cong)) // 10 + 1, index=idxs)
    grouped_bins = data_cong.groupby('bin')
    bin_densities = grouped_bins['density'].mean()
    bin_flows = grouped_bins['flow'].agg(calc_bin_flow)
    return bin_densities, bin_flows


def fit_fds(fnames):
    data = pd.concat((pd.read_csv(fname) for fname in fnames))
    data.reset_index(drop=True, inplace=True)
    n_segments = len(data.filter(regex="speed_").columns)
    flows = calc_flows(data, n_segments)
    data = pd.concat([data, flows], axis=1)
    speed_thresh = 25
    fits = []
    for i in range(n_segments):
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
            cong_bin_densities, cong_bin_flows = calc_cong_bins(cur_data, rho_crit, fit_data['flow_cap'])
            fit_data['cong_fit'] = fit_cong(cong_bin_densities, cong_bin_flows, (rho_crit, fit_data['flow_cap']))
        except np.linalg.LinAlgError:
            fit_data['cong_fit'] = None
        fits.append(fit_data)
        plot_title = "Segment {} training set".format(i + 1)
        # vis.fd.plot_fd(cur_data['density'], cur_data['flow'], fit_data, plot_title)
    return fits


def estimate_flow(density, fd_fit):
    flow_est = np.empty(len(density))
    freeflow_filt = density <= fd_fit['rho_crit']
    if np.any(freeflow_filt):
        flow_est[freeflow_filt] = density[freeflow_filt] * fd_fit['freeflow_fit'][0] + fd_fit['freeflow_fit'][1]
    cong_filt = np.invert(freeflow_filt)
    if np.any(cong_filt):
        flow_est[cong_filt] = density[cong_filt] * fd_fit['cong_fit'][0] + fd_fit['cong_fit'][1]
    return flow_est


def test_fd(fnames, fits):
    data = pd.concat((pd.read_csv(fname) for fname in fnames))
    data.reset_index(drop=True, inplace=True)
    n_segments = len(data.filter(regex="speed_").columns)
    flows = calc_flows(data, n_segments)
    data = pd.concat([data, flows], axis=1)
    for segment_num, fd_fit in enumerate(fits):
        new_col_name = 'flow_est_{}'.format(segment_num)
        density_col_name = "density_{}".format(segment_num)
        flow_col_name = "flow_{}".format(segment_num)
        data[new_col_name] = estimate_flow(data[density_col_name], fd_fit)
        cur_data = pd.DataFrame({'density': data[density_col_name],
                                 'flow': data[flow_col_name]})
        cong_bins_rho, cong_bins_q = calc_cong_bins(cur_data, fd_fit['rho_crit'], fd_fit['flow_cap'])
        title = 'Segment {} validation set'.format(segment_num + 1)
        vis.fd.plot_test_data(cur_data, fd_fit, cong_bins_rho, cong_bins_q, title)
