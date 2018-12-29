import numpy as np
from matplotlib import pyplot as plt


def plot_fd_triangle(fit, free=True):
    fd_plot_params = {'color': 'black', 'linewidth': 2.5}
    if free is True:
        rho = np.array([0, fit['rho_crit']])
        plt.plot(rho, fit['freeflow_fit'][0] * rho + fit['freeflow_fit'][1], **fd_plot_params)
    else:
        x_intercept = -fit['cong_fit'][1] / fit['cong_fit'][0]
        rho = np.array([fit['rho_crit'], x_intercept])
        plt.plot(rho, fit['cong_fit'][0] * rho + fit['cong_fit'][1], **fd_plot_params)


def plot_fd(rho, q, fit, title):
    """

    :param rho: segment densities
    :param q: segment flows
    :param fit:
    :return:
    """
    # font = {'fontname': 'Garamond', 'size': 14}
    free_filt = fit['freeflow_filt']
    cong_filt = np.invert(free_filt)
    rho_cong, q_cong = (rho[cong_filt], q[cong_filt])
    plt.scatter(rho[free_filt], q[free_filt], color='b', s=2)
    plot_fd_triangle(fit, free=True)
    # plt.plot(rho, np.full(len(rho), fit['flow_cap']))
    plt.scatter(rho_cong, q_cong, color='r', s=2)
    # plt.scatter([fit['rho_crit']], [fit['flow_cap']])
    if 'cong_fit' in fit and fit['cong_fit'] is not None:
        plot_fd_triangle(fit, free=False)
    plt.xlabel('Density (veh/m)')
    plt.ylabel('Flow (veh/s)')
    plt.title(title)
    plt.xlim([0, 0.5])
    plt.ylim([0, 4])
    plt.show()


def plot_cong_bins(rho, q):
    plt.scatter(rho, q, marker='s', s=5, c='purple')
    #plt.show()


def plot_test_data(data, fit, cong_rho, cong_q, title):
    # don't use existing filter in fit dict: needs to be recalculated for test data!
    free_filt = data['density'] < fit['rho_crit']
    data_free = data[free_filt]
    plt.scatter(data_free['density'], data_free['flow'], color='g', s=2)
    plt.scatter(cong_rho, cong_q, color='m', s=3, marker='^')
    plot_fd_triangle(fit, free=True)
    if fit['cong_fit'] is not None:
        plot_fd_triangle(fit, free=False)
    plt.xlabel('Density (veh/m)')
    plt.ylabel('Flow (veh/s)')
    plt.title(title)
    plt.xlim([0, 0.5])
    plt.ylim([0, 4])
    plt.show()
