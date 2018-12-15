import numpy as np
from matplotlib import pyplot as plt

def plot_fd(rho, q, fit):
    """

    :param rho: segment densities
    :param q: segment flows
    :param fit:
    :return:
    """
    free_filt = fit['freeflow_filt']
    cong_filt = np.invert(free_filt)
    rho_cong, q_cong = (rho[cong_filt], q[cong_filt])
    plt.scatter(rho[free_filt], q[free_filt], color='b', s=2)
    plt.plot(rho[free_filt], fit['freeflow_fit'][0] * rho[free_filt] + fit['freeflow_fit'][1], color='b')
    plt.plot(rho, np.full(len(rho), fit['flow_cap']))
    plt.scatter(rho_cong, q_cong, color='r', s=2)
    plt.scatter([fit['rho_crit']], [fit['flow_cap']])
    if 'cong_fit' in fit and fit['cong_fit'] is not None:
        plt.plot(rho_cong, fit['cong_fit'][0] * rho_cong + fit['cong_fit'][1], color='r')
    # plt.title(str(i))
    plt.show()


def plot_cong_bins(rho, q):
    plt.scatter(rho, q, marker='s', s=5, c='purple')
    #plt.show()


def plot_test_data(data, n_segments, fits):
    for i in range(n_segments):
        fit = fits[i]
        col_rho = "density_{}".format(i)
        col_q = "flow_{}".format(i)
        plt.scatter(data[col_rho], data[col_q])

        rho_fit = [0, fit['rho_crit']]
        q_fit = [fit['freeflow_fit'][1], fit['flow_cap']]
        if fit['cong_fit'] is not None:
            rho_fit.append(-fit['cong_fit'][1] / fit['cong_fit'][0])
            q_fit.append(0)
        plt.plot(rho_fit, q_fit)
        plt.show()
