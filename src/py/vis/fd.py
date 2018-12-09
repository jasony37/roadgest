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
    if 'cong_fit' in fit:
        plt.plot(rho_cong, fit['cong_fit'][0] * rho_cong + fit['cong_fit'][1], color='r')
    # plt.title(str(i))
    plt.show()

def plot_cong_bins(rho, q):
    plt.scatter(rho, q, marker='s', s=5, c='purple')
    #plt.show()