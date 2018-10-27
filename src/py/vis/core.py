from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap


def plot_cabs_in_time(cab_traces, time1, time2):
    rel_traces = cab_traces['time'].between(time1, time2, inclusive=True)
    rel_traces = cab_traces[rel_traces]
    bm = Basemap(llcrnrlon=-122.5515, llcrnrlat=37.6898,
                 urcrnrlon=-122.2820, urcrnrlat=37.8096,
                 resolution='h', projection="merc")
    x, y = bm(rel_traces['long'].values, rel_traces['lat'].values)
    bm.scatter(x, y, s=3)
    plt.show()