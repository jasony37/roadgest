from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import geotiler


def plot_timestamps(cab_traces, t1, t2):
    times = cab_traces['time'].between(t1, t2, inclusive=True)
    times = cab_traces[times]['time']
    print(len(times))
    plt.hist(times, bins=range(t1, t2))
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.show()


def plot_cabs_in_time(cab_traces, time1, time2):
    rel_traces = cab_traces['time'].between(time1, time2, inclusive=True)
    rel_traces = cab_traces[rel_traces]
    extent = (-122.6, 37.7, -122.28, 37.8)

    bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[1],
                 urcrnrlon=extent[2], urcrnrlat=extent[3],
                 projection="merc")
    bm.drawmapboundary()
    x, y = bm(rel_traces['long'].values, rel_traces['lat'].values)
    extent_x, extent_y = bm(extent[2], extent[3])

    gt_map = geotiler.Map(extent=extent,
                          zoom=14)
    mapimg = geotiler.render_map(gt_map)
    plt.imshow(mapimg, extent=[0, extent_x, 0, extent_y])
    plt.scatter(x[rel_traces['segment'] >= 0], y[rel_traces['segment'] >= 0], color='r', s=5)
    plt.scatter(x[rel_traces['segment'] < 0], y[rel_traces['segment'] < 0], color='b', s=5)
    plt.show()
