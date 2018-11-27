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


def plot_cabs_in_time(cab_traces, times, extents):
    rel_traces = cab_traces['time'].between(times[0], times[1], inclusive=True)
    rel_traces = cab_traces[rel_traces]
    extent = (-122.6, 37.7, -122.28, 37.8)

    bm = Basemap(llcrnrlon=extents.min['long'], llcrnrlat=extents.min['lat'],
                 urcrnrlon=extents.max['long'], urcrnrlat=extents.max['lat'],
                 projection="merc")
    bm.drawmapboundary()
    x, y = bm(rel_traces['long'].values, rel_traces['lat'].values)
    upper_x, upper_y = bm(extents.max['long'], extents.max['lat'])

    gt_map = geotiler.Map(extent=extents.to_tuple(),
                          zoom=16)
    mapimg = geotiler.render_map(gt_map)
    plt.imshow(mapimg, extent=[0, upper_x, 0, upper_y])
    plt.scatter(x[rel_traces['segment'] >= 0], y[rel_traces['segment'] >= 0], color='r', s=5)
    plt.scatter(x[rel_traces['segment'] < 0], y[rel_traces['segment'] < 0], color='b', s=5)
    plt.show()
