from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import geotiler


def plot_cabs_in_time(cab_traces, time1, time2):
    rel_traces = cab_traces['time'].between(time1, time2, inclusive=True)
    rel_traces = cab_traces[rel_traces]
    extent = (-122.5, 37.6663, -122.37, 37.7377)

    bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[1],
                 urcrnrlon=extent[2], urcrnrlat=extent[3],
                 projection="merc")
    bm.drawmapboundary()
    x, y = bm(rel_traces['long'].values, rel_traces['lat'].values)
    extent_x, extent_y = bm(extent[2], extent[3])

    gt_map = geotiler.Map(extent=extent,
                          zoom=16)
    mapimg = geotiler.render_map(gt_map)
    plt.imshow(mapimg, extent=[0, extent_x, 0, extent_y])
    plt.scatter(x, y, s=5)
    plt.show()
