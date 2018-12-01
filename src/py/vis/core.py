from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import geotiler
import numpy as np


def plot_timestamps(cab_traces, t1, t2):
    times = cab_traces['time'].between(t1, t2, inclusive=True)
    times = cab_traces[times]['time']
    print(len(times))
    plt.hist(times, bins=range(t1, t2))
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.show()


def generate_map(extents):
    bm = Basemap(llcrnrlon=extents.min['long'], llcrnrlat=extents.min['lat'],
                 urcrnrlon=extents.max['long'], urcrnrlat=extents.max['lat'],
                 projection="merc")
    bm.drawmapboundary()
    upper_x, upper_y = bm(extents.max['long'], extents.max['lat'])
    gt_map = geotiler.Map(extent=extents.to_tuple(),
                          zoom=17)
    mapimg = geotiler.render_map(gt_map)
    plt.imshow(mapimg, extent=[0, upper_x, 0, upper_y])
    return bm


def normalize_vel(velocity, vel_norm):
    vel_norm.fillna(0.01)
    vel_norm[vel_norm < 0.000001] = 0.01
    vx_norm = velocity['vx'].div(vel_norm)
    vy_norm = velocity['vy'].div(vel_norm)
    return vx_norm, vy_norm


def plot_cabs_in_time(cab_traces, times, extents):
    rel_traces = cab_traces['time'].between(times[0], times[1], inclusive=True)
    rel_traces = cab_traces[rel_traces]

    bm = generate_map(extents)
    x, y = bm(rel_traces['long'].values, rel_traces['lat'].values)

    vx_norm, vy_norm = normalize_vel(rel_traces[['vx', 'vy']], rel_traces['speed'])

    road_filt = rel_traces['segment'] >= 0
    plt.quiver(x[road_filt], y[road_filt], vx_norm[road_filt], vy_norm[road_filt],
               color='r', scale=0.05, scale_units='dots')
    road_filt = np.invert(road_filt)
    nan_filt = np.logical_or(np.isnan(vx_norm), np.isnan(vy_norm))
    filt = np.logical_and(road_filt, nan_filt)
    plt.scatter(x[filt], y[filt], c='b', s=5)
    filt = np.logical_and(road_filt, np.invert(nan_filt))
    plt.quiver(x[filt], y[filt], vx_norm[filt], vy_norm[filt],
               color='b', scale=0.05, scale_units='dots')
    plt.show()
