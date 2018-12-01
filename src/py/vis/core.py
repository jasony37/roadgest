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


def normalize_vel(velocity, vel_norm):
    vel_norm.fillna(0.01)
    vel_norm[vel_norm < 0.000001] = 0.01
    vx_norm = velocity['vx'].div(vel_norm)
    vy_norm = velocity['vy'].div(vel_norm)
    return vx_norm, vy_norm


class MapPlotter(object):
    def __init__(self, cab_traces, extents):
        self.extents = extents
        self._upper_coords = (None, None)
        self.mapimg = None
        self.bm = self.generate_map()
        self.cab_traces = cab_traces
        self.scatter = None
        self.quiver = None

    def generate_map(self):
        extents = self.extents
        bm = Basemap(llcrnrlon=extents.min['long'], llcrnrlat=extents.min['lat'],
                     urcrnrlon=extents.max['long'], urcrnrlat=extents.max['lat'],
                     projection="merc")
        bm.drawmapboundary()
        self._upper_coords = bm(extents.max['long'], extents.max['lat'])
        gt_map = geotiler.Map(extent=extents.to_tuple(),
                              zoom=17)
        self.mapimg = geotiler.render_map(gt_map)
        return bm

    def plot_cabs_in_time(self, times):
        if self.scatter is not None:
            self.scatter.remove()
        if self.quiver is not None:
            self.quiver.remove()
        rel_traces = self.cab_traces['time'].between(times[0], times[1], inclusive=True)
        rel_traces = self.cab_traces[rel_traces]

        x, y = self.bm(rel_traces['long'].values, rel_traces['lat'].values)

        vx_norm, vy_norm = normalize_vel(rel_traces[['vx', 'vy']], rel_traces['speed'])

        road_filt = rel_traces['segment'] >= 0
        plt.quiver(x[road_filt], y[road_filt], vx_norm[road_filt], vy_norm[road_filt],
                   color='r', scale=0.05, scale_units='dots')
        road_filt = np.invert(road_filt)
        nan_filt = np.logical_or(np.isnan(vx_norm), np.isnan(vy_norm))
        filt = np.logical_and(road_filt, nan_filt)
        plt.imshow(self.mapimg, extent=[0, self._upper_coords[0], 0, self._upper_coords[1]])
        plt.xlim([0, self._upper_coords[0]])
        plt.ylim([0, self._upper_coords[1]])
        self.scatter = plt.scatter(x[filt], y[filt], c='b', s=5)
        filt = np.logical_and(road_filt, np.invert(nan_filt))
        self.quiver = plt.quiver(x[filt], y[filt], vx_norm[filt], vy_norm[filt],
                                color='b', scale=0.05, scale_units='dots')
        plt.show()