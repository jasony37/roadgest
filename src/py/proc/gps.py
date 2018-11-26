import numpy as np
import pandas as pd

earth_rad = 6.378137e6


def long_to_x(long, lat_center, long_center):
    # see https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
    lat_center_rad = np.radians(lat_center)
    return earth_rad * np.radians(long - long_center) * np.cos(lat_center_rad)


def lat_to_y(lat, lat_center):
    return earth_rad * np.radians(lat - lat_center)


def dist_sqr(p1, p2):
    vec = p2 - p1
    return np.sum(np.square(vec), 1)


def assign_segment(loc_xy, road_section, thresh):
    dists = road_section.point_segment_dists(loc_xy).squeeze()
    idx_min = dists.idxmin()
    return idx_min if dists.loc[idx_min] <= thresh else -1


def assign_segments(cab_traces, road_section, thresh):
    """
    :param cab_traces: must contain entries 'x' and 'y' of location of interest
    :param road_section: RoadSection instance
    :param thresh: distance to segment threshold
    :return: idx of segment that should be assigned to loc,
    or -1 if loc's distance to the section is outside thresh
    """
    assignments = pd.Series([-1] * len(cab_traces.index), index=cab_traces.index)
    loc_xy = cab_traces[['x', 'y']]
    need_calcs = road_section.min_pt_dist_approx(loc_xy) <= thresh
    if np.any(need_calcs):
        assignments[need_calcs] = loc_xy[need_calcs].apply(assign_segment,
                                                           axis=1,
                                                           args=(road_section, thresh))
    return assignments
