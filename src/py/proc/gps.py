import numpy as np
import pandas as pd

earth_rad = 6.378137e6 # radius of Earth in meters


def long_to_x(long, lat_center, long_center):
    # see https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
    lat_center_rad = np.radians(lat_center)
    return earth_rad * np.radians(long - long_center) * np.cos(lat_center_rad)


def lat_to_y(lat, lat_center):
    return earth_rad * np.radians(lat - lat_center)


def dist_sqr(p1, p2):
    vec = p2 - p1
    return np.sum(np.square(vec), 1)


def assign_segment(loc_xy, road_section, dist_thresh, angle_thresh):
    dists = road_section.point_segment_dists(loc_xy).squeeze()
    idx_min = dists.idxmin()
    min_dist = dists.loc[idx_min]
    if min_dist <= dist_thresh:
        angle_diff = np.abs(loc_xy['dir'] - road_section.segments['angle'][idx_min])
        if angle_diff <= angle_thresh:
            return idx_min
    return -1


def assign_segments(cab_traces, road_section, dist_thresh, angle_thresh):
    """
    :param cab_traces: must contain entries 'x' and 'y' of location of interest
    :param road_section: RoadSection instance
    :param dist_thresh: distance to segment threshold
    :param angle_thresh: threshold of gps direction to road angle, in radians
    :return: idx of segment that should be assigned to loc,
    or -1 if loc's distance to the section is outside thresh
    """
    assignments = pd.Series([-1] * len(cab_traces.index), index=cab_traces.index)
    loc_xy = cab_traces[['x', 'y', 'dir']]
    need_calcs = road_section.min_pt_dist_approx(loc_xy) <= dist_thresh
    if np.any(need_calcs):
        assignments[need_calcs] = loc_xy[need_calcs].apply(assign_segment,
                                                           axis=1,
                                                           args=(road_section, dist_thresh,
                                                                 angle_thresh))
    return assignments
