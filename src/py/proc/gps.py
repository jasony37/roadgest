import numpy as np

earth_rad = 6.378137e6


def long_to_x(long, lat_center, long_center):
    # see https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
    lat_center_rad = np.radians(lat_center)
    return earth_rad * np.radians(long - long_center) * np.cos(lat_center_rad)


def lat_to_y(lat, lat_center):
    return earth_rad * np.radians(lat - lat_center)


def dist_sqr(p1, p2):
    vec = np.subtract(p2, p1)
    return np.sum(np.square(vec))


def p2l_dist(point, linept1, linept2):
    """
    Distance of point to line segment
    If point's project lies outside the segment, the distance is to the
    closest endpoint
    :param point: [x, y]
    :param linept1: [x, y]
    :param linept2: [x, y]
    :return:
    """
    line_vec = np.subtract(linept2, linept1)
    len_sqr = np.sum(np.square(line_vec))
    if len_sqr == 0.0:
        return dist_sqr(point, linept1)
    else:
        pt_proj_dist = np.dot(np.subtract(point, linept1), line_vec / len_sqr)
        pt_proj_dist = np.min([np.max([0.0, pt_proj_dist]), 1.0])
        pt_proj = linept1 + pt_proj_dist * line_vec
        return np.linalg.norm(pt_proj - point)

