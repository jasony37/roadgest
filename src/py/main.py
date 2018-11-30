import argparse
import numpy as np

import vis.core
import proc.cabdata
import proc.road
import proc.estimator
import proc.detectors


def args_setup():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('datadir', help="Path to cabspottingdata")
    args_parser.add_argument('road', help="Path to road section CSV")
    args_parser.add_argument('vds', help="Path to directory containing stationary detector data")
    return args_parser.parse_args()


def process_cab_data(args, road_section, time_lims=None):
    cab_data = proc.cabdata.CabData(args.datadir)
    if cab_data.delta_data_exists() is False:
        cab_data.calc_xy(*road_section.center)
        cab_data.calc_deltas()
        cab_data.assign_road_segments(road_section, 11.0, np.deg2rad(30.0), time_lims)
        cab_data.save_cabtraces()
    return cab_data


def main():
    args = args_setup()
    road_section = proc.road.RoadSection(args.road)
    time_lims = (1211301600, 1211305000)
    cab_data = process_cab_data(args, road_section, None)
    estimate = proc.estimator.RoadStateEstimator(road_section, 5, time_lims[0])
    vels = cab_data.calc_avg_segment_vels(road_section.segments, (time_lims[0], time_lims[0] + 5))
    vels = np.sqrt(np.square(vels).sum(axis=1))
    # vis.core.plot_cabs_in_time(cab_data.cab_traces, time_lims, road_section.extents)
    # vis.core.plot_timestamps(cab_data.cab_traces, 1212991838, 1212995438)


if __name__ == "__main__":
    main()
