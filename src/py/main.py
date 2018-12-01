import argparse
import numpy as np
import logging

import vis.core
import proc.cabdata
import proc.road
import proc.estimator
import proc.detectors


def args_setup():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('datadir', help="Path to cabspottingdata")
    args_parser.add_argument('road', help="Path to road section CSV")
    args_parser.add_argument('vdsdir', help="Path to directory containing stationary detector data")
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
    logging.basicConfig(level=logging.INFO)
    road_section = proc.road.RoadSection(args.road, args.vdsdir)
    time_lims = (1211029200, 1213089934)
    cab_data = process_cab_data(args, road_section, None)
    estimate = proc.estimator.RoadStateEstimator(road_section, 5, time_lims[0])
    # map_plotter = vis.core.MapPlotter(cab_data.cab_traces, road_section.extents)
    while estimate.time < time_lims[1]:
        estimate.run_iteration(cab_data)
        # map_plotter.plot_cabs_in_time([estimate.time - estimate.timestep + 1, estimate.time])
    # vis.core.plot_timestamps(cab_data.cab_traces, 1212991838, 1212995438)


if __name__ == "__main__":
    main()
